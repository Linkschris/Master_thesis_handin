import gurobipy as gp
from gurobipy import GRB
import numpy as np
from data import data_loader, rbf_kernel_function, compute_kernel_matrix, regularized_poly_kernel_function
from omegaconf import OmegaConf

from electrolyser_efficiency import get_efficiency

# import wandb
# from dotenv import load_dotenv

# from electrolyzer import electrolyzer_efficiency
from itertools import chain


def flatten_mvar(mvar):
    nested = mvar.tolist()
    while isinstance(nested[0], list):
        nested = list(chain.from_iterable(nested))
    return nested


class expando(object):
    """
    A small class which can have attributes set
    """

    pass


class HindsightModel:
    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        model_param = OmegaConf.load(f"configs/{configfile_name}.yaml")

        # model_param = main_electrolyzer(model_param)

        OmegaConf.save(model_param, f"configs/{configfile_name}.yaml")
        self.data = data

        self.model_param = model_param
        self.timelimit = timelimit
        self.wind_capacity = self.model_param["max_wind_capacity"]
        # self.electrolyzer_capacity = self.model_param["max_electrolyzer_capacity"]
        self.price_quantile = self.model_param["price_quantile"]

        self.data["energy_RE"] = self.data["energy_RE"] * self.wind_capacity
        print("energy_RE mean after scaling:", self.data["energy_RE"].mean())

        self.data["intercept"] = 1

        self.S = np.array([t for t in range(0, self.model_param["N_s"])])

        self.s2_max = get_efficiency(self.model_param)
        # Gurobi model
        self.model = gp.Model("Model")
        self.model.Params.TimeLimit = self.timelimit
        self.model.Params.Seed = 42  # Reproducibility

        # Common parameters
        self.hours = len(self.data)
        self.days = int(self.hours / 24)
        self.lambda_DA_RE = self.data["lambda_DA_RE"].to_numpy()
        self.lambda_DA_FC = self.data["lambda_DA_FC"].to_numpy()
        self.lambda_IM = self.data["lambda_IM"].to_numpy()
        self.energy_RE = self.data["energy_RE"].to_numpy()
        self.energy_FC = self.data["energy_FC"].to_numpy() * self.wind_capacity
        self.lambda_FC_normalized = self.data["lambda_DA_FC_normalized"].to_numpy()

        # ------------------------------------------------------------------------------------------------------
        # Quantile of lambda_DA_RE
        new_price_quantile = [float("-inf")]
        if self.price_quantile is None:
            new_price_quantile.append(15.44133708970593 * self.model_param["hydrogen_price"])
        elif not self.price_quantile:
            pass
        else:
            for p in self.price_quantile:
                # get quantile of lambda_DA_RE
                new_price_quantile.append(np.quantile(self.lambda_DA_RE, p))
        new_price_quantile.append(float("inf"))

        self.price_quantile = new_price_quantile
        print("Price quantile: ", self.price_quantile)
        self.n_price_domains = len(self.price_quantile) - 1

        self.variables = expando()  # Store variables
        self.constraints = expando()  # Store constraints
        self.results = expando()  # Store results
        self.results.objective_value = None

    def _initialize_variables(self):
        """Initialize base variables for optimization."""
        self.variables.volume_DA = self.model.addMVar(
            shape=self.hours,
            lb=-self.model_param["max_electrolyzer_capacity"],
            ub=self.model_param["max_wind_capacity"],
            vtype=GRB.CONTINUOUS,
            name="volume_DA",
        )

        # self.variables.consumption_electrolyzer = self.model.addMVar(
        #     shape=self.hours,
        #     lb=0,
        #     ub=self.model_param["max_electrolyzer_capacity"],
        #     vtype=GRB.CONTINUOUS,
        #     name="consumption_electrolyzer",
        # )

        self.variables.volume_IM = self.model.addMVar(
            shape=self.hours,
            lb=-1 * (self.model_param["max_wind_capacity"] + self.model_param["max_electrolyzer_capacity"]),
            ub=self.model_param["max_wind_capacity"] + self.model_param["max_electrolyzer_capacity"],
            vtype=GRB.CONTINUOUS,
            name="volume_IM",
        )

        self.variables.curtailment = self.model.addMVar(
            shape=self.hours, lb=0, ub=self.model_param["max_wind_capacity"], vtype=GRB.CONTINUOUS, name="curtailment"
        )

        # self.variables.hydrogen_output = self.model.addMVar(
        #     shape=self.hours,
        #     lb=0,
        #     vtype=GRB.CONTINUOUS,
        #     name="hydrogen_output",
        # )

        # self.variables.electrolyzer_on = self.model.addMVar(
        #     shape=self.hours, lb=0, ub=1, vtype=GRB.BINARY, name="electrolyzer_on"
        # )
        # self.variables.electrolyzer_off = self.model.addMVar(
        #     shape=self.hours, lb=0, ub=1, vtype=GRB.BINARY, name="electrolyzer_off"
        # )
        # self.variables.electrolyzer_standby = self.model.addMVar(
        #     shape=self.hours, lb=0, ub=1, vtype=GRB.BINARY, name="electrolyzer_standby"
        # )
        # self.variables.electrolyzer_cold_start = self.model.addMVar(
        #     shape=self.hours, lb=0, ub=1, vtype=GRB.BINARY, name="electrolyzer_cold_start"
        # )
        # self.variables.intended_consumed_electricity = self.model.addMVar(
        #     shape=self.hours, lb=0, vtype=GRB.CONTINUOUS, name="intended_consumed_electricity"
        # )

        self.variables.reg_term = self.model.addMVar(
            shape=1,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="reg_term",
        )

    def _set_constraints(self):
        """Define basic constraints applicable to all models."""
        self.constraints.power_imbalance = self.model.addConstr(
            self.energy_RE
            >= self.variables.volume_DA
            + self.variables.volume_IM
            #    + self.variables.consumption_electrolyzer
            + self.variables.curtailment,
            name="power_imbalance",
        )

        # self.constraints.minimum_daily_hydrogen_production = self.model.addConstrs(
        #     (
        #         gp.quicksum(self.variables.hydrogen_output[t] for t in range(int(24 * d), int(24 * (d + 1))))
        #         >= self.model_param["min_daily_hydrogen_production"] - 1
        #         for d in range(self.days)
        #     ),
        #     name="minimum_daily_hydrogen_production",
        # )

    #
    # self.constraints.max_electrolyzer_consumption = self.model.addConstr(
    #     self.variables.consumption_electrolyzer
    #     <= self.model_param["max_electrolyzer_capacity"] * self.variables.electrolyzer_on
    #     + self.variables.electrolyzer_standby
    #     * self.model_param["max_electrolyzer_capacity"]
    #     * self.model_param["p_standby"],
    #     name="max_electrolyzer_consumption",
    # )
    #
    # self.constraints.min_electrolyzer_consumption = self.model.addConstr(
    #     self.variables.consumption_electrolyzer
    #     >= self.model_param["p_min"]
    #     * self.model_param["max_electrolyzer_capacity"]
    #     * self.variables.electrolyzer_on
    #     + self.variables.electrolyzer_standby
    #     * self.model_param["p_standby"]
    #     * self.model_param["max_electrolyzer_capacity"],
    #     name="min_electrolyzer_consumption",
    # )
    #
    # self.constraints.no_cold_start = self.model.addConstr(
    #     self.variables.electrolyzer_cold_start[0] == 0, name="no_cold_start"
    # )
    #
    # self.constraints.always_on = self.model.addConstr(self.variables.electrolyzer_on == 1, name="always_on")
    #
    # self.constraints.electrolyzer_operation = self.model.addConstr(
    #     self.variables.electrolyzer_on + self.variables.electrolyzer_off + self.variables.electrolyzer_standby == 1,
    #     name="electrolyzer_operation",
    # )
    #
    # self.constraints.cold_start_detection = self.model.addConstrs(
    #     (
    #         self.variables.electrolyzer_cold_start[t]
    #         >= self.variables.electrolyzer_off[t - 1]
    #         + self.variables.electrolyzer_on[t]
    #         + self.variables.electrolyzer_standby[t]
    #         - 1
    #         for t in range(self.hours)[1:]
    #     ),
    #     name="cold_start_detection",
    # )
    #
    # self.constraints.hydrogen_production = self.model.addConstrs(
    #     (
    #         (
    #             self.variables.hydrogen_output[t]
    #             <= (
    #                 self.variables.intended_consumed_electricity[t] * self.model_param["a"][int(s)]
    #                 + self.variables.electrolyzer_on[t] * self.model_param["b"][int(s)]
    #             )
    #             * self.model_param["eta_storage"]
    #         )
    #         for t in range(self.hours)
    #         for s in self.S
    #     ),
    #     name="hydrogen_production",
    # )
    #
    # self.constraints.min_intended_electrolyser_consumption = self.model.addConstr(
    #     self.variables.intended_consumed_electricity
    #     >= self.model_param["p_min"]
    #     * self.model_param["max_electrolyzer_capacity"]
    #     * self.variables.electrolyzer_on,
    #     name="min_intended_electrolyser_consumption",
    # )
    #
    # self.constraints.max_intended_electrolyser_consumption = self.model.addConstr(
    #     self.variables.intended_consumed_electricity
    #     <= self.model_param["max_electrolyzer_capacity"] * self.variables.electrolyzer_on,
    #     name="max_intended_electrolyser_consumption",
    # )
    #
    # self.constraints.electrolyzer_constraint = self.model.addConstr(
    #     self.variables.consumption_electrolyzer
    #     == self.variables.intended_consumed_electricity
    #     + self.variables.electrolyzer_standby
    #     * self.model_param["p_standby"]
    #     * self.model_param["max_electrolyzer_capacity"],
    #     name="electrolyzer_constraint",
    # )

    def _set_objective(self):
        """Set the objective function (base model)."""
        base_revenue = gp.quicksum(
            self.lambda_DA_RE[t] * self.variables.volume_DA[t] + self.lambda_IM[t] * self.variables.volume_IM[t]
            #        + self.model_param["hydrogen_price"] * self.variables.hydrogen_output[t]
            #        - self.variables.electrolyzer_cold_start[t]
            #        * self.model_param["cold_start_cost"]
            #        * self.model_param["max_electrolyzer_capacity"]
            for t in np.arange(self.hours)
        )

        if hasattr(self.variables, "abs_alpha_DA"):
            flat_vars = flatten_mvar(self.variables.abs_alpha_DA)
            self.variables.reg_term = self.regulator_value * gp.quicksum(flat_vars)
        elif hasattr(self.variables, "abs_regression_coefficients_DA"):
            flat_vars = flatten_mvar(self.variables.abs_regression_coefficients_DA)
            self.variables.reg_term = self.regulator_value * gp.quicksum(flat_vars)
        else:
            print("No regularization term found")
            self.variables.reg_term = 0

        self.model.setObjective(base_revenue - self.variables.reg_term, GRB.MAXIMIZE)

    def build_model(self):
        """Build and initialize the optimization model."""
        self._initialize_variables()
        self._set_constraints()
        self._set_objective()

    def run_model(self):
        """Solve the optimization model and store results."""
        # self.model.setParam("Crossover", 0)

        # self.model.setParam("BarConvTol", 1e-5)  # Default is 1e-8; higher means faster, less accurate
        # self.model.setParam("Crossover", 0)  # 0 = disable crossover
        # self.model.setParam("CrossoverBasis", 1)
        # self.model.setParam("BarHomogeneous", 1)

        self.build_model()

        self.model.write("models/only_DA_non_linear_model.lp")

        self.model.optimize()

        # load_dotenv()

        # project = os.getenv("WANDB_PROJECT")
        # entity = os.getenv("WANDB_ENTITY")

        # wandb.init(project=project, entity=entity, config=self.model_param, reinit=True)

        if self.model.status == GRB.Status.OPTIMAL:
            # save model as lp file
            print("Model is optimal")

            self.data["volume_DA"] = self.variables.volume_DA.X
            self.data["curtailment"] = self.variables.curtailment.X
            self.data["volume_IM"] = self.variables.volume_IM.X
            #    self.data["electrolyzer_on"] = self.variables.electrolyzer_on.X
            #    self.data["electrolyzer_off"] = self.variables.electrolyzer_off.X
            #    self.data["electrolyzer_standby"] = self.variables.electrolyzer_standby.X
            #    self.data["electrolyzer_cold_start"] = self.variables.electrolyzer_cold_start.X
            #    self.data["consumption_electrolyzer"] = self.variables.consumption_electrolyzer.X
            #    self.data["intended_consumed_electricity"] = self.variables.intended_consumed_electricity.X
            #    self.data["hydrogen_output"] = self.variables.hydrogen_output.X

            self.data["DA_revenue"] = self.lambda_DA_RE * self.variables.volume_DA.X
            self.data["IM_revenue"] = self.lambda_IM * self.variables.volume_IM.X
            #    self.data["hydrogen_revenue"] = self.model_param["hydrogen_price"] * self.variables.hydrogen_output.X
            self.data["revenue"] = self.data["DA_revenue"] + self.data["IM_revenue"]  # + self.data["hydrogen_revenue"]
            self.data["lambda_IM"] = self.lambda_IM

            return_dict = {
                "objective_value": self.model.ObjVal,
                "formulated_model": self.model,
                "price_quantile": self.price_quantile,
            }

            try:
                self.data["absolute_IM_volume"] = self.variables.abs_IM_volume.X
            except AttributeError:
                pass

            try:
                return_dict["regression_coefficients_DA"] = self.variables.regression_coefficients_DA.X
                print("regression_coefficients_DA: ", self.variables.regression_coefficients_DA.X)
            except AttributeError:
                pass

            try:
                return_dict["regression_coefficients_H"] = self.variables.regression_coefficients_H.X
            except AttributeError:
                pass

            try:
                return_dict["x_train"] = self.x_train
                return_dict["features"] = self.features
            except AttributeError:
                pass

            try:
                return_dict["alphas_DA"] = self.variables.alphas_DA.X
                # return_dict["alphas_H"] = self.variables.alphas_H.X
                return_dict["beta_DA"] = self.variables.beta_DA.X
                # return_dict["beta_H"] = self.variables.beta_H.X
            except AttributeError:
                pass

            return_dict["data"] = self.data

            return return_dict

        elif self.model.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible")

            return None

        else:
            print("Status of the model: ", self.model.status)

            #            # make user interaction request in terminal [y/n]
            #            print("Do you want to run a relaxation? [y/n]")
            #            self.model.computeIIS()
            #            self.model.write("models/model_infeasible.ilp")
            #            relaxation = input()
            #            if relaxation == "y":
            #                orignumvars = self.model.NumVars
            #                self.model.feasRelaxS(0, False, True, False)
            #
            #                print("Relaxation complete")
            #                self.model.optimize()
            #                slacks = self.model.getVars()[orignumvars:]
            #                for sv in slacks:
            #                    if sv.X > 1e-9:
            #                        print("%s = %g" % (sv.VarName, sv.X))
            #
            #            else:
            #                print("Model not solved")

            return None


# --------------------------------------------------------------------


class CVaRModel(HindsightModel):
    """Child class that extends the base model with additional variables and constraints."""

    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(timelimit=timelimit, data=data, configfile_name=configfile_name)
        self.CVaR_value = self.model_param["CVaR_array"][CVaR_position]

    def _initialize_variables(self):
        super()._initialize_variables()

        if self.model_param["CVaR_type"] == "volume":
            self.variables.abs_IM_volume = self.model.addMVar(
                shape=self.hours,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name="abs_IM_volume",
            )
            self.variables.pos_IM = self.model.addMVar(
                shape=self.hours,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name="pos_IM",
            )

            self.variables.neg_IM = self.model.addMVar(
                shape=self.hours,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name="neg_IM",
            )

        elif self.model_param["CVaR_type"] == "value":
            self.variables.imbalance_value = self.model.addMVar(
                shape=self.hours,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                name="imbalance_value",
            )

        elif self.model_param["CVaR_type"] == "deviation":
            self.variables.deviation = self.model.addMVar(
                shape=self.hours,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                name="deviation",
            )
            self.variables.pos_deviation = self.model.addMVar(
                shape=self.hours,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name="pos_deviation",
            )
            self.variables.neg_deviation = self.model.addMVar(
                shape=self.hours,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name="neg_deviation",
            )
            self.variables.abs_deviation = self.model.addMVar(
                shape=self.hours,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name="abs_deviation",
            )

        elif self.model_param["CVaR_type"] == "deviation_value":
            self.variables.deviation_value = self.model.addMVar(
                shape=self.hours,
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                name="deviation_value",
            )
            self.variables.pos_deviation_value = self.model.addMVar(
                shape=self.hours,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name="pos_deviation_value",
            )
            self.variables.neg_deviation_value = self.model.addMVar(
                shape=self.hours,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name="neg_deviation_value",
            )

        elif self.model_param["CVaR_type"] == "none":
            print("CVaR is not used")
        else:
            raise ValueError("CVaR_type must be either 'volume', 'value', or 'deviation'")

        if self.model_param["CVaR_type"] in ["volume", "value", "deviation", "deviation_value"]:
            self.variables.VaR = self.model.addMVar(
                shape=1,
                lb=0,
                vtype=GRB.CONTINUOUS,
                name="VaR",
            )
            self.variables.CVaR = self.model.addMVar(shape=1, lb=0, vtype=GRB.CONTINUOUS, name="CVaR")
            self.variables.CVar_aux = self.model.addMVar(shape=self.hours, lb=0, vtype=GRB.CONTINUOUS, name="CVar_aux")

    def _set_constraints(self):
        """Modify constraints (extend or override base constraints)."""
        super()._set_constraints()  # Call base constraints
        if self.model_param["CVaR_type"] == "volume":
            self.constraints.abs_IM_volume = self.model.addConstr(
                self.variables.abs_IM_volume == self.variables.pos_IM + self.variables.neg_IM, name="abs_IM_volume"
            )
            self.constraints.IM_volume = self.model.addConstr(
                self.variables.volume_IM == self.variables.pos_IM - self.variables.neg_IM, name="IM_volume"
            )

            self.constraints.CVar_aux = self.model.addConstr(
                self.variables.CVar_aux >= self.variables.abs_IM_volume - self.variables.VaR, name="CVar_aux"
            )

        elif self.model_param["CVaR_type"] == "value":
            self.constraints.imbalance_value = self.model.addConstr(
                self.variables.imbalance_value == (self.lambda_DA_RE - self.lambda_IM) * self.variables.volume_IM,
            )
            self.constraints.CVar_aux = self.model.addConstr(
                self.variables.CVar_aux >= self.variables.imbalance_value - self.variables.VaR, name="CVar_aux"
            )
        elif self.model_param["CVaR_type"] == "deviation":
            self.constraints.deviation_calc = self.model.addConstr(
                self.variables.deviation == (self.variables.volume_DA - self.energy_FC), name="deviation_calc"
            )

            self.constraints.abs_deviation = self.model.addConstr(
                self.variables.abs_deviation == self.variables.pos_deviation + self.variables.neg_deviation,
                name="abs_deviation",
            )
            self.constraints.deviation = self.model.addConstr(
                self.variables.deviation == self.variables.pos_deviation - self.variables.neg_deviation,
                name="deviation",
            )

            self.constraints.CVar_aux = self.model.addConstr(
                self.variables.CVar_aux >= self.variables.abs_deviation - self.variables.VaR, name="CVar_aux"
            )

        elif self.model_param["CVaR_type"] == "deviation_value":
            self.constraints.deviation_value = self.model.addConstr(
                self.variables.deviation_value
                == (self.lambda_DA_RE - self.lambda_IM) * (self.energy_FC - self.variables.volume_DA),
                name="deviation_value",
            )
            self.constraints.CVar_aux = self.model.addConstr(
                self.variables.CVar_aux >= self.variables.deviation_value - self.variables.VaR, name="CVar_aux"
            )

        elif self.model_param["CVaR_type"] == "none":
            print("CVaR is not used")

        else:
            raise ValueError("CVaR_type must be either 'volume', 'value', or 'deviation'")

        if self.model_param["CVaR_type"] in ["volume", "value", "deviation"]:
            self.constraints.CVaR = self.model.addConstr(
                self.variables.CVaR <= self.CVaR_value, name="CVaR_upper_limit"
            )

            self.constraints.CVaR = self.model.addConstr(
                self.variables.CVaR
                == self.variables.VaR
                + (1 / (self.hours * (1 - self.model_param["alpha"])))
                * gp.quicksum(self.variables.CVar_aux[t] for t in np.arange(self.hours)),
                name="CVaR",
            )


# --------------------------------------------------------------------


class hourly_BaseModel(CVaRModel):
    """Child class that extends the base model with additional variables and constraints."""

    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(timelimit=timelimit, data=data, CVaR_position=CVaR_position, configfile_name=configfile_name)
        self.regulator_value = self.model_param["beta_array"][hyperparameter_position]

    def _initialize_variables(self):
        super()._initialize_variables()
        self.variables.regression_coefficients_DA = self.model.addMVar(
            shape=(24, self.n_features, self.n_price_domains),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="regression_coefficients_DA",
        )

        #        self.variables.regression_coefficients_H = self.model.addMVar(
        #            shape=(24, self.n_features, self.n_price_domains),
        #            lb=-GRB.INFINITY,
        #            ub=GRB.INFINITY,
        #            vtype=GRB.CONTINUOUS,
        #            name="regression_coefficients_H",
        #        )

        self.variables.abs_regression_coefficients_DA = self.model.addMVar(
            shape=(24, self.n_features, self.n_price_domains),
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="abs_regression_coefficients_DA",
        )

    #        self.variables.abs_regression_coefficients_H = self.model.addMVar(
    #            shape=(24, self.n_features, self.n_price_domains),
    #            lb=0,
    #            ub=GRB.INFINITY,
    #            vtype=GRB.CONTINUOUS,
    #            name="abs_regression_coefficients_H",
    #        )

    def _set_constraints(self):
        """Modify constraints (extend or override base constraints)."""
        super()._set_constraints()  # Call base constraints
        for t in range(self.hours):
            h = t % 24
            p_domain = 0
            for i in range(self.n_price_domains):
                if self.lambda_DA_RE[t] >= self.price_quantile[i] and self.lambda_DA_RE[t] < self.price_quantile[i + 1]:
                    p_domain = i
            self.constraints.pred = self.model.addConstr(
                self.variables.volume_DA[t]
                == self.variables.regression_coefficients_DA[h, :, p_domain].reshape(1, self.n_features)
                @ self.x_train.iloc[t].to_numpy().reshape(self.n_features, 1),
                name="linear_policy",
            )

        self.constraints.increasing = self.model.addConstr(
            self.variables.regression_coefficients_DA[:, 1, :] >= 0, name="increasing_policy"
        )

        # Initialize an empty list to hold all the constraints
        self.constraints.increasing_domain = []

        for day in range(int(self.days)):
            for i in range(self.n_price_domains - 1):
                for h in range(24):
                    manipulated_features = self.x_train.iloc[day * 24 + h].copy()
                    manipulated_features["lambda_DA_RE"] = self.price_quantile[i + 1]
                    manipulated_features_np = manipulated_features.to_numpy().reshape(self.n_features, 1)

                    lhs = (
                        self.variables.regression_coefficients_DA[h, :, i].reshape(1, self.n_features)
                        @ manipulated_features_np
                    )
                    rhs = (
                        self.variables.regression_coefficients_DA[h, :, i + 1].reshape(1, self.n_features)
                        @ manipulated_features_np
                    )

                    constr = self.model.addConstr(lhs <= rhs, name=f"increasing_domain_day_{day}_domain_{i}_hour_{h}")
                    self.constraints.increasing_domain.append(constr)

        #        for t in range(self.hours):
        #            h = t % 24
        #            p_domain = 0
        #            for i in range(self.n_price_domains):
        #                if self.lambda_DA_RE[t] >= self.price_quantile[i] and self.lambda_DA_RE[t] < self.price_quantile[i + 1]:
        #                    p_domain = i
        #            self.constraints.pred = self.model.addConstr(
        #                self.variables.intended_consumed_electricity[t]
        #                == self.variables.regression_coefficients_H[h, :, p_domain].reshape(1, self.n_features)
        #                @ self.x_train.iloc[t].to_numpy().reshape(self.n_features, 1),
        #                name="linear_policy",
        #            )

        self.model.addConstr(self.variables.abs_regression_coefficients_DA >= self.variables.regression_coefficients_DA)
        self.model.addConstr(
            self.variables.abs_regression_coefficients_DA >= -self.variables.regression_coefficients_DA
        )


#        self.model.addConstr(self.variables.abs_regression_coefficients_H >= self.variables.regression_coefficients_H)
#        self.model.addConstr(self.variables.abs_regression_coefficients_H >= -self.variables.regression_coefficients_H)


class hourly_linear_PolicyModel(hourly_BaseModel):
    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(
            timelimit=timelimit,
            data=data,
            hyperparameter_position=hyperparameter_position,
            CVaR_position=CVaR_position,
            configfile_name=configfile_name,
        )
        self.features = self.model_param["linear_features"]
        self.x_train = self.data[self.features]
        self.n_features = len(self.features)
        print("Linear model is selected")


class hourly_non_linear_PolicyModel(hourly_BaseModel):
    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(
            timelimit=timelimit,
            data=data,
            hyperparameter_position=hyperparameter_position,
            CVaR_position=CVaR_position,
            configfile_name=configfile_name,
        )
        self.features = self.model_param["non_linear_features"]
        self.x_train = self.data[self.features]
        self.n_features = len(self.features)
        print("Non-linear model is selected")


class aggregated_Base_PolicyModel(CVaRModel):
    """Child class that extends the base model with additional variables and constraints."""

    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(timelimit=timelimit, data=data, CVaR_position=CVaR_position, configfile_name=configfile_name)
        self.regulator_value = self.model_param["beta_array"][hyperparameter_position]

    def _initialize_variables(self):
        super()._initialize_variables()

        self.variables.regression_coefficients_DA = self.model.addMVar(
            shape=(self.n_features, self.n_price_domains),
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="regression_coefficients_DA",
        )

        #        self.variables.regression_coefficients_H = self.model.addMVar(
        #            shape=(self.n_features, self.n_price_domains),
        #            lb=-GRB.INFINITY,
        #            ub=GRB.INFINITY,
        #            vtype=GRB.CONTINUOUS,
        #            name="regression_coefficients_H",
        #        )

        self.variables.abs_regression_coefficients_DA = self.model.addMVar(
            shape=(24, self.n_features, self.n_price_domains),
            lb=0,
            ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="abs_regression_coefficients_DA",
        )

    #        self.variables.abs_regression_coefficients_H = self.model.addMVar(
    #            shape=(24, self.n_features, self.n_price_domains),
    #            lb=0,
    #            ub=GRB.INFINITY,
    #            vtype=GRB.CONTINUOUS,
    #            name="abs_regression_coefficients_H",
    #        )

    def _set_constraints(self):
        super()._set_constraints()

        for t in range(self.n_samples):
            p_domain = 0
            for i in range(self.n_price_domains):
                if self.lambda_DA_RE[t] >= self.price_quantile[i] and self.lambda_DA_RE[t] < self.price_quantile[i + 1]:
                    p_domain = i

            x_t = self.x_train.iloc[t].to_numpy().reshape(self.n_features, 1)

            # Prediction constraints for volume_DA
            self.model.addConstr(
                self.variables.volume_DA[t]
                == self.variables.regression_coefficients_DA[:, p_domain].reshape(1, self.n_features) @ x_t,
                name=f"linear_policy_DA_{t}",
            )

            # Prediction constraints for intended electricity
        #            self.model.addConstr(
        #                self.variables.intended_consumed_electricity[t]
        #                == self.variables.regression_coefficients_H[:, p_domain].reshape(1, self.n_features) @ x_t,
        #                name=f"linear_policy_H_{t}",
        #            )

        # Optional: Monotonicity of response w.r.t. price (assumes price is 2nd feature = index 1)
        self.model.addConstr(
            self.variables.regression_coefficients_DA[1, :] >= 0,
            name="increasing_policy",
        )

        # Enforce monotonicity between price domains for each t
        self.constraints.increasing_domain = []
        for t in range(self.n_samples):
            for i in range(self.n_price_domains - 1):
                manipulated_features = self.x_train.iloc[t].copy()
                manipulated_features["lambda_DA_RE"] = self.price_quantile[i + 1]
                x_manip = manipulated_features.to_numpy().reshape(self.n_features, 1)

                lhs = self.variables.regression_coefficients_DA[:, i].reshape(1, self.n_features) @ x_manip
                rhs = self.variables.regression_coefficients_DA[:, i + 1].reshape(1, self.n_features) @ x_manip

                constr = self.model.addConstr(lhs <= rhs, name=f"increasing_domain_t{t}_i{i}")
                self.constraints.increasing_domain.append(constr)

        self.model.addConstr(self.variables.abs_regression_coefficients_DA >= self.variables.regression_coefficients_DA)
        self.model.addConstr(
            self.variables.abs_regression_coefficients_DA >= -self.variables.regression_coefficients_DA
        )


#        self.model.addConstr(self.variables.abs_regression_coefficients_H >= self.variables.regression_coefficients_H)
#        self.model.addConstr(self.variables.abs_regression_coefficients_H >= -self.variables.regression_coefficients_H)


class linear_PolicyModel(aggregated_Base_PolicyModel):
    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(
            timelimit=timelimit,
            data=data,
            hyperparameter_position=hyperparameter_position,
            CVaR_position=CVaR_position,
            configfile_name=configfile_name,
        )
        self.features = self.model_param["linear_features"]
        self.x_train = self.data[self.features]
        self.n_features = len(self.features)
        self.n_samples = self.x_train.shape[0]
        print("Linear model (non-hourly) is selected")


class non_linear_PolicyModel(aggregated_Base_PolicyModel):
    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(
            timelimit=timelimit,
            data=data,
            hyperparameter_position=hyperparameter_position,
            CVaR_position=CVaR_position,
            configfile_name=configfile_name,
        )
        self.features = self.model_param["non_linear_features"]
        print("Features used:", self.features)
        self.x_train = self.data[self.features]
        self.n_features = len(self.features)
        self.n_samples = self.x_train.shape[0]

        print("Non-hourly non-linear model selected")


# --------------------------------------------------------------------
class KernelBaseClass(CVaRModel):
    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(
            timelimit=timelimit,
            data=data,
            hyperparameter_position=hyperparameter_position,
            CVaR_position=CVaR_position,
            configfile_name=configfile_name,
        )
        self.features = self.model_param["kernel_features"]
        self.x_train = self.data[self.features]

        self.realized_prices = self.x_train["lambda_DA_RE"].to_numpy()
        self.x_train = self.x_train.drop(columns=["lambda_DA_RE"])
        correlation_matrix = self.x_train.corr().abs()
        np.fill_diagonal(correlation_matrix.values, 0)
        print(correlation_matrix.max().sort_values(ascending=False))

        cond_number = np.linalg.cond(self.x_train.to_numpy())
        print("Condition number of feature matrix:", cond_number)

        self.n_samples = self.x_train.shape[0]

    def _initialize_variables(self):
        super()._initialize_variables()

        self.variables.alphas_DA = self.model.addMVar(
            shape=self.n_samples, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alphas_DA"
        )
        #        self.variables.alphas_H = self.model.addMVar(
        #            shape=self.n_samples, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alphas_H"
        #        )
        self.variables.beta_DA = self.model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta_DA"
        )
        #        self.variables.beta_H = self.model.addVar(
        #            lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta_H"
        #        )

        self.variables.abs_alpha_DA = self.model.addMVar(
            shape=self.n_samples,
            lb=0,
            vtype=GRB.CONTINUOUS,
            name="abs_alpha_DA",
        )

    #        self.variables.abs_alpha_H = self.model.addMVar(
    #            shape=self.n_samples,
    #            lb=0,
    #            vtype=GRB.CONTINUOUS,
    #            name="abs_alpha_H",
    #        )

    def get_kernel_function(self):
        """Override this in subclasses to change kernel behavior."""
        raise NotImplementedError("Subclasses must implement compute_kernel_matrix()")

    def _set_constraints(self):
        super()._set_constraints()
        K = self.get_kernel_function()
        K /= np.max(np.abs(K))
        print("Kernel matrix :", K)
        alphas_DA = self.variables.alphas_DA.tolist()
        #        alphas_H = self.variables.alphas_H.tolist()

        self.constraints.da_kernel_pred = self.model.addConstrs(
            (
                self.variables.volume_DA[t]
                == self.variables.beta_DA * self.realized_prices[t] + gp.LinExpr(K[t, :].tolist(), alphas_DA)
                for t in range(self.n_samples)
            ),
            name="kernel_policy_DA",
        )

        #        self.constraints.h_kernel_pred = self.model.addConstrs(
        #            (
        #                self.variables.intended_consumed_electricity[t]
        #                == self.variables.beta_H * self.realized_prices[t] + gp.LinExpr(K[t, :].tolist(), alphas_H)
        #                for t in range(self.n_samples)
        #            ),
        #            name="kernel_policy_H",
        #        )

        self.model.addConstr(self.variables.abs_alpha_DA >= self.variables.alphas_DA)
        self.model.addConstr(self.variables.abs_alpha_DA >= -self.variables.alphas_DA)


#        self.model.addConstr(self.variables.abs_alpha_H >= self.variables.alphas_H)
#        self.model.addConstr(self.variables.abs_alpha_H >= -self.variables.alphas_H)


class RBF_Kernel(KernelBaseClass):
    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(
            timelimit=timelimit,
            data=data,
            hyperparameter_position=hyperparameter_position,
            CVaR_position=CVaR_position,
            configfile_name=configfile_name,
        )
        self.gamma = self.model_param["kernel_gamma"]
        self.regulator_value = self.model_param["rbf_kernel_alpha_array"][hyperparameter_position]

        print("RBF-Kernel-based policy model selected")

    def get_kernel_function(self):
        return compute_kernel_matrix(
            data=self.x_train,
            kernel_function=rbf_kernel_function,
            gamma=self.gamma,
        )


class Polynominal_Kernel(KernelBaseClass):
    def __init__(
        self, timelimit=100, data=None, hyperparameter_position=None, CVaR_position=None, configfile_name=None
    ):
        super().__init__(
            timelimit=timelimit,
            data=data,
            hyperparameter_position=hyperparameter_position,
            CVaR_position=CVaR_position,
            configfile_name=configfile_name,
        )
        self.degree = self.model_param["kernel_degree"]
        self.regulator_value = self.model_param["poly_kernel_alpha_array"][hyperparameter_position]

        print("Polynomial-Kernel-based policy model selected")

    def get_kernel_function(self):
        return compute_kernel_matrix(
            data=self.x_train,
            kernel_function=regularized_poly_kernel_function,
            degree=self.degree,
        )


if __name__ == "__main__":
    train_set, val_set, test_set = data_loader("data/processed/data_small.csv")
    configfile_name = "kernel"
    # print(train_set.groupby("date").count()[train_set.groupby("date").count()["datetime"] != 24])

    model = HindsightModel(timelimit=400, data=train_set, configfile_name=configfile_name)
