import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from omegaconf import OmegaConf
from electrolyser_efficiency import area, find_i_from_p, h_prod
from data import smooth_spikes


class expando(object):
    """
    A small class which can have attributes set
    """

    pass


class HydrogenOptimizer:
    def __init__(self, data: pd.DataFrame, model_param: dict, timelimit=60, model_type="minimum"):
        """
        Initialize optimizer.
        """
        self.data = data.copy()
        self.model_param = model_param
        self.timelimit = timelimit
        self.model_type = model_type
        self.hydrogen_price = model_param["hydrogen_price"]

        self.hours = len(self.data)
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.days = self.data["date"].dt.normalize().nunique()

        self.model = gp.Model("Hydrogen_Minimum_Optimization")
        self.model.Params.TimeLimit = self.timelimit

        self.S = np.array([t for t in range(0, self.model_param["N_s"])])

        self.variables = expando()  # Store variables
        self.constraints = expando()  # Store constraints
        self.results = expando()  # Store results
        self.results.objective_value = None

        self._initialize_variables()
        self._set_constraints()
        self._set_objective()

    def _initialize_variables(self):
        if self.model_type == "minimum":
            self.variables.consumption = self.model.addMVar(
                shape=self.hours,
                lb=self.model_param["p_min"] * self.model_param["max_electrolyzer_capacity"],
                ub=self.model_param["max_electrolyzer_capacity"],
                vtype=GRB.CONTINUOUS,
                name="electricity_consumption",
            )

            self.variables.missing_output = self.model.addMVar(
                shape=self.hours,
                lb=0,
                ub=self.model_param["max_electrolyzer_capacity"],
                vtype=GRB.CONTINUOUS,
                name="missing_output",
            )

        elif self.model_type == "revenue":
            self.variables.consumption = self.model.addMVar(
                shape=self.hours,
                lb=0,
                ub=self.model_param["max_electrolyzer_capacity"],
                vtype=GRB.CONTINUOUS,
                name="electricity_consumption",
            )
        else:
            raise ValueError("Invalid model type. Use 'minimum' or 'revenue'.")

        self.variables.hydrogen_output = self.model.addMVar(
            shape=self.hours, lb=0, vtype=GRB.CONTINUOUS, name="hydrogen_output"
        )

    def _set_constraints(self):
        if self.model_type == "minimum":
            # Minimum daily hydrogen production
            dates = self.data["date"].dt.normalize()
            for day in dates.unique():
                idx = dates[dates == day].index
                self.constraints.min_daily_h2 = self.model.addConstr(
                    gp.quicksum(self.variables.hydrogen_output[i] for i in idx)  # +self.variables.missing_output[i]
                    >= self.model_param["min_daily_hydrogen_production"],
                    name=f"min_daily_h2_{day}",
                )

        elif self.model_type == "revenue":
            pass
        else:
            raise ValueError("Invalid model type. Use 'minimum' or 'revenue'.")

        self.constraints.hydrogen_production = self.model.addConstrs(
            (
                (
                    self.variables.hydrogen_output[t]
                    <= (
                        self.variables.consumption[t] * self.model_param["a"][int(s)]
                        + 1 * self.model_param["b"][int(s)]
                    )
                    * self.model_param["eta_storage"]
                )
                for t in range(self.hours)
                for s in self.S
            ),
            name="hydrogen_production",
        )

    def _set_objective(self):
        if self.model_type == "minimum":
            # Set objective to minimize total cost
            prices = self.data["lambda_DA_FC"].values
            self.model.setObjective(
                gp.quicksum(
                    prices[i] * self.variables.consumption[i] for i in range(self.hours)
                ),  # + self.variables.missing_output[i]*1.5*self.hydrogen_price
                GRB.MINIMIZE,
            )
        elif self.model_type == "revenue":
            # Set objective to maximize total revenue
            prices = self.data["lambda_DA_RE"].values

            self.model.setObjective(
                gp.quicksum(
                    self.hydrogen_price * self.variables.hydrogen_output[i] - prices[i] * self.variables.consumption[i]
                    for i in range(self.hours)
                ),
                GRB.MAXIMIZE,
            )
        else:
            raise ValueError("Invalid model type. Use 'minimum' or 'revenue'.")

    def optimize(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            if self.model_type == "minimum":
                self.data["minimum_consumption"] = self.variables.consumption.X
                self.data["minimum_hydrogen_output"] = self.variables.hydrogen_output.X
            elif self.model_type == "revenue":
                self.data["revenue_hydrogen_output"] = self.variables.hydrogen_output.X
                self.data["revenue_consumption"] = self.variables.consumption.X

                print(np.unique(self.data["revenue_consumption"].values))
                wrong_min_value = self.data["revenue_consumption"].values.min()
                print(f"Wrong min value: {wrong_min_value}")
                self.data["revenue_consumption"] = np.where(
                    self.data["revenue_consumption"] == wrong_min_value, 0, self.data["revenue_consumption"]
                )

            if "minimum_consumption" and "revenue_consumption" in self.data.columns:
                self.data["scheduled_consumption"] = self.data["minimum_consumption"] + self.data["revenue_consumption"]
                self.data["scheduled_consumption"] = self.data["scheduled_consumption"].clip(
                    lower=0, upper=self.model_param["max_electrolyzer_capacity"]
                )

                hydrogen_output_list = []
                for t in range(self.hours):
                    AA = area(self.model_param)
                    i = find_i_from_p([self.data["scheduled_consumption"][t]], AA, self.model_param["T_op"])
                    hydrogen_output_list.append(
                        h_prod(i, self.model_param["T_op"], self.model_param["Pr"], AA)[0]
                        * self.model_param["eta_storage"]
                    )
                self.data["scheduled_hydrogen_output"] = hydrogen_output_list
            return self.data
        else:
            print("Optimization did not find optimal solution.")
            return None


if __name__ == "__main__":
    data = pd.read_csv("data/processed/data.csv", index_col=0, header=0)
    data = smooth_spikes(data, column="lambda_DA_FC", window=12, threshold=3.0)
    config_file_name = "base_case"
    model_parameter = OmegaConf.load(f"configs/{config_file_name}.yaml")

    # Example usage
    hydrogen_production_planner = HydrogenOptimizer(data, model_parameter, timelimit=60, model_type="minimum")
    data = hydrogen_production_planner.optimize()
    hydrogen_production_planner = HydrogenOptimizer(data, model_parameter, timelimit=60, model_type="revenue")
    data = hydrogen_production_planner.optimize()
    data.to_csv("data/processed/hydrogen_optimized_data.csv", index=False)
