# test_framework/model_tester.py
import pandas as pd
import numpy as np
from train import train_model

from saving import save_results
from data import data_loader, rbf_kernel_function, poly_kernel_function
from model import CVaRModel, linear_PolicyModel
from model import HindsightModel
from model import (
    non_linear_PolicyModel,
    RBF_Kernel,
    hourly_linear_PolicyModel,
    hourly_non_linear_PolicyModel,
    Polynominal_Kernel,
)
from omegaconf import OmegaConf

# Explicitly group model classes by behavior
OPTIMIZATION_MODELS = [HindsightModel, CVaRModel]
POLICY_MODELS = [linear_PolicyModel, non_linear_PolicyModel]
HOURLY_POLICY_MODELS = [hourly_linear_PolicyModel, hourly_non_linear_PolicyModel]
KERNEL_MODELS = [RBF_Kernel, Polynominal_Kernel]


class BaseModelTester:
    def __init__(self, model_class, config, test_set, config_file_name, train_result_dict, dataset_size="full"):
        self.model_class = model_class
        self.train_result_dict = train_result_dict
        self.model_name = model_class.__name__
        self.config = config
        self.test_set = test_set.copy()
        self.config_file_name = config_file_name
        self.result_df = None
        self.dataset_size = dataset_size
        self.objective_value = None
        self.lambda_DA_RE = self.test_set["lambda_DA_RE"].copy()

    def run_model(self):
        """Override in subclasses"""
        raise NotImplementedError

    def compute_postprocessing(self):
        df = self.result_df
        if df is None:
            pass
        else:
            df["imbalance"] = df["energy_RE"] - df["bids_elec"]  # - df["consumption_electrolyser"]
            df["DA_revenue"] = df["lambda_DA_RE"] * df["bids_elec"]
            #        df["H_revenue"] = self.config["hydrogen_price"] * df["hydrogen_output"]
            df["IM_revenue"] = df["lambda_IM"] * df["imbalance"]
            df["total_revenue"] = df["DA_revenue"] + df["IM_revenue"]  # + df["H_revenue"]
            self.result_df = df

    def save_test_results(self):
        save_results(
            return_dict={"data": self.result_df, "objective_value": self.objective_value},
            model_name=self.model_name,
            config_file_name=self.config_file_name,
            dataset_size=self.dataset_size,
            mode="test",
        )


class OptimizationModelTester(BaseModelTester):
    def __init__(self, model_class, config, test_set, config_file_name, train_result_dict, **kwargs):
        super().__init__(model_class, config, test_set, config_file_name, train_result_dict)
        self.optional_args = kwargs  # Store for use in run_model if needed

    def run_model(self):
        return_dict = train_model(
            self.model_class,
            self.test_set,
            self.config_file_name,
            saving=False,
            **self.optional_args,  # Inject optional params (like CVaR_position)
        )
        self.result_df = pd.DataFrame(
            {
                "datetime": self.test_set["datetime"],
                "bids_elec": return_dict["data"]["volume_DA"],
                #                "consumption_electrolyser": return_dict["data"]["consumption_electrolyzer"],
                "curtailment": return_dict["data"]["curtailment"],
                #                "hydrogen_output": return_dict["data"]["hydrogen_output"],
                "lambda_DA_RE": self.test_set["lambda_DA_RE"],
                "energy_FC": self.test_set["energy_FC"] * self.config["max_wind_capacity"],
                "energy_RE": self.test_set["energy_RE"] * self.config["max_wind_capacity"],
                "lambda_IM": self.test_set["lambda_IM"],
            }
        )


class hourly_PolicyModelTester(BaseModelTester):
    def run_model(self):
        # model_name = self.model_name
        x_train = self.train_result_dict["x_train"]

        features = x_train.columns.tolist()
        regression_coeffs_DA = self.train_result_dict["regression_coefficients_DA"]
        #        regression_coeffs_H = self.train_result_dict["regression_coefficients_H"]
        price_quantile = self.train_result_dict["price_quantile"]
        self.test_set["intercept"] = 1
        x_test = self.test_set[features].copy()
        x_test = x_test[["intercept"] + [f for f in features if f != "intercept"]]
        self.test_set[features] = x_test

        bids_elec = []
        #        consumption_electrolyser = []
        #        hydrogen_output = []

        for t in range(x_test.shape[0]):
            h = t % 24
            lambda_DA_RE = x_test.loc[t, "lambda_DA_RE"]
            for pr in range(len(price_quantile) - 1):
                if price_quantile[pr] < lambda_DA_RE <= price_quantile[pr + 1]:
                    x_row = x_test.loc[t].to_numpy()
                    bids_elec.append(regression_coeffs_DA[h, :, pr] @ x_row)
                    #                    electrolyser_bid = regression_coeffs_H[h, :, pr] @ x_row
                    #                    electrolyser_bid = np.clip(
                    #                        electrolyser_bid,
                    #                        self.config["p_min"] * self.config["max_electrolyzer_capacity"],
                    #                        self.config["max_electrolyzer_capacity"],
                    #                    )
                    #                    consumption_electrolyser.append(electrolyser_bid)
                    break

        #            A = area(self.config)
        #            i = find_i_from_p(np.array([consumption_electrolyser[-1]]), A, self.config["T_op"])
        #            hydrogen_output.append(h_prod(i, self.config["T_op"], self.config["Pr"], A)[0] * self.config["eta_storage"])

        df = pd.DataFrame(
            {
                "datetime": self.test_set["datetime"],
                "bids_elec": np.clip(
                    bids_elec, -self.config["max_electrolyzer_capacity"], self.config["max_wind_capacity"]
                ),
                #                "consumption_electrolyser": np.clip(
                #                    consumption_electrolyser,
                #                    self.config["p_min"] * self.config["max_electrolyzer_capacity"],
                #                    self.config["max_electrolyzer_capacity"],
                #                ),
                #                "hydrogen_output": hydrogen_output,
                "lambda_DA_RE": self.test_set["lambda_DA_RE"],
                "energy_RE": self.test_set["energy_RE"] * self.config["max_wind_capacity"],
                "lambda_IM": self.test_set["lambda_IM"],
            }
        )

        self.result_df = df


class PolicyModelTester(BaseModelTester):
    def run_model(self):
        x_train = self.train_result_dict["x_train"]
        features = x_train.columns.tolist()

        regression_coeffs_DA = self.train_result_dict[
            "regression_coefficients_DA"
        ]  # shape: (n_features, n_price_domains)
        #        regression_coeffs_H = self.train_result_dict[
        #            "regression_coefficients_H"
        #        ]  # shape: (n_features, n_price_domains)
        price_quantile = self.train_result_dict["price_quantile"]

        self.test_set["intercept"] = 1
        x_test = self.test_set[features].copy()
        x_test = x_test[["intercept"] + [f for f in features if f != "intercept"]]
        self.test_set[features] = x_test

        bids_elec = []
        #        consumption_electrolyser = []
        #        hydrogen_output = []

        for t in range(x_test.shape[0]):
            lambda_DA_RE = x_test.loc[t, "lambda_DA_RE"]

            # Determine price domain
            for pr in range(len(price_quantile) - 1):
                if price_quantile[pr] < lambda_DA_RE <= price_quantile[pr + 1]:
                    x_row = x_test.loc[t].to_numpy()
                    bids_elec.append(regression_coeffs_DA[:, pr] @ x_row)
                    #                    electrolyser_bid = regression_coeffs_H[:, pr] @ x_row
                    #                    electrolyser_bid = np.clip(
                    #                        electrolyser_bid,
                    #                        self.config["p_min"] * self.config["max_electrolyzer_capacity"],
                    #                        self.config["max_electrolyzer_capacity"],
                    #                    )
                    #                    consumption_electrolyser.append(electrolyser_bid)
                    break

        #            A = area(self.config)
        #            i = find_i_from_p(np.array([consumption_electrolyser[-1]]), A, self.config["T_op"])
        #            h2_output = h_prod(i, self.config["T_op"], self.config["Pr"], A)[0] * self.config["eta_storage"]
        #            hydrogen_output.append(h2_output)

        df = pd.DataFrame(
            {
                "datetime": self.test_set["datetime"],
                "bids_elec": np.clip(
                    bids_elec, -self.config["max_electrolyzer_capacity"], self.config["max_wind_capacity"]
                ),
                #                "consumption_electrolyser": np.clip(
                #                    consumption_electrolyser,
                #                    self.config["p_min"] * self.config["max_electrolyzer_capacity"],
                #                    self.config["max_electrolyzer_capacity"],
                #                ),
                #                "hydrogen_output": hydrogen_output,
                "lambda_DA_RE": self.test_set["lambda_DA_RE"],
                "energy_RE": self.test_set["energy_RE"] * self.config["max_wind_capacity"],
                "lambda_IM": self.test_set["lambda_IM"],
            }
        )

        self.result_df = df


class KernelModelTester(BaseModelTester):
    def run_model(self):
        try:
            x_train = self.train_result_dict["x_train"]
            x_train_np = x_train.to_numpy()
            features = x_train.columns.tolist()
            x_test = self.test_set[features].copy().to_numpy()

            alphas_DA = self.train_result_dict["alphas_DA"]
            #        alphas_H = self.train_result_dict["alphas_H"]
            beta_DA = self.train_result_dict["beta_DA"]
            #        beta_H = self.train_result_dict["beta_H"]

            # Select kernel function based on model class
            model_name = self.model_name
            if "RBF_Kernel" in model_name:
                kernel_func = rbf_kernel_function
                kernel_params = {"gamma": self.config["kernel_gamma"]}
            elif "Polynominal_Kernel" in model_name:
                kernel_func = poly_kernel_function
                kernel_params = {"degree": self.config["kernel_degree"]}
            else:
                raise ValueError(f"Unknown kernel model: {model_name}")

            bids_elec = []
            #        consumption_electrolyser = []
            #        hydrogen_output = []

            for t in range(x_test.shape[0]):
                k_t = np.array([kernel_func(x_i, x_test[t], **kernel_params) for x_i in x_train_np])

                bid_DA = beta_DA * self.lambda_DA_RE[t] + np.dot(k_t, alphas_DA)
                #            volume_H = beta_H * self.lambda_DA_RE[t] + np.dot(k_t, alphas_H)

                bid_DA = np.clip(bid_DA, -self.config["max_electrolyzer_capacity"], self.config["max_wind_capacity"])
                #            volume_H = np.clip(
                #                volume_H,
                #                self.config["p_min"] * self.config["max_electrolyzer_capacity"],
                #                self.config["max_electrolyzer_capacity"],
                #            )
                #
                #            A = area(self.config)
                #            i = find_i_from_p(np.array([volume_H]), A, self.config["T_op"])
                #            h = h_prod(i, self.config["T_op"], self.config["Pr"], A)[0] * self.config["eta_storage"]

                bids_elec.append(bid_DA)
            #            consumption_electrolyser.append(volume_H)
            #            hydrogen_output.append(h)

            self.result_df = pd.DataFrame(
                {
                    "datetime": self.test_set["datetime"],
                    "bids_elec": np.array(bids_elec).flatten(),
                    #                "consumption_electrolyser": np.array(consumption_electrolyser).flatten(),
                    #                "hydrogen_output": np.array(hydrogen_output).flatten(),
                    "lambda_DA_RE": self.test_set["lambda_DA_RE"],
                    "energy_RE": self.test_set["energy_RE"] * self.config["max_wind_capacity"],
                    "lambda_IM": self.test_set["lambda_IM"],
                }
            )

        except Exception:
            self.result_df = None


def test_model(model_class, test_set, config_file_name, train_result_dict, saving=True, **kwargs):
    config = OmegaConf.load(f"configs/{config_file_name}.yaml")

    print(f"Testing {model_class.__name__}")

    if model_class in POLICY_MODELS:
        tester = PolicyModelTester(model_class, config, test_set, config_file_name, train_result_dict)
    elif model_class in OPTIMIZATION_MODELS:
        if model_class == CVaRModel:
            tester = OptimizationModelTester(
                model_class, config, test_set, config_file_name, train_result_dict, **kwargs
            )
        else:
            tester = OptimizationModelTester(model_class, config, test_set, config_file_name, train_result_dict)

    elif model_class in KERNEL_MODELS:
        tester = KernelModelTester(model_class, config, test_set, config_file_name, train_result_dict)
    elif model_class in HOURLY_POLICY_MODELS:
        tester = hourly_PolicyModelTester(model_class, config, test_set, config_file_name, train_result_dict)
    else:
        raise ValueError(f"Model class {model_class.__name__} not assigned to a model group.")

    tester.run_model()
    tester.compute_postprocessing()
    return_dict = {
        "data": tester.result_df,
    }

    if saving:
        tester.save_test_results()

    return return_dict


if __name__ == "__main__":
    data_size = "full"  # or "full"

    if data_size == "small":
        # Load small test set
        train_set, val_set, test_set = data_loader("data/processed/data_small.csv")
    else:
        # Load full test set
        train_set, val_set, test_set = data_loader("data/processed/data.csv")

    # Define model list
    models_to_test = [non_linear_PolicyModel]

    # Define config
    config_file_name = "base_case"

    for model in models_to_test:
        print(f"Testing {model.__name__}")
        # Run test
        return_dict = test_model(model, test_set, config_file_name, saving=False)
