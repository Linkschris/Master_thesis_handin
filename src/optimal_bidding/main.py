from data import get_rolling_windows
from sklearn.preprocessing import MinMaxScaler
from saving import save_results
from analyse_results import analyse_test_results, analyse_train_results, analyse_val_results
from test import test_model
from train import train_model
from model import (
    hourly_non_linear_PolicyModel,
    non_linear_PolicyModel,
    RBF_Kernel,
    Polynominal_Kernel,
    linear_PolicyModel,
    hourly_linear_PolicyModel,
    CVaRModel,
    HindsightModel,
)
import pandas as pd
from omegaconf import OmegaConf

OPTIMIZATION_MODELS = [HindsightModel]
POLICY_MODELS = [non_linear_PolicyModel, linear_PolicyModel]
HOURLY_POLICY_MODELS = [hourly_linear_PolicyModel, hourly_non_linear_PolicyModel]
KERNEL_MODELS = [RBF_Kernel, Polynominal_Kernel]
MODEL_LIST = OPTIMIZATION_MODELS + HOURLY_POLICY_MODELS + POLICY_MODELS + KERNEL_MODELS


if __name__ == "__main__":
    config_file_name = "low_CVaR_no_reg"
    model_list = POLICY_MODELS + HOURLY_POLICY_MODELS + [RBF_Kernel]  # MODEL_LIST
    data_set_size = "small"  # "full" or "small"

    if data_set_size == "full":
        data_set = pd.read_csv("data/processed/data.csv", index_col=0, header=0)
    else:
        # Load small test set
        data_set = pd.read_csv("data/processed/data_small.csv", index_col=0, header=0)

    model_parameters = OmegaConf.load(f"configs/{config_file_name}.yaml")

    train_size = model_parameters["train_length"]
    val_size = model_parameters["val_length"]
    test_size = model_parameters["test_length"]
    windows = get_rolling_windows(data_set, train_size=train_size, val_size=val_size, test_size=test_size)
    print(windows)

    for model_class in model_list:
        train_result_df = pd.DataFrame()
        test_result_df = pd.DataFrame()

        validation_results = []  # List of dicts, one entry per combination per window

        for i, (train_start, train_end, val_start, val_end, test_start, test_end) in enumerate(windows[13:]):
            print(f"Rolling window {i + 1}/{len(windows)}")
            train_set = data_set[train_start:train_end].reset_index()
            val_set = data_set[val_start:val_end].reset_index()
            final_train_set = data_set[train_start:val_end].reset_index()
            test_set = data_set[test_start:test_end].reset_index()

            print(
                f"Train set shape: {train_set.shape}"
                f", Validation set shape: {val_set.shape}, "
                f"Final train set shape: {final_train_set.shape}, "
                f"Test set shape: {test_set.shape}"
            )

            exclude_cols = {
                "datetime",
                "date",
                "DK1_Imbalance_Price_(EUR)",
                "DK2_Imbalance_Price_(EUR)",
                "energy_FC",
                "energy_RE",
            }
            # Anything related to 'lambda' or 'price' should be excluded
            exclude_keywords = ["lambda", "price"]

            # Final feature columns = all except exclusions
            feature_cols = [
                col
                for col in final_train_set.columns
                if col not in exclude_cols and not any(keyword in col.lower() for keyword in exclude_keywords)
            ]

            # Fit scaler on final_train_set
            scaler = MinMaxScaler()
            scaler.fit(final_train_set[feature_cols])

            scaled_sets = []

            for df in [train_set, val_set, final_train_set, test_set]:
                scaled_values = scaler.transform(df[feature_cols])
                scaled_df = pd.DataFrame(scaled_values, columns=feature_cols)

                retained_cols = [
                    col
                    for col in df.columns
                    if col in exclude_cols or any(keyword in col.lower() for keyword in exclude_keywords)
                ]
                retained_df = df[retained_cols].reset_index(drop=True)

                new_df = pd.concat([retained_df, scaled_df], axis=1)
                scaled_sets.append(new_df)

            # Reassign to actual variables
            train_set, val_set, final_train_set, test_set = scaled_sets

            # select hyperparameter_array
            if model_class == HindsightModel:
                hyperparameter_array = None
            elif model_class == CVaRModel:
                hyperparameter_array = [0]
            elif model_class in POLICY_MODELS or model_class in HOURLY_POLICY_MODELS:
                hyperparameter_array = model_parameters["beta_array"]
            elif model_class == RBF_Kernel:
                hyperparameter_array = model_parameters["rbf_kernel_alpha_array"]
            elif model_class == Polynominal_Kernel:
                hyperparameter_array = model_parameters["poly_kernel_alpha_array"]
            else:
                raise ValueError(f"Model {model_class} not recognized")

            if hyperparameter_array is not None:
                num_cvar = len(model_parameters["CVaR_array"])
                columns = list(range(num_cvar))  # Or use actual values as names
                objective_value_df = pd.DataFrame(columns=columns)

                for hyperparameter_position in range(len(hyperparameter_array)):
                    for CVaR_position in range(len(model_parameters["CVaR_array"])):
                        objective_value = 0
                        print(
                            f"Training {model_class} with hyperparameter: {hyperparameter_array[hyperparameter_position]} and CVaR: {model_parameters['CVaR_array'][CVaR_position]}"
                        )

                        # Train the model
                        train_result_dict = train_model(
                            model_class,
                            train_set,
                            config_file_name,
                            hyperparameter_position=hyperparameter_position,
                            CVaR_position=CVaR_position,
                            saving=False,
                        )
                        extra_args = {}
                        if model_class == CVaRModel:
                            extra_args = {"CVaR_position": CVaR_position}

                        val_result_dict = test_model(
                            model_class, val_set, config_file_name, train_result_dict, saving=False, **extra_args
                        )

                        # Extract test objective value
                        objective_value = analyse_val_results(val_result_dict["data"], config_file_name)

                        # Save it
                        validation_results.append(
                            {
                                "window": i,
                                "hyperparameter": hyperparameter_position,
                                "CVaR": CVaR_position,
                                "objective_value": objective_value,
                            }
                        )

                # Select only dictonaries from the list, if window is the current window
                val_values_list = [d for d in validation_results if d["window"] == i]

                if val_values_list:
                    best_combination = max(val_values_list, key=lambda x: x["objective_value"])
                    best_hyperparameter_position = best_combination["hyperparameter"]
                    best_CVaR_position = best_combination["CVaR"]
                else:
                    best_combination = None
                    best_hyperparameter_position = None
                    best_CVaR_position = None

                # Train the model with the best hyperparameter
                train_result_dict = train_model(
                    model_class,
                    final_train_set,
                    config_file_name,
                    hyperparameter_position=best_hyperparameter_position,
                    CVaR_position=best_CVaR_position,
                    saving=False,
                )
                extra_args = {}
                if model_class == CVaRModel:
                    extra_args = {"CVaR_position": CVaR_position}

                test_result_dict = test_model(
                    model_class, test_set, config_file_name, train_result_dict, saving=False, **extra_args
                )
            else:
                train_result_dict = train_model(model_class, final_train_set, config_file_name, saving=False)
                test_result_dict = test_model(model_class, test_set, config_file_name, train_result_dict, saving=False)

            # Append results to DataFrames
            if train_result_dict is not None:
                train_result_df = pd.concat([train_result_df, train_result_dict["data"].iloc[:24]], ignore_index=True)
            else:
                print(f"Warning: Skipping result due to optimization failure at CVaR position {CVaR_position}")

            test_result_df = pd.concat([test_result_df, test_result_dict["data"]], ignore_index=True)

        objective_value_train = analyse_train_results(train_result_df, config_file_name)
        objective_value = analyse_test_results(test_result_df, config_file_name)

        save_results(
            {
                "data": train_result_df,
                "objective_value": objective_value_train,
                "validation_results": validation_results,
            },
            model_class.__name__,
            config_file_name,
            dataset_size=data_set_size,
            mode="train",
        )
        save_results(
            {
                "data": test_result_df,
                "objective_value": objective_value,  # , "hydrogen_penalty": hydrogen_penalty
            },
            model_class.__name__,
            config_file_name,
            dataset_size=data_set_size,
            mode="test",
        )
