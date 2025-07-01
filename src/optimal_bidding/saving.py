import os
import numpy as np
import json


def save_results(
    return_dict: dict,
    model_name: str,
    config_file_name: str,
    dataset_size: str,  # "small" or "full"
    mode: str = "train",  # "train", "val", "test"
):
    """
    Save the output of a model run (train/test/val) in a structured folder.
    """
    assert mode in ["train", "test", "val"], "Mode must be one of: train, test, val"
    assert dataset_size in ["small", "full"], "Dataset size must be 'small' or 'full'"

    save_path = f"results/{mode}/{model_name}/{dataset_size}_data/{config_file_name}"
    os.makedirs(save_path, exist_ok=True)

    # Save model LP file if available
    if "formulated_model" in return_dict:
        return_dict["formulated_model"].write(f"{save_path}/model.lp")

    # Save objective value
    with open(f"{save_path}/objective_value.txt", "w") as f:
        f.write(str(return_dict.get("objective_value", "N/A")))

    with open(f"{save_path}/hydrogen_penalty.txt", "w") as f:
        f.write(str(return_dict.get("hydrogen_penalty", "N/A")))

    # Save CSV data
    if "data" in return_dict:
        return_dict["data"].to_csv(f"{save_path}/results.csv", index=False)

    if "validation_results" in return_dict:
        with open(f"{save_path}/validation_results.json", "w") as f:
            json.dump(return_dict["validation_results"], f, indent=2)

    # Save quantile
    if "price_quantile" in return_dict:
        np.save(f"{save_path}/price_quantile.npy", return_dict["price_quantile"])

    # save regression_coefficients_DA and regression_coefficients_H if available
    if "regression_coefficients_DA" in return_dict:
        np.save(f"{save_path}/regression_coefficients_DA.npy", return_dict["regression_coefficients_DA"])
    if "regression_coefficients_H" in return_dict:
        np.save(f"{save_path}/regression_coefficients_H.npy", return_dict["regression_coefficients_H"])

    # Save x_train if available
    if "x_train" in return_dict:
        return_dict["x_train"].to_csv(f"{save_path}/x_train.csv", index=False)

    if "alphas_DA" in return_dict:
        np.save(f"{save_path}/alphas_DA.npy", return_dict["alphas_DA"])
    if "alphas_H" in return_dict:
        np.save(f"{save_path}/alphas_H.npy", return_dict["alphas_H"])

    print(f"Results saved to: {save_path}")
