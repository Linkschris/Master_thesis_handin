import pandas as pd
import matplotlib.pyplot as plt


def analyse_val_results(df, configfile_name):
    """Analyse the results of the optimization."""
    # Load the data
    if df is None:
        return 0

    data = df.copy()

    data["datetime"] = pd.to_datetime(data["datetime"], utc=True).dt.tz_convert("Europe/Copenhagen")
    #    daily_data = data.groupby(data["datetime"].dt.date).agg(
    #        {
    #            #            "hydrogen_output": "sum",
    #            #            "consumption_electrolyser": "sum",
    #            "imbalance": "sum",
    #            "DA_revenue": "sum",
    #            #            "H_revenue": "sum",
    #            "IM_revenue": "sum",
    #            "total_revenue": "sum",
    #        }
    #    )
    #    config = OmegaConf.load(f"configs/{configfile_name}.yaml")

    #    daily_data["difference_hydrogen"] = daily_data["hydrogen_output"] - config["min_daily_hydrogen_production"]
    #    daily_data["missing_hydrogen"] = daily_data["difference_hydrogen"].clip(upper=0)
    #    daily_data["penalty_hydrogen"] = daily_data["missing_hydrogen"] * config["hydrogen_price"] * 3

    objective_value = data["total_revenue"].sum()  # + daily_data["penalty_hydrogen"].sum()
    #    hydrogen_penalty = daily_data["penalty_hydrogen"].sum()

    return objective_value  # , hydrogen_penalty


def analyse_test_results(df, configfile_name):
    """Analyse the results of the optimization."""
    # Load the data
    data = df.copy()

    data["datetime"] = pd.to_datetime(data["datetime"], utc=True).dt.tz_convert("Europe/Copenhagen")
    #    daily_data = data.groupby(data["datetime"].dt.date).agg(
    #        {
    #            #            "hydrogen_output": "sum",
    #            #            "consumption_electrolyser": "sum",
    #            "imbalance": "sum",
    #            "DA_revenue": "sum",
    #            #            "H_revenue": "sum",
    #            "IM_revenue": "sum",
    #            "total_revenue": "sum",
    #        }
    #    )
    #    config = OmegaConf.load(f"configs/{configfile_name}.yaml")

    #    daily_data["difference_hydrogen"] = daily_data["hydrogen_output"] - config["min_daily_hydrogen_production"]
    #    daily_data["missing_hydrogen"] = daily_data["difference_hydrogen"].clip(upper=0)
    #    daily_data["penalty_hydrogen"] = daily_data["missing_hydrogen"] * config["hydrogen_price"] * 3
    #
    print("These are the test results:")
    print(data.describe())

    objective_value = data["total_revenue"].sum()  # + daily_data["penalty_hydrogen"].sum()
    #    hydrogen_penalty = daily_data["penalty_hydrogen"].sum()

    print(f"Total revenue: {data['total_revenue'].sum()}")
    #    print(f"Hydrogen penalty: {hydrogen_penalty}")
    print(f"Objective value: {objective_value}")

    return objective_value  # , hydrogen_penalty


def analyse_train_results(df, configfile_name):
    """Analyse the results of the optimization."""
    # Load the data
    data = df.copy()

    data["datetime"] = pd.to_datetime(data["datetime"], utc=True).dt.tz_convert("Europe/Copenhagen")
    #    daily_data = data.groupby(data["datetime"].dt.date).agg(
    #        {
    #            #            "hydrogen_output": "sum",
    #            #            "consumption_electrolyzer": "sum",
    #            "volume_IM": "sum",
    #            "DA_revenue": "sum",
    #            #            "hydrogen_revenue": "sum",
    #            "IM_revenue": "sum",
    #        }
    #    )
    #    config = OmegaConf.load(f"configs/{configfile_name}.yaml")

    #    daily_data["difference_hydrogen"] = daily_data["hydrogen_output"] - config["min_daily_hydrogen_production"]
    #    daily_data["missing_hydrogen"] = daily_data["difference_hydrogen"].clip(upper=0)
    #    daily_data["penalty_hydrogen"] = daily_data["missing_hydrogen"] * config["hydrogen_price"] * 3

    print("These are the train results:")
    print(data.describe())
    total_revenue = data["DA_revenue"].sum() + data["IM_revenue"].sum()  # + data["hydrogen_revenue"].sum()

    objective_value = total_revenue  # + daily_data["penalty_hydrogen"].sum()

    print(f"Total revenue: {total_revenue}")
    #    print(f"Hydrogen penalty: {daily_data['penalty_hydrogen'].sum()}")
    print(f"Objective value: {objective_value}")

    return objective_value


def visualize_bidding_curves(modeltype):
    """Visualize the bidding curves for training set."""
    # Load the data
    data = pd.read_csv(f"results/train/{modeltype}/full_data/results.csv")
    # make scatterplot of lambda_DA_RE and volume_DA
    plt.scatter(data["lambda_DA_RE"], data["volume_DA"])
    plt.xlabel("Price")
    plt.ylabel("Volume")
    plt.title("Electricity bidding curves for training set")

    """Visualize the bidding curves for test set."""
    # Load the data
    data = pd.read_csv(f"results/test/{modeltype}/elec_bidding.csv")
    print(data.shape)
    print(data.head())

    data.T.plot()
    plt.xlabel("Price")
    plt.ylabel("Volume")
    plt.title("Electricity bidding curves for test set")
    plt.show()


if __name__ == "__main__":
    modeltype = "KernelPolicyModel"  # "non_linear_PolicyModel" or "KernelPolicyModel"
    configfile_name = "rolling_window"
    analyse_test_results(
        pd.read_csv(f"results/test/{modeltype}/small_data/{configfile_name}/results.csv"), configfile_name
    )
    # visualize_bidding_curves("PolicyModel")
