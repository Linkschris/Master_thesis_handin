import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def visualize_train_results(data, save_path, objective_value):
    """Visualize the results of the optimization."""
    # select one day without negative prices
    non_negative_prices = data.groupby("date").filter(lambda x: (x["lambda_DA_RE"] >= 0).all())
    neg_prices = data.groupby("date").filter(lambda x: (x["lambda_DA_RE"] < 0).any())
    day = non_negative_prices["date"].unique()[0]
    neg_day = neg_prices["date"].unique()[0]
    day_data = non_negative_prices[non_negative_prices["date"] == day].reset_index(drop=True)
    neg_day_data = neg_prices[neg_prices["date"] == neg_day].reset_index(drop=True)

    # one plot showing all the volumes and then one area chart, which shows the revenue
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    # Define width of each bar
    width = 0.2

    # Define x locations for groups
    x = np.arange(len(day_data.index))
    # Plot each bar with an offset to group them
    ax[0].bar(x - 1 * width, day_data["volume_DA"], width=width, label="DA volume")
    ax[0].bar(x - 0.5 * width, day_data["volume_IM"], width=width, label="IM volume")
    ax[0].bar(x + 0.5 * width, day_data["energy_RE"], width=width, label="RE production")
    ax[0].bar(x + 1.0 * width, day_data["consumption_electrolyzer"], width=width, label="Electrolyzer consumption")
    ax[0].set_title("Volumes")
    ax[0].legend()

    ax[1].plot(day_data["lambda_DA_RE"], label="lambda_DA_RE", marker="o")
    ax[1].plot(day_data["lambda_IM"], label="lambda_IM", marker="o")
    ax[1].grid()
    ax[1].set_title("Prices")
    ax[1].legend()

    ax[2].bar(x - 1.0 * width, day_data["DA_revenue"], width=width, label="DA revenue")
    ax[2].bar(x - 0.0 * width, day_data["IM_revenue"], width=width, label="IM revenue")
    ax[2].bar(x + 1 * width, day_data["hydrogen_revenue"], width=width, label="hydrogen_revenue")
    ax[2].set_title("Revenue")
    ax[2].legend()

    fig.suptitle(f"Objective value: {objective_value}")
    # save the plot
    plt.savefig(f"{save_path}/volume_revenue_non_neg_day.png")

    # plot the negative day
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    # Define width of each bar
    width = 0.2

    # Define x locations for groups
    x = np.arange(len(neg_day_data.index))
    # Plot each bar with an offset to group them
    ax[0].bar(x - 1.5 * width, neg_day_data["volume_DA"], width=width, label="DA volume")
    ax[0].bar(x - 0.5 * width, neg_day_data["volume_IM"], width=width, label="IM volume")
    ax[0].bar(x + 0.5 * width, neg_day_data["energy_RE"], width=width, label="RE production")
    ax[0].bar(x + 1.5 * width, neg_day_data["consumption_electrolyzer"], width=width, label="Electrolyzer consumption")
    ax[0].set_title("Volumes")
    ax[0].legend()

    ax[1].plot(neg_day_data["lambda_DA_RE"], label="lambda_DA_RE", marker="o")
    ax[1].plot(neg_day_data["lambda_IM"], label="lambda_IM", marker="o")
    ax[1].grid()
    ax[1].set_title("Prices")
    ax[1].legend()

    ax[2].bar(x - 1.5 * width, neg_day_data["DA_revenue"], width=width, label="DA revenue")
    ax[2].bar(x - 0.5 * width, neg_day_data["IM_revenue"], width=width, label="IM revenue")
    ax[2].bar(x + 0.5 * width, neg_day_data["hydrogen_revenue"], width=width, label="hydrogen_revenue")
    ax[2].set_title("Revenue")
    ax[2].legend()

    fig.suptitle(f"Objective value: {objective_value}")

    # save the plot
    plt.savefig(f"{save_path}/volume_revenue_neg_day.png")

    # plot histogram of DA_volume
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.hist(data["volume_DA"], bins=50)
    ax.set_title("DA volume histogram")
    plt.savefig(f"{save_path}/volume_DA_histogram.png")

    # plot histogram of consumption_electrolyzer
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.hist(data["consumption_electrolyzer"], bins=50)
    ax.set_title("Electrolyzer consumption histogram")
    plt.savefig(f"{save_path}/consumption_electrolyzer_histogram.png")


def visualize_test_results(save_path):
    result_df = pd.read_csv(save_path)
    # chech if there is non_linear in the path
    if "non_linear" in save_path:
        save_path = "/".join(save_path.split("/")[:-1]) + "/non_linear"
    else:
        save_path = "/".join(save_path.split("/")[:-1]) + "/linear"

    os.makedirs(save_path, exist_ok=True)
    print(save_path)
    result_df.loc[:, "system_state"] = (result_df["lambda_DA_RE"] - result_df["lambda_IM"]).apply(
        lambda x: 1 if x > 0 else -1 if x < 0 else 0
    )
    result_df.loc[:, "imbalance_guess"] = result_df["imbalance"] * result_df["system_state"]

    # make histogram of the bids_elec, imbalance
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.hist(result_df["bids_elec"], bins=50)
    ax.set_title("Bids electricity histogram")
    plt.savefig(f"{save_path}/bids_elec_histogram.png")

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.hist(result_df["imbalance"], bins=50)
    ax.set_title("Imbalance histogram")
    plt.savefig(f"{save_path}/imbalance_histogram.png")

    # save as txt file
    f = open(f"{save_path}/objective_value.txt", "w")
    f.write(str(result_df.loc[:, "total_revenue"].sum()))
    f.close()


if __name__ == "__main__":
    data = pd.read_csv("results/train/linear_PolicyModel/full_data/base_case/train_results.csv", index_col=0)
    save_path = "results/train/linear_PolicyModel/full_data/base_case"
    objective_value = 0.0
    visualize_train_results(data, save_path, objective_value, "linear_PolicyModel")
    # visualize_test_results("results/test/PolicyModel/result_non_linear.csv")
