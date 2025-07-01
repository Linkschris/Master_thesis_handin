import pandas as pd
import numpy as np
import sklearn.model_selection
import os
from typing import Callable


def merge_files(save_path: str = None):
    assert save_path is not None, "Please provide a path to save the merged file."
    if not os.path.exists(save_path):
        print(f"The following path does not exist: {save_path}")
        return

    all_files = os.listdir(save_path)[:-1]  # Exclude the last file
    all_files = [f for f in all_files if f.endswith(".csv")]

    data = pd.DataFrame()
    for file in all_files:
        file_path = os.path.join(save_path, file)

        if save_path.endswith("balancing"):
            df = pd.read_csv(file_path, index_col=None, header=0, sep=";")
        else:
            # For other files, use comma as separator
            df = pd.read_csv(file_path, index_col=None, header=0)
        data = pd.concat([data, df], axis=0, ignore_index=True)

    # Step 1: Determine correct datetime column
    if "Time (CET/CEST)" in data.columns:
        data["datetime_CET"] = data["Time (CET/CEST)"].str.split(" - ").str[0]
        data.drop(columns=["Time (CET/CEST)"], inplace=True)
    elif "Delivery Start (CET)" in data.columns:
        data["datetime_CET"] = data["Delivery Start (CET)"]
        data.drop(columns=["Delivery Start (CET)", "Delivery End (CET)"], inplace=True)
    elif "MTU (CET/CEST)" in data.columns:
        data["datetime_CET"] = data["MTU (CET/CEST)"].str.split(" - ").str[0]
        data.drop(columns=["MTU (CET/CEST)"], inplace=True)
    elif "MTU" in data.columns:
        data["datetime_CET"] = data["MTU"].str.split(" - ").str[0]
        data.drop(columns=["MTU"], inplace=True)
    else:
        raise KeyError(f"Expected datetime column not found in the data. Columns available: {list(data.columns)}")

    # Convert to datetime
    data["datetime_CET"] = pd.to_datetime(data["datetime_CET"], dayfirst=True, errors="coerce")

    # Step 2: Identify duplicated datetimes (typical for 02:00 on fall-back)
    # Only look at values that occur more than once
    duplicated = data["datetime_CET"].duplicated(keep=False)

    # Step 3: Create ambiguous flags: first occurrence = True (DST), second = False (Standard)
    ambiguous_flags = []
    count_map = {}

    for dt in data["datetime_CET"]:
        if dt not in count_map:
            count_map[dt] = 1
            ambiguous_flags.append(True)  # first occurrence, treat as DST
        elif duplicated.loc[data["datetime_CET"] == dt].any():
            count_map[dt] += 1
            ambiguous_flags.append(False)  # second occurrence, treat as CET
        else:
            ambiguous_flags.append(False)  # default for all others

    # Step 4: Localize using 'Europe/Berlin', handling ambiguous and nonexistent
    data["datetime_CET"] = data["datetime_CET"].dt.tz_localize(
        "Europe/Copenhagen", ambiguous=ambiguous_flags, nonexistent="shift_forward"
    )
    # drop datetime_CET duplöicates
    data.drop_duplicates(subset=["datetime_CET"], keep="last", inplace=True)
    # reorder the columns
    data = data[["datetime_CET"] + [col for col in data.columns if col != "datetime_CET"]]

    # if any column name contains load, rename it to load_0
    if save_path.endswith("load"):
        data.columns = ["datetime_CET", "load_FC", "load_RE"]
    elif save_path.endswith("balancing"):
        # drop columns that contain 'Activated'
        data = data.loc[:, ~data.columns.str.contains("Activated")]
        # drop columns that contain 'Down Price (EUR)' or 'Up Price (EUR)'
        data = data.loc[:, ~data.columns.str.contains(r"Down Price \(EUR\)|Up Price \(EUR\)", regex=True)]
        print(data.columns)
        data.columns = [
            "datetime_CET",
            "DK1_Accepted_Down_Volume_(MW)",
            "DK1_Accepted_Up_Volume_(MW)",
            "DK1_Imbalance_Price_(EUR)",
            "DK2_Accepted_Down_Volume_(MW)",
            "DK2_Accepted_Up_Volume_(MW)",
            "lambda_IM",
        ]

    elif save_path.endswith("generation"):
        # drop columns that contain 'Intraday' or 'Current'
        data = data.loc[:, ~data.columns.str.contains("Intraday|Current")]
        data.columns = ["datetime_CET", "Solar_FC", "Wind_offshore_FC", "Wind_onshore_FC"]
    elif save_path.endswith("unavailibility"):
        print(data.columns)
    elif save_path.endswith("residual"):
        data.drop(
            columns=[
                "Area",
                "Energy storage - Actual Aggregated [MW]",
                "Fossil Brown coal/Lignite - Actual Aggregated [MW]",
                "Fossil Coal-derived gas - Actual Aggregated [MW]",
                "Fossil Oil shale - Actual Aggregated [MW]",
                "Fossil Peat - Actual Aggregated [MW]",
                "Geothermal - Actual Aggregated [MW]",
                "Hydro Pumped Storage - Actual Aggregated [MW]",
                "Hydro Pumped Storage - Actual Consumption [MW]",
                "Hydro Run-of-river and poundage - Actual Aggregated [MW]",
                "Hydro Water Reservoir - Actual Aggregated [MW]",
                "Marine - Actual Aggregated [MW]",
                "Nuclear - Actual Aggregated [MW]",
                "Other - Actual Aggregated [MW]",
                "Other renewable - Actual Aggregated [MW]",
            ],
            inplace=True,
            errors="ignore",
        )
        print(data.columns)
        data.columns = [
            "datetime_CET",
            "Biomass_RE",
            "Fossil_Gas_RE",
            "Fossil_Hard_coal_RE",
            "Fossil_Oil_RE",
            "Solar_RE",
            "Waste_RE",
            "Wind_offshore_RE",
            "Wind_onshore_RE",
        ]
    else:
        print("No specific column renaming required.")

    print(f"Data shape after merging: {data.shape}")

    return data


def merge_outage_files(save_path: str = None):
    assert save_path is not None, "Please provide a path to save the merged file."
    if not os.path.exists(save_path):
        print(f"The following path does not exist: {save_path}")
        return

    all_files = os.listdir(save_path)
    all_files = [f for f in all_files if f.endswith(".xlsx")]
    data = pd.DataFrame()
    for file in all_files:
        file_path = os.path.join(save_path, file)
        df = pd.read_excel(file_path, index_col=None, header=None).iloc[9:]
        # rename columns
        df.rename(
            columns={
                0: "Active_status",
                1: "Scheduled",
                2: "Unit_type",
                3: "Period",
                4: "Country",
                5: "Unit",
                6: "norm_cap",
                7: "available_cap",
            },
            inplace=True,
        )
        data = pd.concat([data, df], axis=0, ignore_index=True)
    # Filter for Planned Outages
    data = data[data["Scheduled"] == "Planned outage"]
    data = data[data["Active_status"] == "Active outage"]
    data.drop(columns=["Active_status", "Scheduled", "Unit_type", "Country", "Unit"], inplace=True)

    # Extract number part before any non-digit characters, and convert to numeric
    data["norm_cap"] = pd.to_numeric(
        data["norm_cap"].astype(str).str.extract(r"^(\d+)", expand=False), errors="coerce"
    ).astype("float")
    data["available_cap"] = pd.to_numeric(
        data["available_cap"].astype(str).str.extract(r"^(\d+)", expand=False), errors="coerce"
    ).astype("float")

    data["missing_capacity_RE"] = data["norm_cap"] - data["available_cap"]

    data.reset_index(drop=True, inplace=True)
    return data


def transform_outage_data(data: pd.DataFrame):
    # Step 1: Extract start and end datetime from Period
    data[["start", "end"]] = data["Period"].str.split(" - ", expand=True)
    data["end"] = data["end"].str.extract(r"^(.*?)\s*\(")[0]

    data.drop(columns=["Period"], inplace=True)

    # Step 2: Convert to datetime (assume Europe/Berlin for CET/CEST handling)
    data["start"] = pd.to_datetime(data["start"], format="%d.%m.%Y %H:%M").dt.tz_localize("Europe/Copenhagen")
    data["end"] = pd.to_datetime(data["end"], format="%d.%m.%Y %H:%M").dt.tz_localize("Europe/Copenhagen")

    # round start and end to the next closest hour
    data["start"] = data["start"].dt.round("h")
    data["end"] = data["end"].dt.round("h")

    # print rows where start and end are NaT
    if data["start"].isna().any() or data["end"].isna().any():
        print("Rows with NaT values in start or end:")
        print(data[data["start"].isna() | data["end"].isna()])

    # Step 3: Create an empty time series for accumulating hourly data
    all_hours = pd.date_range(start=data["start"].min(), end=data["end"].max(), freq="h", tz="Europe/Copenhagen")
    hourly_data = pd.DataFrame(index=all_hours)
    hourly_data["missing_capacity_RE"] = 0.0

    # Step 4: Loop through each row and add missing_capacity to the relevant hours
    for _, row in data.iterrows():
        hours = pd.date_range(start=row["start"], end=row["end"], freq="h")
        hourly_data.loc[hours, "missing_capacity_RE"] += row["missing_capacity_RE"]

    # Now hourly_data contains total missing capacity per hour
    hourly_data.reset_index(inplace=True)
    hourly_data.rename(columns={"index": "datetime_CET"}, inplace=True)

    return hourly_data


def calculate_forecast_error(data: pd.DataFrame, forecast_col: str, actual_col: str) -> pd.DataFrame:
    """
    Calculate forecast error as the absolute difference between forecast and actual values.
    """
    name = f"{forecast_col}_error"
    data[name] = data[forecast_col] - data[actual_col]
    return data


# check for two columns, which just differ in FC and RE ending?
def check_fc_re_columns(data: pd.DataFrame) -> None:
    fc_columns = [col for col in data.columns if col.endswith("_FC")]
    re_columns = [col for col in data.columns if col.endswith("_RE")]

    for fc_col in fc_columns:
        corresponding_re_col = fc_col.replace("_FC", "_RE")
        if corresponding_re_col in re_columns:
            data = calculate_forecast_error(data, fc_col, corresponding_re_col)
            print(f"Forecast error calculated for {fc_col} and {corresponding_re_col}")

    return data


#
# def lag_re_columns(data: pd.DataFrame) -> pd.DataFrame:
#    # Ensure datetime_CET is a datetime type
#    data = data.copy()
#    data["datetime_CET"] = pd.to_datetime(data["datetime_CET"])
#
#    for col in data.columns:
#        if col.endswith("_RE"):
#            # Conditional lag based on hour in datetime_CET
#            lagged_values = []
#            for i, ts in enumerate(data["datetime_CET"]):
#                lag_hours = 24 if ts.hour < 12 else 48
#                if i >= lag_hours:
#                    lagged_values.append(data[col].iloc[i - lag_hours])
#                else:
#                    lagged_values.append(None)
#            data[f"{col}_lag_conditional"] = lagged_values
#
#        elif col.endswith("_error"):
#            # Average from two days ago 00:00 to yesterday 12:00 (fixed window)
#            mean_values = []
#            for ts in data["datetime_CET"]:
#                start = (ts - pd.Timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
#                end = (ts - pd.Timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
#                mask = (data["datetime_CET"] >= start) & (data["datetime_CET"] <= end)
#                window = data.loc[mask, col]
#                mean_values.append(window.mean() if not window.empty else None)
#            data[f"{col}_mean_win"] = mean_values
#
#    return data
#
def lag_re_columns(data: pd.DataFrame, rolling_days: int = 30) -> pd.DataFrame:
    columns_to_lag = [
        "abs_diff_markets_RE",
        "diff_markets_RE",
        "lambda_DA_RE",
        "lambda_IM",
        "missing_capacity_RE",
        "load_FC_error",
        "Solar_FC_error",
        "Wind_offshore_FC_error",
        "Wind_onshore_FC_error",
    ]

    data = data.copy()
    data["datetime_CET"] = pd.to_datetime(data["datetime_CET"])
    data.set_index("datetime_CET", inplace=True)

    for col in data.columns:
        if col in columns_to_lag:
            print("Lagging the column: " + col)
            rolling_avg = []
            for current_time in data.index:
                hour = current_time.hour
                start_time = current_time - pd.Timedelta(days=rolling_days)
                mask = (data.index >= start_time) & (data.index < current_time) & (data.index.hour == hour)
                mean_val = data.loc[mask, col].mean()
                rolling_avg.append(mean_val)
            data[f"{col}_hourly_rollavg"] = rolling_avg

    #        elif col.endswith("_error"):
    #            print("Calculating the mean error for the column: " + col)
    #            rolling_avg = []
    #            for current_time in data.index:
    #                start_time = current_time - pd.Timedelta(days=rolling_days)
    #                mask = (data.index >= start_time) & (data.index < current_time)
    #                mean_val = data.loc[mask, col].mean()
    #                rolling_avg.append(mean_val)
    #            data[f"{col}_hourly_rollavg"] = rolling_avg

    data.reset_index(inplace=True)
    return data


def import_data():
    base_path = "data/raw/"
    folder_list = ["load", "generation", "balancing", "residual"]

    full_data = pd.DataFrame()
    for folder in folder_list:
        folder_path = os.path.join(base_path, folder)
        # Merge files in the folder
        merged_data = merge_files(folder_path)

        # print first and last datetime
        print(f"First datetime in {folder}: {merged_data['datetime_CET'].min()}")
        print(f"Last datetime in {folder}: {merged_data['datetime_CET'].max()}")

        # Append to the full data
        if full_data.empty:
            full_data = merged_data
        else:
            full_data = pd.merge(full_data, merged_data, on="datetime_CET", how="inner")

    unit_data = pd.read_csv("data/raw/windfarm_data.csv", index_col=0)

    datetime_index = pd.date_range(start="1/1/2019", end="31/12/2020", freq="h", tz="CET", inclusive="left")

    unit_data.drop(
        columns=["UP", "DW", "Offshore DK2", "Offshore DK1", "Onshore DK2", "Onshore DK1"], axis=1, inplace=True
    )

    unit_data["datetime_CET"] = datetime_index
    unit_data.rename(
        columns={
            "forward_FC": "lambda_DA_FC",
            "forward_RE": "lambda_DA_RE",
            "ImbalancePriceEUR": "lambda_IM",
            "production_RE": "energy_RE",
            "production_FC": "energy_FC",
        },
        inplace=True,
    )
    full_data_set = pd.merge(full_data, unit_data, on="datetime_CET", how="inner")
    unavailability_data = merge_outage_files("data/raw/unavailability")
    unavailability_data = transform_outage_data(unavailability_data)
    full_data_set = pd.merge(full_data_set, unavailability_data, on="datetime_CET", how="left")
    full_data_set["diff_markets_RE"] = full_data_set["lambda_IM"] - full_data_set["lambda_DA_RE"]
    full_data_set["abs_diff_markets_RE"] = full_data_set["diff_markets_RE"].abs()
    full_data_set["system_state_RE"] = np.where(full_data_set["diff_markets_RE"] > 0, 1, 0)
    full_data_set["system_state_RE"] = np.where(
        full_data_set["diff_markets_RE"] < 0, -1, full_data_set["system_state_RE"]
    )
    full_data_set["system_state_RE"] = full_data_set["system_state_RE"].astype("category")

    full_data_set["date"] = full_data_set["datetime_CET"].dt.date

    full_data_set.dropna(inplace=True)

    column_df = full_data_set[["lambda_DA_FC", "energy_FC"]]
    # column_df = data[["energy_FC"]]
    zeta_list = [10]
    mu_list = np.linspace(0, 1.0, 4)  # [0.5]
    # zeta list elements shall have 2 decimal places
    mu_list = np.round(mu_list, 2)
    # print(zeta_list, flush=True)
    transformed_df = paper_kernel(column_df, mu_list, zeta_list)
    full_data_set = pd.concat([full_data_set, transformed_df], axis=1)

    # round data to 8 decimal places
    # data = data.round(6)
    full_data_set = check_fc_re_columns(full_data_set)
    full_data_set = lag_re_columns(full_data_set, rolling_days=30)
    full_data_set = full_data_set.rename(columns={"datetime_CET": "datetime"})

    full_data_set = smooth_spikes(full_data_set, "lambda_DA_FC", window=12, threshold=5.0)

    full_data_set.to_csv("data/processed/data.csv")

    full_data_set = full_data_set.iloc[24 * 50 : 24 * 150].reset_index(drop=True)
    full_data_set.to_csv("data/processed/data_small.csv")


def data_loader(data, train_ratio=0.5, val_ratio=0.00, test_ratio=0.5):
    assert train_ratio + val_ratio + test_ratio == 1, "Data split ratios must sum to 1."

    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")

    date_selecter = data.groupby("date", as_index=False).count()  # .filter(lambda x: x['date'].dt.hour.nunique() == 24)
    non_24_days = date_selecter[date_selecter["datetime"] != 24]["date"]
    data = data[~data["date"].isin(non_24_days)].reset_index(drop=True)

    split = int(len(data) * (train_ratio + val_ratio) / 24) * 24

    train_val_data = data.iloc[:split].reset_index(drop=True)

    if val_ratio != 0:
        # randomsplit for train and val data
        train_data, val_data = sklearn.model_selection.train_test_split(
            train_val_data, test_size=val_ratio / (train_ratio + val_ratio), shuffle=True, random_state=42
        )
    else:
        train_data = train_val_data.copy()
        val_data = data.iloc[split:split].reset_index(drop=True)

    test_data = data.iloc[split:].reset_index(drop=True)

    print(f"Train data: {train_data.shape[0]} hours")
    print(f"Validation data: {val_data.shape[0]} hours")
    print(f"Test data: {test_data.shape[0]} hours")

    return train_data, val_data, test_data


def paper_kernel(df, mu_list, zeta_list):
    """
    Applies the kernel transformation from the paper to each column in the DataFrame.
    The transformation follows the formula: exp(-zeta * ||mu - x||^2).
    Also plots the kernel function.
    """
    new_df = pd.DataFrame()
    # normalize df

    for col in df.columns:
        if df[col].max() != 1:
            print(f"Normalizing {col}")
            new_col_name = f"{col}_normalized"

            new_df.loc[:, new_col_name] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    for col in df.columns:
        if df[col].max() != 1:
            print(f"Applying kernel transformation to {col}")
            # standardize the column
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        # define zeta list with four quantiles
        mu_list = df[col].quantile([1 / 4, 1 / 2, 3 / 4]).values
        zeta_list = [df[col].std()]
        for mu in mu_list:
            for zeta in zeta_list:
                change_df = df[col]
                change_df = change_df.transform(lambda x: np.exp(-zeta * (mu - x) ** 2))
                transformed_column = pd.DataFrame(change_df)
                transformed_column.columns = [f"{col}_zeta_{zeta}_mu_{mu}"]
                new_df = pd.concat([new_df, transformed_column], axis=1)

    return new_df


def smooth_spikes(
    data: pd.DataFrame, column: str = "lambda_DA_FC", window: int = 12, threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect and smooth spikes in a column based on n×std deviation from neighboring values.

    Parameters:
    - data: DataFrame containing the column to clean.
    - column: The name of the column to clean.
    - window: Number of rows before and after to compute the rolling std (total window = 2*window).
    - threshold: How many stds the current value can deviate from both neighbors before it's replaced.

    Returns:
    - Cleaned DataFrame with possible values replaced.
    """
    data = data.copy()  # Don't modify the original
    values = data[column].values
    new_values = values.copy()

    counter = 0

    for i in range(window, len(values) - window - 1):
        prev_val = values[i - 1]
        curr_val = values[i]
        next_val = values[i + 1]

        local_window = np.concatenate([values[i - window : i], values[i + 1 : i + window + 1]])
        local_std = np.std(local_window)
        prev_distance = curr_val - prev_val
        next_distance = curr_val - next_val

        # Check if the spike is too sharp
        if abs(prev_distance) > threshold * local_std and abs(next_distance) > threshold * local_std:
            counter += 1
            new_values[i] = (prev_val + next_val) / 2

            print(f"Spike detected at index {i} (next): {prev_val} -> {curr_val} -> {next_val}")
            print(f"Local std: {local_std}, Threshold: {threshold * local_std}")
            print(f"Distances: {prev_distance}, {next_distance}")
            print(f"Replacing {curr_val} with {(prev_val + next_val) / 2}")

    print(f"Number of spikes detected and smoothed: {counter}")

    data[column] = new_values
    return data


def get_rolling_windows(data: pd.DataFrame, train_size: int = 15, val_size: int = 1, test_size: int = 1) -> list:
    """
    Generate rolling train/validation/test windows from a time-indexed DataFrame.

    Parameters:
    - data: full DataFrame
    - window_size: size of the test window (e.g., 24 for one day)

    Returns:
    - A list of (train_start, train_end, val_start, val_end, test_start, test_end) tuples
    """

    total_len = len(data)
    print(f"Total length of data: {total_len}")
    windows = []

    for i in range(0, total_len - (train_size + val_size + test_size) * 24, test_size * 24):
        train_start = i
        train_end = train_start + train_size * 24
        val_start = train_end
        val_end = val_start + val_size * 24
        test_start = val_end
        test_end = test_start + test_size * 24
        windows.append((train_start, train_end, val_start, val_end, test_start, test_end))

    return windows


def rbf_kernel_function(x1: np.ndarray, x2: np.ndarray, gamma: float) -> float:
    """
    Computes the RBF kernel between two vectors.
    """
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


def poly_kernel_function(x1: np.ndarray, x2: np.ndarray, degree: int = 2) -> float:
    """
    Computes the polynomial kernel between two vectors.
    """
    return (np.dot(x1, x2) + 1) ** degree


def normalized_poly_kernel_function(x1: np.ndarray, x2: np.ndarray, degree: int = 2, coef0: float = 1.0) -> float:
    """
    Computes the normalized polynomial kernel between two vectors.

    K(x1, x2) = ( (x1 · x2) / (||x1|| * ||x2||) + coef0 ) ** degree

    Parameters:
    - x1, x2: Input vectors
    - degree: Degree of the polynomial kernel
    - coef0: Independent term (bias)

    Returns:
    - Normalized polynomial kernel value
    """
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)

    # To avoid division by zero
    if norm_x1 == 0 or norm_x2 == 0:
        return 0.0

    normalized_dot = dot_product / (norm_x1 * norm_x2)
    return (normalized_dot + coef0) ** degree


def regularized_poly_kernel_function(
    x1: np.ndarray,
    x2: np.ndarray,
    degree: int = 2,
    coef0: float = 1.0,
    lambda_reg: float = 1.0,
) -> float:
    """
    Computes a regularized polynomial kernel between two vectors.

    Regularization is added to the diagonal automatically if x1 and x2 are (numerically) equal.
    """
    kernel_value = (np.dot(x1, x2) + coef0) ** degree

    # Check for diagonal entry using np.allclose (robust to tiny numeric noise)
    if np.allclose(x1, x2):
        kernel_value += lambda_reg

    return kernel_value


def analyze_kernel_matrix(K):
    U, S, Vt = np.linalg.svd(K)
    condition_number = S[0] / S[-1]

    print(f"Largest singular value: {S[0]}")
    print(f"Smallest singular value: {S[-1]}")
    print(f"Condition number: {condition_number:.2e}")


def compute_kernel_matrix(
    data: pd.DataFrame,
    kernel_function: Callable[..., float],
    **kernel_params,
) -> np.ndarray:
    """
    Computes the kernel matrix for the given data and kernel function.

    Parameters:
    - data: DataFrame containing the features.
    - kernel_function: Kernel function to use (e.g., rbf_kernel_function).
    - **kernel_params: Additional parameters for the kernel function.

    Returns:
    - Kernel matrix as a NumPy array.
    """
    assert data is not None, "Data must be provided to compute the kernel matrix."

    n_samples = data.shape[0]
    kernel_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = kernel_function(data.iloc[i].values, data.iloc[j].values, **kernel_params)

    analyze_kernel_matrix(kernel_matrix)

    # cond_K = np.linalg.cond(kernel_matrix)
    # print(kernel_matrix)
    # sns.histplot(kernel_matrix.flatten(), bins=100)
    # plt.title("Kernel matrix entry distribution")
    # plt.show()
    # print("Condition number of kernel matrix:", cond_K)
    return kernel_matrix


def filter_similar_samples(X: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    X_np = X.to_numpy()
    selected_indices = []
    for i, x in enumerate(X_np):
        if not selected_indices:
            selected_indices.append(i)
        else:
            # Compute distance to all already selected samples
            dists = np.linalg.norm(X_np[selected_indices] - x, axis=1)
            if np.min(dists) > threshold:
                selected_indices.append(i)
    return X.iloc[selected_indices]


if __name__ == "__main__":
    import_data()
    data = pd.read_csv("data/processed/data.csv", index_col=0, header=0)
