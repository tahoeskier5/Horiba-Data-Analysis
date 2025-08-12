# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:46:28 2024

@author: SFerneyhough
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore


# Define folder path and filename separately
folder_path = r"\\shr-eng-srv\Share\Corrosion Test Bench\CTB Data\ELF19\20250212-N2_Test_Post_HF_Soak5"
filename = "MaxLab-2025-02-12-15_50_11-62a799_fujikin.csv"

save_path = r'\\shr-eng-srv\Share\Corrosion Test Bench\CTB Data\ELF19\20250212-N2_Test_Post_HF_Soak5\Results'

# Construct full path
file_path = os.path.join(folder_path, filename)

# Load the CSV into a DataFrame (if needed)
data = pd.read_csv(file_path)


def select_columns(df, keywords=None, manual_columns=None):
    """
    Selects columns from the DataFrame that contain any of the specified keywords in their names,
    along with other specified manual columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame to select columns from.
        keywords (list, optional): List of keywords to search for in column names. If None, ignore keyword matching.
        manual_columns (list or set, optional): List or set of column names to manually include.

    Returns:
        pd.DataFrame: The DataFrame with selected columns.
    """
    if keywords is None:
        keywords = []  # Default to an empty list if no keywords are provided

    if manual_columns is None:
        manual_columns = []  # Default to an empty list if no manual columns are provided

    # Ensure manual_columns is a list (to handle sets)
    manual_columns = list(manual_columns)

    # Columns that include any of the keywords in their name
    keyword_columns = [
        col for col in df.columns if any(keyword in col for keyword in keywords)
    ]
    
    # Combine keyword-matching columns with the manually specified columns
    selected_columns = keyword_columns + [col for col in manual_columns if col in df.columns]
    
    # Return the DataFrame with only the selected columns
    return df[selected_columns]


df = select_columns(data, keywords=None, manual_columns={'Time','COM4-3:MicroRoc-0x01-1:P1 Mixed Torr-17',
                                                            'TCP-5:DNetP7000_FCS-P7300-0x01-1:Flow (sccm)-1',
                                                            'TCP-5:DNetP7000_FCS-P7300-0x01-1:Pressure (Torr)-2',
                                                            'TCP-5:DNetP7000_FCS-P7300-0x01-1:Valve Voltage-3',
                                                            'TCP-5:DNetP7000_FCS-P7300-0x01-1:SETPOINT-4',
                                                            'COM6-2:SEC-Z717SJ-0x01-1:Temperature-6'})

df = df.rename(columns={'COM4-3:MicroRoc-0x01-1:P1 Mixed Torr-17':'MicroRoc',
                                  'COM6-2:SEC-Z717SJ-0x01-1:Temperature-6':'Temperature'})




# ########################################################
# # Create a mask for non-NaN values in MicroRoc
# non_nan_mask = df['MicroRoc'].notna()
# # Generate a counter for non-NaN values
# df['NonNaNIndex'] = non_nan_mask.cumsum()
# # Apply the every 5th value logic while keeping the rest of the DataFrame rows
# df.loc[~(((df['NonNaNIndex'] - 3) % 5 == 0) | (~non_nan_mask)), 'MicroRoc'] = None
# # Drop the helper column
# df.drop(columns=['NonNaNIndex'], inplace=True)
# ########################################################




df['Time'] = pd.to_datetime(df['Time'],unit='ms', errors='coerce')

def convert_psi_to_torr(df, psi_keyword="(PSI)", torr_keyword="(Torr)"):
    """
    Converts columns in PSI to Torr in a DataFrame.
    The conversion factor is 1 PSI = 51.7149 Torr.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        psi_keyword (str): The keyword to identify PSI columns in the DataFrame.
        torr_keyword (str): The keyword to replace PSI with Torr in the column names.

    Returns:
        pd.DataFrame: A DataFrame with converted PSI values to Torr and renamed columns.
    """
    # Conversion factor
    psi_to_torr = 51.7149

    # Create a copy of the DataFrame to avoid modifying the original
    df_converted = df.copy()

    # Identify columns with the psi_keyword
    psi_columns = [col for col in df.columns if psi_keyword in col]

    # Convert PSI values to Torr and rename columns
    for col in psi_columns:
        # Perform the conversion
        df_converted[col] = df_converted[col] * psi_to_torr
        
        # Rename the column to indicate the values are now in Torr
        new_col_name = col.replace(psi_keyword, torr_keyword)
        df_converted.rename(columns={col: new_col_name}, inplace=True)

    return df_converted

df = convert_psi_to_torr(df)


def split_into_device_dataframes(df, time_column=None):
    """
    Groups columns by their device identifier (prefix) and creates separate DataFrames for each device.
    Includes all associated columns for each device and the optional time column.
    Excludes rows where the 'Setpoint' value is 0 and where all other values (except the time column) are NaNs.

    The 'MicroRoc' column is stored as a separate key in the resulting dictionary along with the time column.
    Strips device names from column names.

    Parameters:
        df (pd.DataFrame): The input DataFrame to split.
        time_column (str, optional): The name of the time column to include in all resulting DataFrames.

    Returns:
        dict: A dictionary where keys are device identifiers and values are DataFrames for each device,
              and the 'MicroRoc' column (with the time column) is stored under the key 'MicroRoc'.
    """
    # Ensure 'MicroRoc' exists in the DataFrame
    if "MicroRoc" not in df.columns:
        raise ValueError("The 'MicroRoc' column is missing from the DataFrame.")

    # Ensure the time column is included
    if time_column and time_column not in df.columns:
        raise ValueError(f"The specified time column '{time_column}' is missing from the DataFrame.")

    # Dictionary to hold the resulting DataFrames
    device_dataframes = {}

    # Add the MicroRoc DataFrame to the dictionary with the time column
    microroc_columns = ["MicroRoc"]
    if time_column:
        microroc_columns.append(time_column)
    microroc_df = df[microroc_columns].copy()

    # Drop rows where all values except the time column are NaN
    subset_columns = [col for col in microroc_columns if col != time_column]
    microroc_df = microroc_df.dropna(how="all", subset=subset_columns)

    # Add the MicroRoc DataFrame to the dictionary
    device_dataframes["MicroRoc"] = microroc_df

    # Extract unique device prefixes from the column names
    device_prefixes = {
        ":".join(col.split(":")[:2]) for col in df.columns if ":" in col
    }

    # Iterate through each device prefix
    for device_prefix in device_prefixes:
        # Select all columns that belong to this device
        device_columns = [col for col in df.columns if col.startswith(device_prefix)]

        # Find the Setpoint column for the device
        setpoint_column = next((col for col in device_columns if "SETPOINT" in col.upper()), None)
        if not setpoint_column:
            print(f"Setpoint column not found for device {device_prefix}. Skipping.")
            continue

        # Include the time column
        additional_columns = [time_column] if time_column else []

        # Ensure the global Temperature column is included
        if "Temperature" in df.columns:
            additional_columns.append("Temperature")

        relevant_columns = list(set(device_columns + additional_columns))

        # Create a DataFrame for the device
        device_df = df[relevant_columns].copy()

        # Exclude rows where the Setpoint value is 0
        device_df = device_df[device_df[setpoint_column] != 0]

        # Drop rows where all values except the time column are NaNs
        subset_columns = [col for col in relevant_columns if col != time_column]
        device_df = device_df.dropna(how="all", subset=subset_columns)

        # Rename columns to strip the device name prefix
        stripped_columns = {col: col.split(":")[-1] for col in device_df.columns}
        device_df = device_df.rename(columns=stripped_columns)

        # Add the device DataFrame to the dictionary
        device_dataframes[device_prefix] = device_df

    return device_dataframes




individual_dfs = split_into_device_dataframes(df, 'Time')
individual_dfs['TCP-5:DNetP7000_FCS-P7300-0x01-1'] = individual_dfs['TCP-5:DNetP7000_FCS-P7300-0x01-1'].ffill()




def combine_dataframes_with_device_names(dataframes_dict):
    """
    Combines a dictionary of DataFrames into a single DataFrame, adding a 'Device Name' column 
    with the dictionary key, stripping the device-specific prefixes from column names, and 
    merging the MicroRoc DataFrame onto other device DataFrames by the nearest 'Time'.

    Parameters:
        dataframes_dict (dict): A dictionary where keys are device names and values are DataFrames.

    Returns:
        pd.DataFrame: A combined DataFrame with a 'Device Name' column, normalized column names, 
                      and the MicroRoc data aligned by the nearest 'Time'.
    """
    # Separate the MicroRoc DataFrame
    microroc_df = dataframes_dict.pop("MicroRoc", None)
    if microroc_df is not None:
        microroc_df = microroc_df.sort_values("Time")  # Ensure MicroRoc DataFrame is sorted by Time

    combined_df = []

    for device_name, df in dataframes_dict.items():
        # Add the 'Device Name' column
        df = df.copy()
        df['Device Name'] = device_name

        # Normalize column names by stripping the device-specific prefixes
        normalized_columns = {}
        for col in df.columns:
            # Keep the part of the column name after the last ':'
            if ':' in col:
                normalized_columns[col] = col.split(':')[-1]
            else:
                normalized_columns[col] = col  # Keep the column name as is if no prefix

        # Rename the columns
        df = df.rename(columns=normalized_columns)

        # Merge with MicroRoc DataFrame on the nearest 'Time'
        if microroc_df is not None:
            df = pd.merge_asof(
                microroc_df[["Time", "MicroRoc"]],  # Include only relevant columns from MicroRoc
                df.sort_values("Time"),
                on="Time",
                direction="nearest"
            )

        # Append the updated DataFrame to the list
        combined_df.append(df)

    # Combine all DataFrames into one
    return pd.concat(combined_df, ignore_index=True)

combined_df = combine_dataframes_with_device_names(individual_dfs)

combined_df['Temperature'] = combined_df['Temperature'].ffill()
combined_df = combined_df[combined_df['SETPOINT-4'].notna()]


def add_cycle_column(df, column_keyword):
    """
    Adds a 'Cycle' column to the DataFrame, incrementing every time
    the values in a column containing the specified keyword change from their maximum value.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_keyword (str): A keyword to identify the column to track changes.

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'Cycle' column.
    """
    # Find the column containing the keyword
    matching_columns = [col for col in df.columns if column_keyword in col]

    if not matching_columns:
        raise ValueError(f"No column containing the keyword '{column_keyword}' found in the DataFrame.")

    if len(matching_columns) > 1:
        raise ValueError(
            f"Multiple columns containing the keyword '{column_keyword}' found in the DataFrame: {matching_columns}. "
            "Please refine the keyword."
        )

    column_name = matching_columns[0]

    # Identify the maximum value of the column
    max_value = df[column_name].max()

    # Create a boolean series where the column value is not equal to the max value
    is_not_max = df[column_name] != max_value

    # Compute a shift to detect transitions from max to non-max
    transitions = is_not_max.astype(int).diff().fillna(0)

    # Increment the cycle counter on transitions from max to non-max
    cycle_counter = (transitions > 0).cumsum()

    # Add the Cycle column to the DataFrame
    df['Cycle'] = cycle_counter

    return df



combined_df = add_cycle_column(combined_df, 'SETPOINT-4')




def add_scc_column(df, pressure_column, temperature_column, volume=435, Z=1):
    """
    Adds an 'scc' column to a DataFrame by calculating the Standard Cubic Centimeters (SCC)
    based on the given pressure and temperature columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        pressure_column (str): The name of the pressure column.
        temperature_column (str): The name of the temperature column.
        volume (float, optional): Volume in cc (default: 10).
        Z (float, optional): Compressibility factor (default: 1).

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'scc' column.
    """
    def calcSCC(Pressure, Temperature, Volume=435, Z=1):
        """
        Calculates the volume of gas in standard cubic centimeters.

        Parameters
        ----------
        Pressure : float
            Pressure in Torr.
        Temperature : float
            Temperature in Kelvin.
        Volume : float
            Volume in cc.
        Z : float
            Compressibility factor.

        Returns
        -------
        scc : float
            Standard cubic centimeters.
        """
        return Pressure / 760 * Volume * 273.15 / Temperature / Z

    # Ensure the required columns exist
    if pressure_column not in df.columns:
        raise ValueError(f"The column '{pressure_column}' is missing from the DataFrame.")
    if temperature_column not in df.columns:
        raise ValueError(f"The column '{temperature_column}' is missing from the DataFrame.")

    # Calculate the SCC and add it as a new column
    df['scc'] = df.apply(
        lambda row: calcSCC(
            Pressure=row[pressure_column],
            Temperature=(row[temperature_column]+273.15),
            Volume=volume,
            Z=Z
        ),
        axis=1
    )

    return df

combined_df = add_scc_column(combined_df, 'MicroRoc', 'Temperature')

combined_df['dscc'] = combined_df['scc'].diff()
# Calculate the time difference between consecutive rows in minutes
combined_df['dt [min]'] = combined_df['Time'].diff().dt.total_seconds() / 60

combined_df['sccm'] = combined_df['dscc'] / combined_df['dt [min]']

combined_df = combined_df[combined_df['dscc'] > 0]
combined_df = combined_df[
    (combined_df['MicroRoc'] > 10) & 
    (combined_df['MicroRoc'] < 40)]



def filter_by_iqr_and_zscore(df, value_column, group_columns):
    """
    Filters the DataFrame, grouped by specified columns, by keeping only rows 
    between the 25th and 75th percentiles of the value column and then filters 
    further by z-scores within ±1. Adds columns for group min, group max, and z-score.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        value_column (str): The column to filter by using IQR and z-scores.
        group_columns (list of str): The columns to group by.

    Returns:
        pd.DataFrame: The filtered DataFrame with additional 'group_min', 'group_max', and 'zscore' columns.
    """
    # Ensure the value column exists
    if value_column not in df.columns:
        raise ValueError(f"The column '{value_column}' does not exist in the DataFrame.")

    # Ensure all group columns exist
    missing_columns = [col for col in group_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following group columns are missing from the DataFrame: {missing_columns}")

    def filter_by_quartiles_and_zscore(group):
        # Calculate the 25th and 75th percentiles
        q1 = group[value_column].quantile(0.1)
        q3 = group[value_column].quantile(0.9)

        # Filter the group by the interquartile range
        filtered_group = group[(group[value_column] >= q1) & (group[value_column] <= q3)].copy()

        # Skip small groups (with <= 1 row) for z-score calculation
        if len(filtered_group) > 1:
            # Calculate z-scores for the filtered group
            filtered_group['zscore'] = zscore(filtered_group[value_column])
            
            # Filter further by z-scores within ±1
            filtered_group = filtered_group[(filtered_group['zscore'] >= -3) & (filtered_group['zscore'] <= 3)]
        else:
            filtered_group['zscore'] = 0  # Assign z-score of 0 for small groups

        # Calculate group min and max after all filtering
        filtered_group['group_min'] = filtered_group[value_column].min()
        filtered_group['group_max'] = filtered_group[value_column].max()

        return filtered_group

    # Group the DataFrame and apply the filtering and z-score logic
    grouped = df.groupby(group_columns, group_keys=False, dropna=True)
    processed_groups = [filter_by_quartiles_and_zscore(group.reset_index(drop=True)) for name, group in grouped]

    # Concatenate the processed groups back together
    return pd.concat(processed_groups).reset_index(drop=True)


combined_df_filt = filter_by_iqr_and_zscore(combined_df, 'sccm', ['Device Name','SETPOINT-4','Cycle'])


# Exclude the 'Time' column before grouping (if it exists)
filtered_combined_df = combined_df_filt.drop(columns=['Time'], errors='ignore')

# Group by relevant columns
combined_group = filtered_combined_df.groupby(['Device Name', 'SETPOINT-4', 'Cycle'])

# Calculate descriptive statistics
stats_df = (
    combined_group
    .agg(['count', 'mean', 'min', 'max', 'median', 'std'])
    .reset_index()
)

# Flatten the columns of stats_df
stats_df.columns = [f"{x}_{y}" if y else f"{x}" for x, y in stats_df.columns.to_flat_index()]

# Filter for columns containing '_mean' and include grouping keys
mean_columns = ['Device Name', 'SETPOINT-4', 'Cycle'] + [col for col in stats_df.columns if '_mean' in col]
stats_mean = stats_df[mean_columns]

# Group by 'Device Name' and 'Setpoint Readback-10'
stats_mean_group = stats_mean.groupby(['Device Name', 'SETPOINT-4'])

# Aggregate with mean, std, min, max, and count
results_df = stats_mean_group.agg(['mean', 'std', 'min', 'max', 'count'])

# Flatten the columns of results_df
results_df.columns = [f"{x}_{y}" if y else f"{x}" for x, y in results_df.columns.to_flat_index()]

# Filter columns containing 'sccm' and reset index
results_df_sccm = results_df.filter(like='sccm').reset_index()

results_df_sccm['sccm STD [%RDG]'] = results_df_sccm['sccm_mean_std'] / results_df_sccm['sccm_mean_mean'] * 100




def save_result_to_csv(result_df, folder_path, filename, suffix="_results"):
    """
    Saves the result DataFrame as a CSV in the specified folder,
    with a customizable suffix appended to the filename.
    
    Parameters:
        result_df (pd.DataFrame): The DataFrame to save.
        folder_path (str): Path to the folder where the file will be saved.
        filename (str): Name of the original file (used for naming the result file).
        suffix (str): Custom suffix to append to the filename before saving.
                      Default is '_results'.
    """
    # Construct the output filename with the given suffix
    base_name, ext = os.path.splitext(filename)
    result_filename = f"{base_name}{suffix}.csv"
    
    # Construct the full output file path
    result_file_path = os.path.join(folder_path, result_filename)
    
    # Save the DataFrame to CSV
    result_df.to_csv(result_file_path, index=False)
    print(f"Results saved to: {result_file_path}")




save_result_to_csv(results_df_sccm, save_path, filename,suffix='_results')
save_result_to_csv(stats_df, save_path, filename,suffix='_stats')

