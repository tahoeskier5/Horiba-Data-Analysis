# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:08:54 2024

@author: SFerneyhough
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore, linregress
import re


# Define folder path and filename separately
folder_path = r"\\shr-eng-srv\Share\Corrosion Test Bench\CTB Data\ELF19\20250212-N2_Test_Post_HF_Soak5"
filename = "MaxLab-2025-02-12-19_26_24-c55d51_pressure step.csv"

save_path = r'\\shr-eng-srv\Share\Corrosion Test Bench\CTB Data\ELF19\20250212-N2_Test_Post_HF_Soak5\Results'

# Construct full path
file_path = os.path.join(folder_path, filename)

# Load the CSV into a DataFrame (if needed)
data = pd.read_csv(file_path)


def select_columns(df, keywords, manual_columns):
    """
    Selects columns from the DataFrame that contain any of the specified keywords in their names,
    along with other specified manual columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame to select columns from.
        keywords (list): List of keywords to search for in column names.
        manual_columns (list): List of column names to manually include.

    Returns:
        pd.DataFrame: The DataFrame with selected columns.
    """
    # Columns that include any of the keywords in their name
    keyword_columns = [
        col for col in df.columns if any(keyword in col for keyword in keywords)
    ]
    
    # Combine keyword-matching columns with the manually specified columns
    selected_columns = keyword_columns + [col for col in manual_columns if col in df.columns]
    
    # Return the DataFrame with only the selected columns
    return df[selected_columns]



df = select_columns(data, {'P0 (PSI)','P1 (Torr)','P2 (Torr)','Flow (sccm)','Temperature'},{'Time','COM7-4:MicroRoc-0x01-1:P2 Mixed Torr-21',
                                                                                            'COM4-3:MicroRoc-0x01-1:P1 Mixed Torr-17',
                                                                                            'COM5-1:D727J-0x01-1:Setpoint Readback-10'})



df = df.drop(columns={'COM6-2:SEC-Z717SJ-0x01-1:Flow (sccm)-1',
       'COM6-2:SEC-Z717SJ-0x01-1:Temperature-6',
       'COM6-2:SEC-Z717SJ-0x02-2:Flow (sccm)-1',
       'COM6-2:SEC-Z717SJ-0x02-2:Temperature-6',
       'TCP-5:DNetP7000_FCS-P7300-0x01-1:Flow (sccm)-1',})

# #  Pressure Test 2
# df = df.drop(columns={'COM6-2:SEC-Z717SJ-0x01-1:Flow (sccm)-1',
#        'COM6-2:SEC-Z717SJ-0x02-2:Flow (sccm)-1'})



df = df.rename(columns={'COM7-4:MicroRoc-0x01-1:P2 Mixed Torr-21':'MicroRoc',
                        'COM4-3:MicroRoc-0x01-1:P1 Mixed Torr-17':'MicroRoc 2',
                        'COM5-1:D727J-0x01-1:Setpoint Readback-10':'Setpoint'})

df['Time'] = pd.to_datetime(df['Time'],unit='ms', errors='coerce')





# ########################################################
# # Create a mask for non-NaN values in MicroRoc
# non_nan_mask = df['MicroRoc'].notna()
# # Generate a counter for non-NaN values
# df['NonNaNIndex'] = (non_nan_mask.cumsum() - 4)
# # Apply the every 5th value logic while keeping the rest of the DataFrame rows
# df.loc[~((df['NonNaNIndex'] % 5 == 0) | (~non_nan_mask)), 'MicroRoc'] = None
# # Drop the helper column
# df.drop(columns=['NonNaNIndex'], inplace=True)
# ########################################################












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
    Includes all associated columns for each device, the column containing 'Setpoint', and the optional time column.
    Excludes rows where the 'Setpoint' value is 0 and where all other values (except the time column) are NaNs.

    Parameters:
        df (pd.DataFrame): The input DataFrame to split.
        time_column (str, optional): The name of the time column to include in all resulting DataFrames.

    Returns:
        dict: A dictionary where keys are device identifiers and values are DataFrames for each device.
    """
    # Ensure required columns exist
    required_columns = ["MicroRoc", "MicroRoc 2"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The required column '{col}' is missing from the DataFrame.")
    
    if time_column and time_column not in df.columns:
        raise ValueError(f"The specified time column '{time_column}' is missing from the DataFrame.")

    # Ensure there is a global Setpoint column
    setpoint_column = next((col for col in df.columns if "Setpoint" in col), None)
    if not setpoint_column:
        raise ValueError("No column containing the keyword 'Setpoint' was found in the DataFrame.")

    # Dictionary to hold the resulting DataFrames
    device_dataframes = {}

    # Add MicroRoc and MicroRoc 2 DataFrames to the dictionary
    microroc_columns = ["MicroRoc", "MicroRoc 2"]
    if time_column:
        microroc_columns.append(time_column)

    microroc_df = df[microroc_columns].copy()
    # Drop rows where all values except the time column are NaN
    subset_columns = [col for col in microroc_columns if col != time_column]
    microroc_df = microroc_df.dropna(how="all", subset=subset_columns)
    # Add to the dictionary
    device_dataframes["MicroRoc"] = microroc_df[["MicroRoc", time_column]].dropna()
    device_dataframes["MicroRoc 2"] = microroc_df[["MicroRoc 2", time_column]].dropna()

    # Extract unique device prefixes from the column names
    device_prefixes = {
        ":".join(col.split(":")[:2]) for col in df.columns if ":" in col
    }

    # Iterate through each device prefix
    for device_prefix in device_prefixes:
        # Select all columns that belong to this device
        device_columns = [col for col in df.columns if col.startswith(device_prefix)]

        # Include the time column and the global Setpoint column
        additional_columns = [time_column, setpoint_column] if time_column else [setpoint_column]
        relevant_columns = list(set(device_columns + additional_columns))

        # Create a DataFrame for the device
        device_df = df[relevant_columns].copy()

        # Exclude rows where the Setpoint value is 0
        device_df = device_df[(device_df[setpoint_column] <= 750) & (device_df[setpoint_column] != 0)]

        # Drop rows where all values except the time column and Setpoint column are NaNs
        subset_columns = [col for col in relevant_columns if col not in additional_columns]
        device_df = device_df.dropna(how="all", subset=subset_columns)

        # Add the device DataFrame to the dictionary
        device_dataframes[device_prefix] = device_df

    return device_dataframes



individual_dfs = split_into_device_dataframes(df, 'Time')


def add_cycle_column(dataframes, column_keyword):
    """
    Adds a 'Cycle' column to each DataFrame in a dictionary, incrementing every time
    the values in a column containing the specified keyword change from their maximum value.
    Skips the 'MicroRoc' and 'MicroRoc 2' DataFrames if present.

    Parameters:
        dataframes (dict): A dictionary where keys are identifiers and values are DataFrames.
        column_keyword (str): A keyword to identify the column to track changes.

    Returns:
        dict: A new dictionary of DataFrames, each with an additional 'Cycle' column.
    """
    updated_dataframes = {}

    for key, df in dataframes.items():
        # Skip the 'MicroRoc' and 'MicroRoc 2' DataFrames
        if key in {"MicroRoc", "MicroRoc 2"}:
            updated_dataframes[key] = df  # Keep it as is
            continue

        # Find the column containing the keyword
        matching_columns = [col for col in df.columns if column_keyword in col]

        if not matching_columns:
            raise ValueError(f"No column containing the keyword '{column_keyword}' found in DataFrame '{key}'.")

        if len(matching_columns) > 1:
            raise ValueError(
                f"Multiple columns containing the keyword '{column_keyword}' found in DataFrame '{key}': {matching_columns}. "
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
        df = df.copy()  # Avoid SettingWithCopyWarning
        df["Cycle"] = cycle_counter

        # Store the updated DataFrame in the dictionary
        updated_dataframes[key] = df

    return updated_dataframes



individual_dfs = add_cycle_column(individual_dfs, 'Setpoint')

def combine_dataframes_with_device_names(dataframes_dict):
    """
    Combines a dictionary of DataFrames into a single DataFrame, adding a 'Device Name' column 
    with the dictionary key, stripping the device-specific prefixes from column names, and 
    merging the MicroRoc and MicroRoc 2 DataFrames onto other device DataFrames by the nearest 'Time'.

    Parameters:
        dataframes_dict (dict): A dictionary where keys are device names and values are DataFrames.

    Returns:
        pd.DataFrame: A combined DataFrame with a 'Device Name' column, normalized column names, 
                      and the MicroRoc and MicroRoc 2 data aligned by the nearest 'Time'.
    """
    # Separate the MicroRoc and MicroRoc 2 DataFrames
    microroc_df = dataframes_dict.pop("MicroRoc", None)
    microroc2_df = dataframes_dict.pop("MicroRoc 2", None)

    # Ensure MicroRoc and MicroRoc 2 DataFrames are sorted by Time
    if microroc_df is not None:
        microroc_df = microroc_df.sort_values("Time")
    if microroc2_df is not None:
        microroc2_df = microroc2_df.sort_values("Time")

    combined_df = []

    for device_name, df in dataframes_dict.items():
        # Add the 'Device Name' column
        df = df.copy()
        df["Device Name"] = device_name

        # Normalize column names by stripping the device-specific prefixes
        normalized_columns = {}
        for col in df.columns:
            # Keep the part of the column name after the last ':'
            if ":" in col:
                normalized_columns[col] = col.split(":")[-1]
            else:
                normalized_columns[col] = col  # Keep the column name as is if no prefix

        # Rename the columns
        df = df.rename(columns=normalized_columns)

        # Merge with MicroRoc DataFrame on the nearest 'Time'
        if microroc_df is not None:
            df = pd.merge_asof(
                df.sort_values("Time"),
                microroc_df[["Time", "MicroRoc"]],
                on="Time",
                direction="nearest"
            )

        # Merge with MicroRoc 2 DataFrame on the nearest 'Time'
        if microroc2_df is not None:
            df = pd.merge_asof(
                df.sort_values("Time"),
                microroc2_df[["Time", "MicroRoc 2"]],
                on="Time",
                direction="nearest"
            )

        # Append the updated DataFrame to the list
        combined_df.append(df)

    # Combine all DataFrames into one
    return pd.concat(combined_df, ignore_index=True)



combined_df = combine_dataframes_with_device_names(individual_dfs)

combined_df = combined_df[(combined_df['Setpoint'] - combined_df['MicroRoc']) < 50]

mean_time = pd.to_datetime(combined_df['Time']).mean().date()

grouped_df = (
    combined_df.groupby(['Setpoint', 'Device Name', 'Cycle'], group_keys=False)  # Group by the specified columns
    .apply(
        lambda g: g.iloc[:-2]  # Exclude the last 5 rows in each group
        .tail(8)  # Take the last 10 of the remaining rows
        .drop(columns=['Setpoint', 'Time', 'Cycle', 'Device Name'])  # Drop specified columns
        .mean()  # Compute the mean of the remaining numeric columns
    )
    .reset_index()  # Reset the index to flatten the grouped structure
)


grouped_df['P0 Difference from MicroRoc [Torr]'] =  grouped_df['P0 (Torr)-5'] - grouped_df['MicroRoc']
grouped_df['P0 Accuracy [%RDG]'] = grouped_df['P0 Difference from MicroRoc [Torr]'] / grouped_df['MicroRoc'] * 100

grouped_df['P1 Difference from MicroRoc [Torr]'] = grouped_df['P1 (Torr)-2'] - grouped_df['MicroRoc'] 
grouped_df['P1 Accuracy [%RDG]'] = grouped_df['P1 Difference from MicroRoc [Torr]'] / grouped_df['MicroRoc'] * 100

grouped_df['P2 Difference from MicroRoc [Torr]'] =  grouped_df['P2 (Torr)-3'] - grouped_df['MicroRoc']
grouped_df['P2 Accuracy [%RDG]'] = grouped_df['P2 Difference from MicroRoc [Torr]'] / grouped_df['MicroRoc'] * 100



# Exclude the 'Time' column from the DataFrame before grouping
filtered_combined_dfs = grouped_df.drop(columns=['Time'], errors='ignore')

# Group by 'Device Name', 'Setpoint Readback-10', and 'Cycle'
combined_group = filtered_combined_dfs.groupby(['Device Name', 'Setpoint', 'Cycle'])

# Calculate descriptive statistics limited to specific metrics
stats_df = (
    combined_group
    .agg(['count', 'mean', 'min', 'max', 'median', 'std'])
    .reset_index()
)

# Flatten the columns
stats_df.columns = [f"{x}_{y}" if y else f"{x}" for x, y in stats_df.columns.to_flat_index()]

# stats_df.to_csv('stats_df.csv')
stats_df['Date'] = mean_time


# Filter columns containing '_mean'
mean_columns = [col for col in stats_df.columns if '_mean' in col]

# Include grouping keys ('Device Name', 'Setpoint Readback-10', 'Cycle')
grouping_keys = ['Device Name', 'Setpoint', 'Cycle']
mean_columns = grouping_keys + mean_columns

# Create the filtered DataFrame
stats_mean = stats_df[mean_columns]

stats_mean_group = stats_mean.groupby(['Device Name','Setpoint'])

# Aggregate with mean, std, min, max, and count
results_df = stats_mean_group.agg(['mean', 'std', 'min', 'max', 'count'])

# Flatten the columns of results_df
results_df.columns = [f"{x}_{y}" if y else f"{x}" for x, y in results_df.columns.to_flat_index()]

# Filter columns containing 'sccm' and reset index
results_df_P2 = results_df.filter(regex='P2|P1').reset_index()



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



# save results
save_result_to_csv(stats_df,save_path,filename,suffix='_stats')
save_result_to_csv(results_df_P2,save_path,filename,suffix='_results')