# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:24:20 2024

@author: SFerneyhough
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import re


# Define folder path and filename separately

folder_path = r"\\shr-eng-srv\Share\Corrosion Test Bench\CTB Data\ELF19\20250212-N2_Test_Post_HF_Soak5"

filename = "MaxLab-2025-02-12-21_37_45-bd5b6a_seat leak.csv"

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


df = select_columns(data, {'P0 (PSI)','P1 (Torr)','P2 (Torr)','Flow (Sccm)','Temperature'},{'Time'})

# df = df.drop(columns={
#        'COM6-2:SEC-Z717SJ-0x01-1:Temperature-6',
#        'COM6-2:SEC-Z717SJ-0x02-2:Temperature-6',
#        })

# # baseline folder
# df = df.drop(columns={'COM6-2:SEC-Z717SJ-0x01-1:Flow (sccm)-1',
#        'COM6-2:SEC-Z717SJ-0x02-2:Flow (sccm)-1'})



# ########################################################
# # Create a mask for non-NaN values in MicroRoc
# non_nan_mask = df['P2 (Torr)'].notna()

# # Generate a counter for non-NaN values
# df['NonNaNIndex'] = non_nan_mask.cumsum()

# # Skip the very first instance and then take every 5th value
# # Modify the condition to exclude the first instance and then apply the modulo
# df.loc[~(((df['NonNaNIndex'] - 0) % 5 == 0) | (~non_nan_mask)), 'P2 (Torr)'] = None

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
    Groups columns by their device identifier (prefix), removes the device prefix from column names,
    and creates separate DataFrames for each device. Includes the optional time column in all DataFrames.

    Parameters:
        df (pd.DataFrame): The input DataFrame to split.
        time_column (str, optional): The name of the time column to include in all resulting DataFrames.

    Returns:
        dict: A dictionary where keys are device identifiers and values are DataFrames for each device.
    """
    # Ensure the time column exists (if specified)
    if time_column and time_column not in df.columns:
        raise ValueError(f"The specified time column '{time_column}' is missing from the DataFrame.")

    # Dictionary to hold the resulting DataFrames
    device_dataframes = {}

    # Extract unique device prefixes from the column names
    device_prefixes = {
        ":".join(col.split(":")[:2]) for col in df.columns if ":" in col
    }

    # Iterate through each device prefix
    for device_prefix in device_prefixes:
        # Select all columns that belong to this device
        device_columns = [col for col in df.columns if col.startswith(device_prefix)]

        # Include the time column if specified
        additional_columns = [time_column] if time_column else []
        relevant_columns = list(set(device_columns + additional_columns))

        # Create a DataFrame for the device
        device_df = df[relevant_columns].copy()

        # Remove the device prefix from the column names
        new_column_names = {col: col.split(":")[-1] for col in device_columns}
        if time_column:
            new_column_names[time_column] = time_column  # Keep the time column name unchanged

        # Rename columns
        device_df = device_df.rename(columns=new_column_names)

        # Drop rows where all values except the time column are NaNs
        subset_columns = [col for col in device_df.columns if col != time_column]
        device_df = device_df.dropna(how="all", subset=subset_columns).reset_index(drop=True)

        # Add the device DataFrame to the dictionary
        device_dataframes[device_prefix] = device_df

    return device_dataframes


individual_dfs = split_into_device_dataframes(df, 'Time')





# individual_dfs_filt = filter_data_by_keyword(individual_dfs, 'P2 (Torr)', 0.01, 200, 'P0')

def filter_data_by_flow(dataframes_dict, flow_column_keyword, pressure_column_keyword):
    """
    Filters DataFrames in a dictionary based on:
      - A "Flow" column where values are within a specified range,
      - A "Pressure" column where the difference between consecutive rows (.diff()) is within a specified range,
      - Drops rows where Pressure values are below a threshold based on the range of Pressure values,
      - Drops the first 20 rows before filtering.

    Parameters:
        dataframes_dict (dict): Dictionary of DataFrames to process.
        flow_column_keyword (str): Keyword to identify the "Flow" column to filter by.
        pressure_column_keyword (str): Keyword to identify the "Pressure" column for .diff() filtering.

    Returns:
        dict: A new dictionary with filtered DataFrames.
    """
    filtered_dataframes = {}

    for key, df in dataframes_dict.items():
        # Drop the first 20 rows
        df = df.iloc[20:].reset_index(drop=True)

        # Find the Flow column
        flow_column = next((col for col in df.columns if flow_column_keyword in col), None)
        if not flow_column:
            print(f"Flow column with keyword '{flow_column_keyword}' not found in DataFrame '{key}'. Skipping.")
            continue

        # Find the Pressure column
        pressure_column = next((col for col in df.columns if pressure_column_keyword in col), None)
        if not pressure_column:
            print(f"Pressure column with keyword '{pressure_column_keyword}' not found in DataFrame '{key}'. Skipping.")
            continue

        # Calculate the .diff() for the Pressure column
        df['pressure_diff'] = df[pressure_column].diff()

        # Filter rows where:
        # - The Flow value is between 0 and 20 (inclusive)
        # - A specific condition on another column (`P0 (Torr)-5`) is satisfied
        # - The difference (diff) in Pressure is > 0 and < 200
        filtered_df = df[(df[flow_column] < 50) &
                         (df[flow_column] != 0) &
                         (df['P0 (Torr)-5'] > 1500) &
                         (df['pressure_diff'] > 0) &
                         (df['pressure_diff'] < 250)].copy()

        # Calculate the range of pressure values
        max_pressure = filtered_df[pressure_column].max()
        min_pressure = filtered_df[pressure_column].min()
        pressure_range = max_pressure - min_pressure

        # Calculate thresholds based on the range
        lower_bound = min_pressure + 0.05 * pressure_range
        upper_bound = min_pressure + 0.95 * pressure_range

        # Filter based on the thresholds
        filtered_df = filtered_df[(filtered_df[pressure_column] >= lower_bound) & 
                                  (filtered_df[pressure_column] <= upper_bound)]

        # Drop the temporary 'pressure_diff' column
        # filtered_df = filtered_df.drop(columns=['pressure_diff'])

        # Add the filtered DataFrame to the output dictionary
        filtered_dataframes[key] = filtered_df.reset_index(drop=True)

    return filtered_dataframes






individual_dfs_filt = filter_data_by_flow(individual_dfs, 'Flow','P2')



def add_cycle_column(dataframes, pressure_keyword, threshold_factor=0.65, min_diff=0.01):
    """
    Adds a 'Cycle' column to each DataFrame in a dictionary, incrementing only when
    the negative difference of a column containing the specified keyword exceeds a dynamically calculated threshold.

    Parameters:
        dataframes (dict): A dictionary where keys are identifiers and values are DataFrames.
        pressure_keyword (str): A keyword to identify the pressure column to track changes.
        threshold_factor (float): Proportion of the maximum absolute .diff() to use as the threshold.
        min_diff (float): Minimum difference to consider for incrementing the cycle.

    Returns:
        dict: A new dictionary of DataFrames, each with an additional 'Cycle' column.
    """
    updated_dataframes = {}

    for key, df in dataframes.items():
        # Find the column containing the keyword
        matching_columns = [col for col in df.columns if pressure_keyword in col]

        if not matching_columns:
            raise ValueError(f"No column containing the keyword '{pressure_keyword}' found in DataFrame '{key}'.")

        if len(matching_columns) > 1:
            raise ValueError(
                f"Multiple columns containing the keyword '{pressure_keyword}' found in DataFrame '{key}': {matching_columns}. "
                "Please refine the keyword."
            )

        column_name = matching_columns[0]

        # Compute the .diff() of the column
        differences = df[column_name].diff().fillna(0)

        # Calculate the threshold dynamically as threshold_factor * max absolute .diff()
        dynamic_threshold = threshold_factor * differences.abs().max()

        # Ensure the threshold is not below the minimum difference
        threshold = max(dynamic_threshold, min_diff)

        # Initialize cycle counter
        cycle_counter = 0
        cycles = [cycle_counter]

        # Increment cycle counter only when the negative difference exceeds the threshold
        for i in range(1, len(differences)):
            if differences.iloc[i] <= -threshold:  # Check for negative differences exceeding the threshold
                cycle_counter += 1
            cycles.append(cycle_counter)

        # Add the Cycle column to the DataFrame
        df['Cycle'] = cycles

        # Store the updated DataFrame in the dictionary
        updated_dataframes[key] = df

    return updated_dataframes




individual_dfs_filt = add_cycle_column(individual_dfs_filt, 'P2')

def combine_dataframes_with_device_names(dataframes_dict):
    """
    Combines a dictionary of DataFrames into a single DataFrame, adding a 'Device Name' column 
    with the dictionary key and stripping the device-specific prefixes from column names.

    Parameters:
        dataframes_dict (dict): A dictionary where keys are device names and values are DataFrames.

    Returns:
        pd.DataFrame: A combined DataFrame with a 'Device Name' column and normalized column names.
    """
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

        # Append the updated DataFrame to the list
        combined_df.append(df)

    # Combine all DataFrames into one
    return pd.concat(combined_df, ignore_index=True)



combined_df = combine_dataframes_with_device_names(individual_dfs_filt)

def add_scc_column(df, pressure_column, temperature_column, volume=10, Z=1):
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
    def calcSCC(Pressure, Temperature, Volume=10, Z=1):
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

combined_df = add_scc_column(combined_df, 'P2 (Torr)-3', 'Temperature (C)-4')

combined_df['dscc'] = combined_df['scc'].diff()
# Calculate the time difference between consecutive rows in minutes
combined_df['dt [min]'] = combined_df['Time'].diff().dt.total_seconds() / 60

combined_df['sccm'] = combined_df['dscc'] / combined_df['dt [min]']


# filter
combined_df_filt = combined_df[abs(combined_df['dt [min]']) < 0.2]
combined_df_filt = combined_df_filt[combined_df_filt['P2 (Torr)-3'] < 1200]
combined_df_filt = combined_df_filt[combined_df_filt['sccm'] > 0].reset_index(drop = True)

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
        q1 = group[value_column].quantile(0.0)
        q3 = group[value_column].quantile(0.95)

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


combined_df_filt = filter_by_iqr_and_zscore(combined_df_filt, 'sccm', ['Device Name','Cycle'])

# Exclude the 'Time' column before grouping (if it exists)
filtered_combined_df = combined_df_filt.drop(columns=['Time'], errors='ignore')

# Group by relevant columns
combined_group = filtered_combined_df.groupby(['Device Name', 'Cycle'])

# Calculate descriptive statistics
stats_df = (
    combined_group
    .agg(['count', 'mean', 'min', 'max', 'median', 'std'])
    .reset_index()
)

# Flatten the columns of stats_df
stats_df.columns = [f"{x}_{y}" if y else f"{x}" for x, y in stats_df.columns.to_flat_index()]

# Filter for columns containing '_mean' and include grouping keys
mean_columns = ['Device Name', 'Cycle'] + [col for col in stats_df.columns if '_mean' in col]
stats_mean = stats_df[mean_columns]

# Group by 'Device Name' and 'Setpoint Readback-10'
stats_mean_group = stats_mean.groupby(['Device Name'])

# Aggregate with mean, std, min, max, and count
results_df = stats_mean_group.agg(['mean', 'std', 'min', 'max', 'count'])

# Flatten the columns of results_df
results_df.columns = [f"{x}_{y}" if y else f"{x}" for x, y in results_df.columns.to_flat_index()]

# Filter columns containing 'sccm' and reset index
results_df_sccm = results_df.filter(like='sccm').reset_index()

results_df_sccm['sccm STD [%RDG]'] = results_df_sccm['sccm_mean_std'] / results_df_sccm['sccm_mean_mean'] * 100



data = [
    {"Model": "D700", "DUT#": 1, "DUT Config": "KELLER 1Bar, 100um", "DUT SN": "09 2841 7307"},
    {"Model": "D700", "DUT#": 2, "DUT Config": "KELLER 3Bar, 100um", "DUT SN": "09 2841 7223"},
    {"Model": "D700", "DUT#": 3, "DUT Config": "KELLER 1Bar, 320um", "DUT SN": "09 2841 7310"},
    {"Model": "D700", "DUT#": 4, "DUT Config": "KELLER 3Bar, 320um", "DUT SN": "09 2841 7224"},
    {"Model": "D700", "DUT#": 5, "DUT Config": "KELLER 3Bar, 350um", "DUT SN": "09 2841 7227"},
    {"Model": "D700", "DUT#": 6, "DUT Config": "KELLER 3Bar, 350um #2", "DUT SN": "09 2841 7226"},
    {"Model": "D700", "DUT#": 7, "DUT Config": "TE 50psiLD, SiC 100um #1", "DUT SN": "09 2841 7228"},
    {"Model": "D700", "DUT#": 8, "DUT Config": "TE 50psiLD, SiC 100um #2", "DUT SN": "09 2841 7229"},
    {"Model": "Z700", "DUT#": 1, "DUT Config": "std Z700", "DUT SN": "34 1008 3267"},
    {"Model": "Z700", "DUT#": 2, "DUT Config": "std Z700", "DUT SN": "34 5863 8252"},
]

# Create a dictionary grouped by the 'Model' key
dut_dict = {}
for entry in data:
    model = entry["Model"]
    if model not in dut_dict:
        dut_dict[model] = []
    dut_dict[model].append({
        "DUT#": entry["DUT#"],
        "DUT Config": entry["DUT Config"],
        "DUT SN": entry["DUT SN"]
    })



def add_dut_info(df, dut_dict):
    """
    Adds DUT Model, DUT #, and DUT Config columns to the DataFrame based on the 'Device Name' column,
    and reorders the columns to place the new columns right after 'Device Name'.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'Device Name' column.
        dut_dict (dict): Dictionary containing DUT information with models as keys and lists of DUT info.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns and reordered columns.
    """
    # Function to extract DUT Model based on 'D7' or 'Z7'
    def extract_dut_model(device_name):
        if 'D7' in device_name:
            return 'D700'
        elif 'Z7' in device_name:
            return 'Z700'
        return None

    # Function to extract DUT # (last number from Device Name)
    def extract_dut_number(device_name):
        match = re.search(r'(\d+)$', device_name)
        return int(match.group(1)) if match else None

    # Initialize new columns
    df['DUT Model'] = df['Device Name'].apply(extract_dut_model)
    df['DUT #'] = df['Device Name'].apply(extract_dut_number)
    df['DUT Config'] = None  # Default to None
    
    # Populate the 'DUT Config' column using the dictionary
    for model, devices in dut_dict.items():
        for device in devices:
            matching_rows = (df['DUT Model'] == model) & (df['DUT #'] == device['DUT#'])
            df.loc[matching_rows, 'DUT Config'] = device['DUT Config']
    
    # Reorder columns: Move 'DUT Model', 'DUT #', and 'DUT Config' right after 'Device Name'
    cols = df.columns.tolist()  # Get current column order
    new_columns = ['Device Name', 'DUT Model', 'DUT #', 'DUT Config']
    reordered_columns = new_columns + [col for col in cols if col not in new_columns]
    df = df[reordered_columns]  # Reorder the DataFrame

    return df

results_df_sccm = add_dut_info(results_df_sccm, dut_dict)



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


if len(stats_mean) < 40:
    print('problemproblemproblemproblemproblemproblemproblemproblemproblemproblem')


save_result_to_csv(results_df_sccm, save_path, filename, suffix = "_results")
save_result_to_csv(stats_df, save_path, filename, suffix='_stats')













# def plot_key_data(individual_dfs, key="COM5-1:D727J-0x03-3", time_column="Time"):
#     """
#     Plots the data from the specified key in the individual_dfs dictionary,
#     excluding the time column.

#     Parameters:
#         individual_dfs (dict): Dictionary containing DataFrames.
#         key (str): The key of the DataFrame to plot.
#         time_column (str): The name of the time column to exclude.
#     """
#     if key not in individual_dfs:
#         print(f"Key '{key}' not found in individual_dfs.")
#         return
    
#     df = individual_dfs[key]
    
#     if df.empty:
#         print(f"DataFrame for key '{key}' is empty.")
#         return
    
#     # Exclude the time column if it exists
#     if time_column in df.columns:
#         df = df.drop(columns=[time_column])
    
#     plt.figure(figsize=(10, 5))
    
#     df.plot(ax=plt.gca())  # Plot all columns except the time column
    
#     plt.title(f"Plot for {key} (excluding {time_column})")
#     plt.xlabel("Index" if df.index.name is None else df.index.name)
#     plt.ylabel("Values")
#     plt.legend(loc='best')
#     plt.grid(True)
    
#     plt.show()



# plot_key_data(individual_dfs)
