# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:37:34 2025
@author: SFerneyhough
"""

import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

# ---------------------- CONFIGURATION ----------------------
# Define the folder path and filename
FOLDER_PATH = r"\\shr-eng-srv\RnD-Project\Sensors\Pressure\Keller\60C Diaphragm Weld\Mechanical\1 Bar\Step"
FILENAME = "2025-1-28_Keller60C_1Bar_Step-Test_Post-test.csv"
FILE_PATH = os.path.join(FOLDER_PATH, FILENAME)

# ---------------------- LOAD DATA ----------------------
try:
    df = pd.read_csv(FILE_PATH, low_memory=False)
    print("CSV file loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# Convert 'Time' column to datetime
df['Time'] = pd.to_datetime(df['Time'], unit='ms', errors='coerce')

# Drop 'Relative Time (ms)' column if present
if 'Relative Time (ms)' in df.columns:
    df.drop(columns='Relative Time (ms)', inplace=True)

# Rename specific columns
df.rename(columns={
    'TCP-7:PT104-0x01-1:Temp1-1': '60C1 Ref Temp',
    'TCP-7:PT104-0x01-1:Temp2-2': '60C2 Ref Temp',
    'COM7-1:"7250i"-0x01-1:Pressure-1': 'Ruska'
}, inplace=True)

# Ensure 'Ruska' column is numeric
df['Ruska'] = pd.to_numeric(df['Ruska'], errors='coerce')*51.7149

# ---------------------- SENSOR MAPPING ----------------------
sensor_mapping = {
    'COM5': '60C2 Sensor',
    'COM4': '60C1 Sensor'
}

column_sensor_dict = {}

# Iterate over columns and rename analog values
for col in df.columns:
    if "AnalogValue" in col:
        match = re.match(r'(COM\d+)-\d+:.+-(\d+)', col)
        if match:
            com, last_num = match.groups()
            if com in sensor_mapping:
                sensor_name = f"{sensor_mapping[com]}, {last_num}"
                column_sensor_dict[col] = sensor_name

df.rename(columns=column_sensor_dict, inplace=True)

# ---------------------- FUNCTION: SPLIT DATAFRAME ----------------------
def split_dataframe_by_column(df, save_folder=None):
    """
    Splits the dataframe into separate DataFrames for each column (excluding 'Time'),
    keeping the 'Time' column in each.

    Args:
        df (pd.DataFrame): Input DataFrame.
        save_folder (str, optional): Directory to save individual DataFrames. Defaults to None.

    Returns:
        dict: A dictionary where keys are column names and values are DataFrames with 'Time' and the respective column.
    """
    if "Time" not in df.columns:
        raise ValueError("The 'Time' column is missing from the dataset.")

    dataframes = {col: df[['Time', col]].dropna() for col in df.columns if col != "Time"}

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        for col_name, df_col in dataframes.items():
            file_name = f"{col_name.replace(':', '_').replace(' ', '_')}.csv"
            df_col.to_csv(os.path.join(save_folder, file_name), index=False)
        print(f"Saved individual DataFrames in: {save_folder}")

    return dataframes

# ---------------------- FUNCTION: MERGE DATAFRAMES ----------------------
def merge_dataframes(dataframes_dict, primary_key):
    """
    Merges DataFrames using time-based 'asof' merge and removes fully duplicate rows.

    Args:
        dataframes_dict (dict): Dictionary of DataFrames.
        primary_key (str): Column to merge on (e.g., "Time").

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    if not any(primary_key in df.columns for df in dataframes_dict.values()):
        raise ValueError(f"The primary key '{primary_key}' is not found in any of the DataFrames.")

    for df in dataframes_dict.values():
        df.sort_values(by=primary_key, inplace=True)

    merged_df = None
    for col_name, df in dataframes_dict.items():
        if primary_key in df.columns:
            merged_df = df if merged_df is None else pd.merge_asof(merged_df, df, on=primary_key, direction='nearest')

    merged_df = merged_df.loc[~merged_df.drop(columns=[primary_key]).duplicated()]
    print(f"Successfully merged {len(dataframes_dict)} DataFrames on '{primary_key}'.")

    return merged_df

# ---------------------- PROCESS DATA ----------------------
individual_dfs = split_dataframe_by_column(df)
merged_df = merge_dataframes(individual_dfs, 'Time')
df_decimated = merged_df.iloc[::10].reset_index(drop=True)  # Take every 10th row

# ---------------------- FUNCTION: CREATE MATRIX A ----------------------
def createMatrixA(Pref, Voltage, Temp, V_exp, T_exp):
    """
    Creates a matrix A and solves for coefficients using least squares.

    Args:
        Pref (array-like): Reference pressures.
        Voltage (array-like): Voltage values.
        Temp (array-like): Temperature values.
        V_exp (int): Voltage exponent limit.
        T_exp (int): Temperature exponent limit.

    Returns:
        tuple: Coefficients (array) and matrix A.
    """
    n = len(Pref)
    arr = np.zeros((n, V_exp * T_exp))
    
    for x in range(n):
        for t in range(T_exp):
            for v in range(V_exp):
                arr[x, v + t * V_exp] = Voltage.iloc[x]**v * Temp.iloc[x]**t
    
    Coeffs = lstsq(arr, Pref, lapack_driver='gelsy')[0]  # Using a more stable solver
    return Coeffs, arr

# ---------------------- FUNCTION: PROCESS SENSOR DATA ----------------------
def process_sensor_data(df, sensor_cols, temp_col, ref_pressure_col, cf_exp, cf_results_dict, coeffs_dict):
    """
    Processes sensor data for a given set of columns and temperature reference.

    Args:
        df (pd.DataFrame): The input DataFrame containing sensor data.
        sensor_cols (list): List of sensor column names.
        temp_col (str): Column name for the temperature reference.
        ref_pressure_col (str): Column name for the reference pressure.
        cf_exp (int): Exponent limit for curve fitting.
        cf_results_dict (dict): Dictionary to store results (Matrix A).
        coeffs_dict (dict): Dictionary to store coefficients.

    Returns:
        None (results are stored in dictionaries).
    """
    for col in sensor_cols:
        if temp_col not in df or ref_pressure_col not in df:
            raise KeyError(f"Column '{temp_col}' or '{ref_pressure_col}' not found in dataframe.")

        voltage = df[col]
        
        # Compute Coefficients and Matrix A
        Coeffs, A = createMatrixA(df[ref_pressure_col], voltage, df[temp_col], cf_exp, cf_exp)
        
        # Store results in dictionaries
        cf_results_dict[col] = A
        coeffs_dict[col] = Coeffs  # Store Coefficients separately

# ---------------------- APPLY SENSOR PROCESSING ----------------------
cf_results_dict = {}  # Stores Matrix A
coeffs_dict = {}      # Stores Coefficients
cf_exp = 3  # Exponent limit

# Select columns for processing
C1_cols = [col for col in df_decimated.columns if '60C1' in col and 'Temp' not in col]
C2_cols = [col for col in df_decimated.columns if '60C2' in col and 'Temp' not in col]

try:
    # Process data for C1 and C2 sensors
    process_sensor_data(df_decimated, C1_cols, '60C1 Ref Temp', 'Ruska', cf_exp, cf_results_dict, coeffs_dict)
    process_sensor_data(df_decimated, C2_cols, '60C2 Ref Temp', 'Ruska', cf_exp, cf_results_dict, coeffs_dict)

    # Create DataFrame with calculated outputs
    df_temp_comp = pd.DataFrame({
        col: np.matmul(cf_results_dict[col], coeffs_dict[col]) for col in cf_results_dict.keys()
    })

       
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

df_temp_comp['Ruska'] = df_decimated['Ruska']
df_temp_comp['Time'] = df_decimated['Time']

# ---------------------- SAVE COEFFICIENTS TO CSV ----------------------
def save_coefficients_to_csv(coeffs_dict, filename="sensor_coefficients.csv"):
    """
    Saves the coefficients dictionary to a CSV file.

    Args:
        coeffs_dict (dict): Dictionary where keys are sensor names and values are coefficient arrays.
        filename (str): Path to save the CSV file.

    Returns:
        None
    """
    # Convert dictionary to DataFrame (ensure proper format)
    coeffs_df = pd.DataFrame.from_dict(coeffs_dict, orient='index')

    # Rename columns to indicate coefficient indices
    coeffs_df.columns = [f"Coeff_{i+1}" for i in range(coeffs_df.shape[1])]

    # Save to CSV
    coeffs_df.to_csv(filename, index_label="Sensor")

    print(f"Coefficients saved to {filename}")


# Call the function to save Coefficients
save_coefficients_to_csv(coeffs_dict, filename=r"\\shr-eng-srv\RnD-Project\Sensors\Pressure\Keller\60C Diaphragm Weld\Mechanical\1 Bar\Step\sensor_coefficients_post_test.csv")
