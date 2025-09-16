# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:06:12 2025

@author: SFerneyhough
"""

import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------- CONFIGURATION ----------------------
# Define the folder path and filename
FOLDER_PATH = r"\\shr-eng-srv\RnD-Project\Sensors\Pressure\Keller\60C Diaphragm Weld\Mechanical\1 Bar\LTD"
FILENAME = "2025-01-23_60C-Keller_LTD_FINAL.csv"
FILE_PATH = os.path.join(FOLDER_PATH, FILENAME)

FOLDER_PATH_temp_coeffs = r"\\shr-eng-srv\RnD-Project\Sensors\Pressure\Keller\60C Diaphragm Weld\Mechanical\1 Bar\Step"
FILENAME_temp_coeffs = "sensor_coefficients_post_test.csv"
FILE_PATH_temp_coeffs = os.path.join(FOLDER_PATH_temp_coeffs, FILENAME_temp_coeffs)



# ---------------------- LOAD DATA ----------------------
try:
    rawdf = pd.read_csv(FILE_PATH, low_memory=False)
    print("CSV file loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the main CSV: {e}")
    exit()

# Load temperature coefficients CSV (if applicable)
try:
    df_temp_coeffs = pd.read_csv(FILE_PATH_temp_coeffs, low_memory=False)
    print("Temperature coefficients CSV loaded successfully.")
except FileNotFoundError:
    print(f"Warning: Temperature coefficients file not found at {FILE_PATH_temp_coeffs}")
    df_temp_coeffs = None  # Handle missing file gracefully
except Exception as e:
    print(f"An error occurred while loading the temperature coefficients CSV: {e}")
    df_temp_coeffs = None

# ---------------------- PROCESS DATA ----------------------
# Forward-fill NaN values (fix inplace issue)
df = rawdf.ffill()

# Take every 10th row
df = df.iloc[::10].dropna().reset_index(drop=True)

df['Relative Time [Days]'] = df['Relative Time (ms)'] /1000/60/60/24

# Rename specific columns
df.rename(columns={
    'TCP-5:PT104-0x01-1:Temp1-1': '60C1 Ref Temp',
    'TCP-5:PT104-0x01-1:Temp2-2': '60C2 Ref Temp',
    'COM8-6:Inficon BAG302-0x01-1:Output Pressure-1': 'Reference Pressure'}, inplace=True)


df['Reference Pressure [Torr]'] = pd.to_numeric(df['Reference Pressure'], errors='coerce') * 51.7149

# Dropping Columns
dropped_columns = ['Time',
                    'Relative Time (ms)',
                    'COM5-1:Keller_Inficon_Test_PCB-0x01-1:AnalogRef-31',
                    'COM5-1:Keller_Inficon_Test_PCB-0x01-1:EnvTemp-32',
                    'COM4-0:Keller_Inficon_Test_PCB-0x01-1:AnalogRef-31',
                    'COM4-0:Keller_Inficon_Test_PCB-0x01-1:EnvTemp-32',
                    'TCP-4:"F4T1A2EAA1D2AAA"-0x01-1:Temperature Set Point-4',
                    'TCP-4:"F4T1A2EAA1D2AAA"-0x01-1:Temperature Process Value-3',
                    'TCP-5:PT104-0x01-1:Temp3-3',
                    'TCP-5:PT104-0x01-1:Temp4-4',
                    'Reference Pressure'
                    ]

df = df.drop(columns= dropped_columns)

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





def createMatrixA(Pref, Voltage, Temp, V_exp, T_exp):
    """
    Creates a matrix A

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
    
    return arr

# ---------------------- CONVERT COEFFICIENTS DATAFRAME TO DICTIONARY ----------------------
def convert_coeffs_df_to_dict(coeffs_df):
    """
    Converts a DataFrame of coefficients into a dictionary.

    Args:
        coeffs_df (pd.DataFrame): DataFrame containing coefficients.

    Returns:
        dict: Dictionary with sensor names as keys and coefficient arrays as values.
    """
    coeffs_dict = coeffs_df.set_index("Sensor").T.to_dict(orient="list")  # Convert to dictionary format
    return coeffs_dict



# Convert to dictionary format
coeffs_dict = convert_coeffs_df_to_dict(df_temp_coeffs)




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

        # Ensure the sensor name is in coeffs_dict
        if col not in coeffs_dict:
            print(f"Warning: No coefficients found for {col}. Skipping.")
            continue
        
        # Retrieve coefficients from dictionary
        Coeffs = np.array(coeffs_dict[col])  # Convert list to NumPy array
        
        # Compute Matrix A
        A = createMatrixA(df[ref_pressure_col], voltage, df[temp_col], cf_exp, cf_exp)
        
        # Store results
        cf_results_dict[col] = A
        coeffs_dict[col] = Coeffs  # Keep Coefficients in dictionary
        
        

# ---------------------- APPLY SENSOR PROCESSING ----------------------
cf_results_dict = {}  # Stores Matrix A
cf_exp = 3  # Exponent limit

# Select columns for processing
C1_cols = [col for col in df.columns if '60C1' in col and 'Reference Pressure [Torr]' not in col]
C2_cols = [col for col in df.columns if '60C2' in col and 'Reference Pressure [Torr]' not in col]

try:
    # Process data for C1 and C2 sensors
    process_sensor_data(df, C1_cols, '60C1 Ref Temp', 'Reference Pressure [Torr]', cf_exp, cf_results_dict, coeffs_dict)
    process_sensor_data(df, C2_cols, '60C2 Ref Temp', 'Reference Pressure [Torr]', cf_exp, cf_results_dict, coeffs_dict)

    # Create DataFrame with calculated outputs
    df_temp_comp = pd.DataFrame({
        col: np.matmul(cf_results_dict[col], coeffs_dict[col]) for col in cf_results_dict.keys()
    })


except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")


df_temp_comp['Relative Time [Days]'] = df['Relative Time [Days]']
df_temp_comp['Reference Pressure [Torr]'] = df['Reference Pressure [Torr]']

# zero df_temp_comp
df_temp_comp = df_temp_comp - df_temp_comp.iloc[0]



# def drift_plots(df, y_data, title, xlabel, ylabel, ylim=None, output_path=None):
#     """
#     Creates a line plot with custom colors for each group and optional y-axis limits.

#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         The DataFrame containing the data to plot.
#     y_data : str
#         The name of the column to plot on the y-axis.
#     title : str
#         The title of the plot.
#     xlabel : str
#         The label for the x-axis.
#     ylabel : str
#         The label for the y-axis.
#     ylim : tuple, optional
#         The limits for the y-axis in the form (ymin, ymax). Default is None.
#     output_path : str, optional
#         The file path to save the plot. If None, the plot is not saved. Default is None.
    
#     Returns:
#     --------
#     None
#         Displays the plot and optionally saves it to a file.
#     """
    
#     # Define custom colors for each group
#     custom_palette = {
#        '60C1 Sensor': 'Blue',        # Blue for 60C1
#        '60C2 Sensor': 'Purple',      # Purple for 60C2
#        'BCP3 Sensor': 'Green',       # Green for BCP3
#        'BCP4 Sensor': 'Orange',      # Orange for BCP4
#        'Reference Pressure [Torr]': 'Red'  # Red for Reference Pressure
#     }
    
#     # Create a full-screen figure
#     plt.figure(figsize=(16, 9))

#     sns.set_context('talk')
#     # Create the line plot with seaborn
#     sns.lineplot(data=df, x='Relative Time [Days]', y=y_data, hue='Group', style='Sensor', palette=custom_palette, legend=False)

#     # Customize the plot with the provided title, x-label, and y-label
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)

#     # Optionally set the y-axis limits
#     if ylim:
#         plt.ylim(ylim)

#     # Set the x-axis ticks to show every day
#     max_days = int(df['Relative Time [Days]'].max())
#     plt.xticks(np.arange(0, max_days + 1, step=5))

#     # Manually add the legend with custom labels and colors
#     legend_elements = [
#         Line2D([0], [0], color='blue', lw=2, label='60C1'),
#         Line2D([0], [0], color='purple', lw=2, label='60C2'),
#         # Line2D([0], [0], color='green', lw=2, label='BCP3'),
#         # Line2D([0], [0], color='orange', lw=2, label='BCP4'),
#         Line2D([0], [0], color='red', lw=2, label='Reference Pressure')
#     ]
#     plt.legend(handles=legend_elements, title='Sensor', loc='best')

#     # Adjust layout to make space for the legend
#     plt.tight_layout()

#     # Optionally save the plot to a file
#     if output_path:
#         plt.savefig(output_path, bbox_inches='tight')

#     # Display the plot
#     plt.show()

    
def drift_plots(df, y_data, title, xlabel, ylabel, ylim=None, output_path=None):
    """
    Creates a line plot using Seaborn's 'husl' color palette and includes all sensors in the legend.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    y_data : str
        The name of the column to plot on the y-axis.
    title : str
        The title of the plot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    ylim : tuple, optional
        The limits for the y-axis in the form (ymin, ymax). Default is None.
    output_path : str, optional
        The file path to save the plot. If None, the plot is not saved. Default is None.
    
    Returns:
    --------
    None
        Displays the plot and optionally saves it to a file.
    """
    
    # Create a full-screen figure
    plt.figure(figsize=(16, 9))

    sns.set_context('talk')

    # Create the line plot with seaborn using the 'husl' palette
    sns.lineplot(data=df, x='Relative Time [Days]', y=y_data, hue='Sensor', palette='deep')

    # Customize the plot with the provided title, x-label, and y-label
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Optionally set the y-axis limits
    if ylim:
        plt.ylim(ylim)

    # Set the x-axis ticks to show every 5 days
    max_days = int(df['Relative Time [Days]'].max())
    plt.xticks(np.arange(0, max_days + 1, step=5))

    # Automatically generate the legend
    plt.legend(title='Sensor', loc='best')

    # Adjust layout to make space for the legend
    plt.tight_layout()

    # Optionally save the plot to a file
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    # Display the plot
    plt.show()






# Rename only the columns that contain 'Sensor'
sensor_columns = [col for col in df_temp_comp.columns if 'Sensor' in col]
rename_mapping = {col: f"Sensor {i+1}" for i, col in enumerate(sensor_columns)}
df_temp_comp.rename(columns=rename_mapping, inplace=True)





# Temp Comp Drift
columns_to_average = df_temp_comp.columns.difference(['Relative Time [Days]'])



# Calculate the moving average for the selected columns
# df_temp_comp[columns_to_average] = df_temp_comp[columns_to_average].rolling(window=300).mean()
df_temp_comp = df_temp_comp.dropna().reset_index(drop = True)

df_temp_comp_melted = df_temp_comp.melt(id_vars=['Relative Time [Days]'],var_name='Sensor')
df_temp_comp_melted['Group'] = df_temp_comp_melted['Sensor'].apply(lambda x: x.split(',')[0])


# drift_plots(df_temp_comp_melted, 'value', 'KELLER 1bar D700 Pressure Sensor Drift at vacuum and 60C', 'Relative Time [Days]', 'Sensor Drift [Torr]', ylim = [-.6,.2])


print(df.isna().sum())
df = df.dropna()

df_temp_comp['60C1 Ref Temp'] = df['60C1 Ref Temp']
df_temp_comp[columns_to_average] = df_temp_comp[columns_to_average].rolling(window=1000).mean()


df_diff = df_temp_comp.diff()
# plt.plot(df_diff['60C1 Ref Temp'],df_diff['Sensor 1'])

# Extract x and y values
X = df_diff[['60C1 Ref Temp']].values  # Independent variable
y = df_diff['Sensor 1'].values  # Dependent variable

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict y values
y_pred = model.predict(X)

# Calculate R^2 value
r2 = r2_score(y, y_pred)

# Plot scatter and regression line
plt.scatter(df_diff['60C1 Ref Temp'], df_diff['Sensor 1'], label='Data points', alpha=0.6)
plt.plot(df_diff['60C1 Ref Temp'], y_pred, color='red', label=f'Regression line\n$R^2$={r2:.4f}')
plt.xlabel('60C1 Ref Temp')
plt.ylabel('Sensor 1')
plt.legend()
plt.title('Scatter Plot with Regression Line')

# Display plot
plt.show()

# Return R^2 value
r2