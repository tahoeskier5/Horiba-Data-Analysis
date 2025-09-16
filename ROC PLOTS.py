# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:13:40 2024

@author: SFerneyhough
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np




def plot_error_lines(folder_path, file_name, FS_flow, SN, Gas, Temp=25, SP_error=0.0045, FS_threshold=0.10, FS_error=0.00045):
    """
    Generate a plot of error lines based on the data provided.

    Parameters:
        folder_path (str): The path to the folder containing the CSV file.
        file_name (str): The name of the CSV file.
        FS_flow (dict): A dictionary mapping DUT numbers to their corresponding FS flow values.
        SN (str): Device serial number.
        Gas (str): Gas type.
        Temp (int, optional): Temperature. Default is 25.
        SP_error (float, optional): The SP error value. Default is 0.0045.
        FS_threshold (float, optional): The FS threshold value. Default is 0.10.
        FS_error (float, optional): The FS error value. Default is 0.00045.
        
    """
    
    
    # Read the CSV file
    file_path = os.path.join(folder_path, file_name)
    raw_df = pd.read_csv(file_path)

    # Split the filename using whitespace as delimiter
    parts = file_name.split()

    # Extract the first part which should contain the date
    date_str = parts[0]


    # Select relevant columns
    columns_to_keep = {'dut_setpoint', 'DUT#_', 'Pch Target [torr]', 'Flow_Pch [sccm]_mean'}
    df = raw_df[list(columns_to_keep)]

    # Add FS_flow and Error [%FS] columns
    df['FS_flow'] = df['DUT#_'].map(FS_flow)
    df['Error [%FS]'] = (df['Flow_Pch [sccm]_mean'] / (df['dut_setpoint'] / 100 * df['FS_flow']) - 1) * 100

    # Filter out rows not in FS_flow keys
    df = df.loc[df['DUT#_'].isin(FS_flow.keys())]

    # Group by setpoint, Pch Target, and DUT#
    df_mean = df.groupby(['dut_setpoint', 'Pch Target [torr]', 'DUT#_'])['Error [%FS]'].mean().reset_index()

    # Generate setpoints with equal spacing between 0.1 and 100
    setpoints = np.logspace(np.log10(0.1), np.log10(100), num=1000)
    df_spec = pd.DataFrame({'Setpoint': setpoints})

    # Define spec line function
    def spec_line(row):
        if row['Setpoint'] / 100 >= FS_threshold:
            return SP_error * 100
        else:
            return FS_error / row['Setpoint'] * 100 * 100

    # Calculate spec lines
    df_spec['Spec Line +'] = df_spec.apply(spec_line, axis=1)
    df_spec['Spec Line -'] = df_spec.apply(spec_line, axis=1) * -1

    # Separating into different dataframes based on 'DUT#_'
    dfs = {}
    for dut in df_mean['DUT#_'].unique():
        dfs[dut] = df_mean[df_mean['DUT#_'] == dut]

    # Plotting
    sns.set_context("talk")
    for dut, df_dut in dfs.items():
        plt.figure(figsize=(20, 10))  # Full-screen size

        # Plot spec lines
        sns.lineplot(data=df_spec, x='Setpoint', y='Spec Line +', color='r', linestyle='--')
        sns.lineplot(data=df_spec, x='Setpoint', y='Spec Line -', color='r', linestyle='--')

        # Plot error lines
        sns.lineplot(data=df_dut, x='dut_setpoint', y='Error [%FS]', palette='tab20', hue='Pch Target [torr]')

        # Adding FS_flow value to the title
        if dut in FS_flow:
            fs_flow_value = FS_flow[dut]
            plt.title(f'[{Temp}C] D500 MFC Accuracy on ROC4.0\nDUT#{dut}, {fs_flow_value} sccm { Gas}\nS/N {SN}\n{date_str}')
        else:
            plt.title(f'DUT#{dut}')

        plt.xlabel('% Full Scale')
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.ylabel('Error [%FS]')
        plt.legend(title='P2 Target [Torr]', loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        plt.tight_layout()
        plt.ylim(-2, 2)
        plt.grid()
        
        # Format x-axis tick labels to display as whole numbers
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))

    plt.show()

# Define parameters
folder_path = r"C:\Users\sferneyhough\Desktop\ROC Data Plotting"
file_name = "2024-01-12 T104818 DUT135 Nitrogen flowdata.csv"
FS_flow = {1: 218, 3: 221, 5: 219}

# Call the function
plot_error_lines(folder_path, file_name, FS_flow, Gas='N2', SN='34 3604 5351')


