# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:48:21 2024

@author: SFerneyhough
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from itertools import cycle
import colorcet as cc


FS_flow = {5: 22, 3: 8}
SN = {5:'34 1384 2321', 3:'8675309'}
SP_error=0.0045
FS_threshold=0.10
FS_error=0.00045


folder_path = r'\\shr-eng-srv\Share\ROC4 Data v2\Projects\20231209 P000007 HF Ratio\Trial #2\N2 Calibration data\20240401 (4.1) N2 DUT35 Verify'
file_name = '2024-04-02 T113950 DUT05 N2 flowdata.csv'
# ROC ID dict
ROC_dict = {1:'ROC 4.0', 2:'ROC 4.1', 3:'Roc 4.2'}



# Read the CSV file
file_path = os.path.join(folder_path, file_name)
raw_df = pd.read_csv(file_path)

# Split the filename using whitespace as delimiter
parts = file_name.split()

# Extract the first part which should contain the date
date_str = parts[0]


# Select relevant columns
columns_to_keep = {'dut_setpoint', 'Run #_', 'DUT#_', 'Pch Target [torr]', 'Flow_Pch [sccm]_mean', 'met_chamber_number','system_id_mean'}
df = raw_df[list(columns_to_keep)]

# Add FS_flow and Error [%FS] columns
df['FS_flow'] = df['DUT#_'].map(FS_flow)
df['Error [%FS]'] = (df['Flow_Pch [sccm]_mean'] / (df['dut_setpoint'] / 100 * df['FS_flow']) - 1) * 100


# # Error limits
# df = df.loc[df['Error [%FS]'] >= -1.0]
# df = df.loc[df['Error [%FS]'] <= 1.0]
# df = df.reset_index()



# Filter out rows not in FS_flow keys
df = df.loc[df['DUT#_'].isin(FS_flow.keys())]

# Group by setpoint, Pch Target, and DUT#
df_mean = df.groupby(['dut_setpoint', 'Pch Target [torr]', 'DUT#_', 'met_chamber_number','system_id_mean'])['Error [%FS]'].mean().reset_index()

# Repeatability calcs- only if there are more than 2 runs
# Check the count of unique values in 'Run #_'
df['run_count'] = df['Run #_'].nunique()

# Filter the DataFrame for rows where the count is greater than 2
df_filtered = df[df['run_count'] > 2].copy()

# Perform calculations only if there are more than 2 numbers in 'Run #_'
if not df_filtered.empty:
    df_mean2 = df_filtered.groupby(['dut_setpoint', 'Pch Target [torr]', 'Run #_', 'DUT#_', 'met_chamber_number'])['Error [%FS]'].mean().reset_index()
    
    # Define custom function to calculate (max - min) / 2
    def max_min_div_2(series):
        return (series.max() - series.min()) / 2
    
    df_midpoint = df_mean2.groupby(['dut_setpoint','Pch Target [torr]','DUT#_','met_chamber_number'])['Error [%FS]'].agg(max_min_div_2).reset_index()
    df_midpoint.rename(columns={'Error [%FS]': 'Repeatability [%FS]'}, inplace=True)
    
    # add repeat to df_mean
    df_mean['Repeatability [%FS]'] = df_midpoint['Repeatability [%FS]']


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
    # Plot Error [%FS]
    plt.figure(figsize=(20, 10))
    
    # Plot spec lines
    sns.lineplot(data=df_spec, x='Setpoint', y='Spec Line +', color='r', linestyle='--', linewidth=2)
    sns.lineplot(data=df_spec, x='Setpoint', y='Spec Line -', color='r', linestyle='--', linewidth=2)
    
    # Define marker styles and palette
    marker_styles = ['o', 's', 'd', '^', 'v', '<', '>', 'x', '+', '*']
    marker_cycler = cycle(marker_styles)
    num_colors = len(df_dut['Pch Target [torr]'].unique())
    color_palette = sns.color_palette('tab20', num_colors)

    # Plot Error [%FS]
    for i, (group_name, group_data) in enumerate(df_dut.groupby('Pch Target [torr]')):
        sns.lineplot(data=group_data, x='dut_setpoint', y='Error [%FS]', palette=[color_palette[i]], hue='Pch Target [torr]', linewidth=2,
                    marker=next(marker_cycler), style='Pch Target [torr]')
    
    # Adding FS_flow value to the title
    if dut in FS_flow:
        fs_flow_value = FS_flow[dut]
        plt.title(f'[C] D500 MFC Accuracy on {ROC_dict[dfs[dut]["system_id_mean"].iloc[0]]}\nDUT#{dut}, {fs_flow_value} sccm \nS/N {SN.get(dut, "Serial Number Not Available")}\n{date_str}')
    else:
        plt.title(f'DUT#{dut}')
    
    plt.xlabel('% Full Scale')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.ylabel('Error [%FS]')
    plt.legend(title='P2 Target [Torr]', loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.tight_layout()
    plt.ylim(-2, 2)
    plt.grid()
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    plt.show()
    
    # Save the plot
    plot_file_name_error = f'plot_error_dut_{dut}.png'
    plot_file_path_error = os.path.join(folder_path, plot_file_name_error)
    # plt.savefig(plot_file_path_error)
    # plt.close()  # Close the current plot
    
    # Check if df_filtered is not empty
    if not df_filtered.empty:
        # Plot Repeatability [%FS]
        plt.figure(figsize=(20, 10))
        
        # Plot Repeatability [%FS]
        for i, (group_name, group_data) in enumerate(df_dut.groupby('Pch Target [torr]')):
            sns.lineplot(data=group_data, x='dut_setpoint', y='Repeatability [%FS]', palette=[color_palette[i]], hue='Pch Target [torr]', linewidth=2,
                        marker=next(marker_cycler), style='Pch Target [torr]')
        
        # Adding FS_flow value to the title
        if dut in FS_flow:
            fs_flow_value = FS_flow[dut]
            plt.title(f'[C] D500 MFC Repeatability on {ROC_dict[dfs[dut]["system_id_mean"].iloc[0]]}\nDUT#{dut}, {fs_flow_value} sccm \nS/N {SN.get(dut, "Serial Number Not Available")}\n{date_str}')
        else:
            plt.title(f'DUT#{dut}')
        
        plt.xlabel('% Full Scale')
        plt.xscale('log')  # Set x-axis to logarithmic scale
        plt.ylabel('Repeatability [%FS]')
        plt.legend(title='P2 Target [Torr]', loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        plt.tight_layout()
        plt.grid()
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
        plt.show()
        
        # Save the plot
        plot_file_name_repeatability = f'plot_repeatability_dut_{dut}.png'
        plot_file_path_repeatability = os.path.join(folder_path, plot_file_name_repeatability)
        # plt.savefig(plot_file_path_repeatability)