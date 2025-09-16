# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 08:28:04 2023

@author: SFerneyhough
"""

import pandas as pd
import os, glob
import numpy as np
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Set the folder path
folder_path = r"C:\Users\sferneyhough\Desktop\Bellows puff size code"

# Get file paths with .xlsx extension in the folder
file_paths = glob.glob(folder_path + "/*.xlsx")

# Read each Excel file, assign the filename as a column, and store the data in a list
data = [pd.read_excel(file_path).assign(filename=os.path.basename(file_path)) for file_path in file_paths]

# Concatenate all data into a single DataFrame
df_combined = pd.concat(data, ignore_index=True)

# Convert 'Time' column to datetime
df_combined['datetime'] = pd.to_datetime(df_combined['Time'])

# Calculate the pressure gauge by subtracting the initial pressure from each row
df_combined['Pressure gage'] = df_combined['Pressure / psi'] - df_combined.at[0, 'Pressure / psi']

# Filter the DataFrame to keep rows where the pressure gauge is greater than 0.25
df_combined = df_combined.loc[df_combined['Pressure gage'] > 0.25]

# Reset the index of the DataFrame
df_combined.reset_index(drop=True, inplace=True)

# Assign a filename index to each row within each filename group
df_combined['filename_index'] = df_combined.groupby('filename', as_index=False).cumcount()

# Calculate the elapsed time for each row within each filename group
df_combined['elapsed_time'] = df_combined.groupby('filename')['datetime'].transform(lambda x: (x - x.min()).dt.total_seconds())

# Generate a custom color palette with more distinct colors
num_files = len(df_combined['filename'].unique())
colors = sns.hls_palette(num_files, l=.4, s=.8)

# Set the seaborn theme and plot style
sns.set_theme(context='talk', style='whitegrid')

# Plot a scatter plot for each unique filename
for i, filename in enumerate(df_combined['filename'].unique()):
    # Filter the DataFrame for the current filename
    df_filtered = df_combined[df_combined['filename'] == filename]
    
    # Scatter plot with elapsed time as x-axis and pressure gauge as y-axis
    plt.scatter(df_filtered['elapsed_time'], df_filtered['Pressure gage'], label=filename, color=colors[i])

# Set the x-axis label and y-axis label
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Pressure gauge')

# Place the legend outside the plot to the right
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust the plot layout to eliminate overlapping elements
plt.tight_layout()

# Display the plot
plt.show()