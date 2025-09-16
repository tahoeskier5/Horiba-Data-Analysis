# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:26:49 2025
@author: SFerneyhough
"""

import pandas as pd
import numpy as np
from d500sim.linalgmethods import *
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
import os
import pyarrow.parquet as pq

start_time = time.perf_counter()


# Load your data
file_path_1 = r'D:\Desktop\FRC calibration\concatenated_data.csv'
df_all_data = pd.read_csv(file_path_1, low_memory=False)
df_100 = df_all_data[df_all_data['SN'].isin(df_all_data['SN'].unique()[:100])].drop_duplicates().reset_index(drop=True)

df_100 = df_100.drop(columns={'RecipeNo',
                              'Flow[SCCM]',
                              'StdFlowPerPath[SCCM]',
                              'StdTag',
                              'Count'})


MD_coeffs = pd.read_csv(r'D:\Desktop\FRC calibration\md_coeffs_andrew_sim.csv')
MD_coeffs = MD_coeffs['Coefficients'].tolist()


# Define temperature coefficients
AFSTcoeffs_sim = [1.467809045, -0.019529758, 0.01445421, 0.018480124, -0.004772833]


class MFCAnalysis:
    def __init__(self, Tcoeffs, df, m=4, n=5):
        """Initialize the MFC Analysis class."""
        self.df = df
        self.m = m
        self.n = n
        self.Tcoeffs = Tcoeffs
        
        # Initialize MFC object
        self.MFC = d500sim(df['P1[torr]'], df['P2[torr]'], df['Temp[degC]'], df['StdFlow[SCCM]'])
        self.MFC.Tempcoeffs = Tcoeffs
        self.MDcoeffs = None
        self.CF_Coeffs = None
        self.MD_results = None

    def initialize_mfc(self, MD_Coeffs, df):
        """Perform temperature correction and MD fit."""
        
        # Correct naming issue
        self.MDcoeffs = MD_Coeffs 

        
        LBX = np.log10(np.power(df['P1[torr]'], 1.9099999666214) - np.power(df['P2[torr]'], 1.9099999666214))
        print('LBX: ', LBX)
        arr = createMatrixA(LBX, df['P2[torr]'], int(MD_Coeffs[0]), int(MD_Coeffs[1]))
        MDoutput = 10.0**np.matmul(arr, MD_Coeffs[2:])
        print('MDoutput: ', MDoutput)
        
        # Fix issue with undefined T
        T = df['Temp[degC]']  # Assigning temperature from the dataframe
        
        if T.iloc[0] != 298.15:  # Use .iloc[0] to check the first value
            TempRatio = 298.15 / (T + 273.15)
            Tfactor = self.MFC.T_fit_output(TempRatio, LBX)  # Ensure correct object is used
            MDoutput = MDoutput * Tfactor
            
        return MDoutput, self.MDcoeffs

    def clone_fit(self, sample_df, cm, cn):
        """Perform Clone fit on the given data."""
        self.MFC.Clonefit(sample_df['StdFlow[SCCM]'], sample_df['P1[torr]'], sample_df['P2[torr]'], sample_df['Temp[degC]'], cm, cn)
        self.CF_Coeffs = std_vector(self.MFC.Clonecoeffs, np.float32)

        # Store Clone fit results in a DataFrame
        CF_results = pd.DataFrame({'CF Out': self.MFC.Clonefitoutput})
        return CF_results, self.CF_Coeffs

    def cf_output(self, P1, P2, T, CF_Coeffs):
        """Calculate Clone Flow using CF coefficients."""
        PowX = 1.9099999666214
        LBX = np.log10(np.power(P1, PowX) - np.power(P2, PowX))
        arr = createMatrixA(LBX, P2, int(CF_Coeffs['Coefficients'][0]), int(CF_Coeffs['Coefficients'][1]))
        CFoutput = np.matmul(arr, CF_Coeffs['Coefficients'][2:])
        return CFoutput

    def run_analysis(self, cm_values, cn_values, num_points_list, df):
        """Run Clone Fit analysis over all combinations of parameters and append results for each SN, writing to Parquet for efficient storage."""
        
        results_parquet_path = "running_results_andrew_sim.parquet"  # Use Parquet for results
        sample_parquet_path = "sample_data_andrew_sim.parquet"  # New Parquet file for sampled data
        
        all_results = []  # Store modified DataFrames
        sample_dfs_output = []  # Store sampled DataFrames for debugging
        
        random.seed(42)  # Set base seed for reproducibility
        
        for sn in df['SN'].unique():
            df_sn = df[df['SN'] == sn].copy()
            sn_results = []  # Store results for current SN
    
            for num_points in num_points_list:
                for iteration in range(100):  
                    if len(df_sn) < num_points:
                        continue  # Skip if not enough data points
                    
                    random.seed(42 + iteration)  # Set deterministic seed per iteration
                    
                    indices = random.sample(range(len(df_sn)), num_points)  # Ensure same sample set for all SNs in this iteration
                    sample_df = df_sn.iloc[indices].reset_index(drop=True)
                    
                    if sample_df['SetP2[torr]'].nunique() < 5:
                        print(f"Skipping SN={sn}, num_points={num_points}, iteration={iteration} (Insufficient unique P2[torr])")
                        continue
                    
                    # Add Iteration metadata
                    sample_df['Iteration'] = iteration
                    sample_df['Seed Index'] = 42 + iteration  # Indicate how seed is modified per iteration
                    sample_df['Random Index'] = iteration    
                    
                    for cm in cm_values:
                        for cn in cn_values:
                            if num_points < (cm + 1) * (cn + 1):
                                print(f"Skipping Cm={cm}, Cn={cn}, Num Points={num_points} (Insufficient data)")
                                continue  # Skip invalid combinations
                            
                            try:
                                # Clone Fit Calculation
                                CF_results, CF_Coeffs = self.clone_fit(sample_df, cm, cn)
    
                                # Generate CF Output
                                Clone_Output = self.cf_output(
                                    df_sn['P1[torr]'],
                                    df_sn['P2[torr]'],
                                    df_sn['Temp[degC]'],
                                    CF_Coeffs
                                )
    
                                # Clone Flow Calculation
                                df_sn_copy = df_sn.copy()  # Prevent modifying df_sn directly
                                df_sn_copy['Clone Output'] = Clone_Output
                                df_sn_copy['Clone Flow'] = df_sn_copy['Clone Output'] * df_sn_copy['MD Flow']
    
                                # Error Calculation
                                df_sn_copy['MD vs Std Error [%RDG]'] = (
                                    (df_sn_copy['MD Flow'] - df_sn_copy['StdFlow[SCCM]']) / df_sn_copy['StdFlow[SCCM]'] * 100
                                )
                                df_sn_copy['CF vs Std Error [%RDG]'] = (
                                    (df_sn_copy['Clone Flow'] - df_sn_copy['StdFlow[SCCM]']) / df_sn_copy['StdFlow[SCCM]'] * 100
                                )
    
                                # Add metadata
                                df_sn_copy['Num Points'] = num_points
                                df_sn_copy['Cm'] = cm
                                df_sn_copy['Cn'] = cn
                                df_sn_copy['Iteration'] = iteration
                                df_sn_copy['Seed Index'] = 42
                                df_sn_copy['Random Index'] = iteration
    
                                # Store results
                                sn_results.append(df_sn_copy)
    
                                # Ensure sample_df contains Cm and Cn
                                sample_df['Cm'] = cm
                                sample_df['Cn'] = cn
                                sample_df['Num Points'] = num_points
                                sample_dfs_output.append(sample_df.copy())
    
                            except Exception as e:
                                print(f"Error for SN={sn}, Cm={cm}, Cn={cn}, Num Points={num_points}, Iteration={iteration}: {e}")
                                continue
    
            # Append SN results
            if sn_results:
                sn_results_df = pd.concat(sn_results, ignore_index=True)
                all_results.append(sn_results_df)
    
                # Save results to Parquet in chunks
                if os.path.exists(results_parquet_path):
                    existing_df = pd.read_parquet(results_parquet_path)
                    combined_df = pd.concat([existing_df, sn_results_df], ignore_index=True)
                    combined_df.to_parquet(results_parquet_path, engine='pyarrow', compression='snappy', index=False)
                else:
                    sn_results_df.to_parquet(results_parquet_path, engine='pyarrow', compression='snappy', index=False)
    
        # Final concatenation
        results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        if sample_dfs_output:
            sample_dfs = pd.concat(sample_dfs_output, ignore_index=True)
        else:
            print("Warning: No valid sampled data was generated. Creating an empty DataFrame.")
            sample_dfs = pd.DataFrame(columns=['SN', 'Iteration', 'Seed Index', 'Random Index', 'Cm', 'Cn'])  # Define expected structure
    
        # Save sampled data to Parquet
        if not sample_dfs.empty:
            if os.path.exists(sample_parquet_path):
                existing_samples_df = pd.read_parquet(sample_parquet_path)
                combined_samples_df = pd.concat([existing_samples_df, sample_dfs], ignore_index=True)
                combined_samples_df.to_parquet(sample_parquet_path, engine='pyarrow', compression='snappy', index=False)
            else:
                sample_dfs.to_parquet(sample_parquet_path, engine='pyarrow', compression='snappy', index=False)
    
        # Debugging Output: Ensure all Cm and Cn combinations exist
        print(f"Unique Cm values in sample_dfs: {sample_dfs['Cm'].unique()}")
        print(f"Unique Cn values in sample_dfs: {sample_dfs['Cn'].unique()}")
    
        return results_df, sample_dfs
        




# Initialize MFCAnalysis object
mfc_analysis = MFCAnalysis(Tcoeffs=AFSTcoeffs_sim, df=df_100)

# Perform MD fit initialization
MD_results, MD_Coeffs = mfc_analysis.initialize_mfc(MD_coeffs, df_100)

# Add MD Flow to the main dataframe
df_100['MD Flow'] = MD_results * 72 #######################################

# Define parameter ranges
cm_values = [1,2,3,4,5]
cn_values = [0,1,2]
num_points_list = [10,15,20]

# Run analysis
results, sample_dfs = mfc_analysis.run_analysis(cm_values, cn_values, num_points_list, df_100)

results_group = results.groupby(['SN','Num Points','Cm','Cn','Iteration'],as_index=False)['CF vs Std Error [%RDG]'].std()

def plot_error_by_cm_cn(data):
    """
    Create a scatter plot of CF vs Std Error [%RDG] vs Cm x Cn.
    
    Args:
        data (pd.DataFrame): The input DataFrame with columns 'Cm', 'Cn', 
                             'CF vs Std Error [%RDG]', and 'Num Points'.
    """
    # Create a new column for Cm x Cn
    data['Cm x Cn'] = (data['Cm'] + 1) * (data['Cn'] + 1)
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=data,
        x='Num Points',
        y='CF vs Std Error [%RDG]',
        hue='Cm x Cn',
        palette='viridis',
        size='Num Points',  # Optional: Size points based on Num Points
        sizes=(40, 200),    # Scale for point sizes
        legend='brief'
    )
    
    # Add labels and title
    plt.title('CF vs Std Error [%RDG] by Cm x Cn and Num Points', fontsize=14)
    plt.xlabel('Cm x Cn', fontsize=12)
    plt.ylabel('CF vs Std Error [%RDG]', fontsize=12)
    plt.legend(title='Num Points', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.ylim([0,50])
    # Show the plot
    plt.show()
    
# plot_error_by_cm_cn(results_group)

def plot_error_convergence(results_group, sn):
    """
    Plot the convergence of error values through iterations for a given SN,
    with Num Points distinguished by color and separated by Cm x Cn in FacetGrids.
    Includes an overall legend for Num Points.
    
    Args:
        results_group (pd.DataFrame): DataFrame containing columns:
                                      'SN', 'Num Points', 'Iteration', 'CF vs Std Error [%RDG]', 'Cm x Cn'.
        sn (int): The serial number (SN) to filter the results for.
    """
    # Compute Cm x Cn and filter data for the given SN
    results_group['Cm x Cn'] = (results_group['Cm'] + 1) * (results_group['Cn'] + 1)
    df_sn = results_group[results_group['SN'] == sn].copy()
    
    # Track the minimum error per iteration for each Cm x Cn and Num Points
    df_sn['Cumulative Min Error'] = df_sn.groupby(['Num Points', 'Cm x Cn'])['CF vs Std Error [%RDG]'].cummin()

    # Create a FacetGrid for Cm x Cn
    g = sns.FacetGrid(
        df_sn, 
        col='Cm x Cn', 
        col_wrap=4,  # Adjust number of columns as needed
        height=4, 
        sharey=False  # Allow independent y-axis scales
    )
    
    # Add line plots to the FacetGrid, disabling confidence intervals
    g.map_dataframe(
        sns.lineplot,
        x='Iteration',
        y='Cumulative Min Error',
        hue='Num Points',
        marker='o',
        palette='tab10',
        errorbar=None  # Disable confidence interval shading
    )

    # Add a shared legend for Num Points
    g.add_legend(title='Num Points')

    # Customize the plots
    g.set_titles(col_template='Cm x Cn = {col_name}')
    g.set_axis_labels('Iteration', 'Cumulative Min Error [%RDG]')
    g.tight_layout()

    # Add a title for the entire FacetGrid
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Error Convergence by Num Points (SN={sn})', fontsize=16)

    # Show the plot
    plt.show()


# Define the function to filter outliers based on 2 standard deviations
def filter_by_std(df, group_cols, target_col):
    """
    Filters rows in a DataFrame where the target column is within 2 standard deviations
    of the mean for each group.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    group_cols (list): List of columns to group by.
    target_col (str): Column to filter based on standard deviation.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    # Group by specified columns and calculate mean and std deviation
    grouped = df.groupby(group_cols)[target_col].agg(['mean', 'std']).reset_index()

    # Merge back with the original DataFrame
    df = df.merge(grouped, on=group_cols)

    # Print standard deviation for verification
    print(f"Standard deviation of {target_col}:\n", df['std'].describe())

    # Filter rows within 2 standard deviations
    df = df[
        (df[target_col] >= df['mean'] - 2 * df['std']) &
        (df[target_col] <= df['mean'] + 2 * df['std'])
    ]

    # Drop the extra columns used for filtering
    df = df.drop(columns=['mean', 'std'])

    return df

# remove extreme outliers
results_group = results_group[results_group['CF vs Std Error [%RDG]'] < 100]


# Apply the function to results_group
filtered_results_group = filter_by_std(results_group, ['SN', 'Num Points', 'Cm', 'Cn'], 'CF vs Std Error [%RDG]')




# # Plot convergence for a single SN
# plot_error_convergence(filtered_results_group, sn=928416706)




# # Save results to a CSV
# results.to_csv("clone_fit_results.csv", index=False)

end_time = time.perf_counter()  # End timer
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")