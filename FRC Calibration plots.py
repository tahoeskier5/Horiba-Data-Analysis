# -*- coding: utf-8 -*-
"""
Calibration Data Analysis Script
Created on Fri Jan 31 09:11:14 2025
@author: SFerneyhough
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio
import os

pio.renderers.default = "browser"  # Open Plotly plots in browser

# =============================================================================
# File Paths
# =============================================================================
file_path = r"D:\Desktop\FRC calibration\running_results_test_andrew_sim_numpaths.parquet"
file_path2 = r"D:\Desktop\FRC calibration\sample_data_test_andrew_sim_numpaths.parquet"
file_path3 = r"D:\Desktop\FRC calibration\concatenated_data.csv"
file_path4 = r"D:\Desktop\FRC calibration\benchmark_validation_results.csv"

# =============================================================================
# Load Data
# =============================================================================
df = pd.read_parquet(file_path)
df_sample = pd.read_parquet(file_path2)
df_prod_results = pd.read_csv(file_path3)
df_benchmark = pd.read_csv(file_path4)

# =============================================================================
# Preprocessing
# =============================================================================
df['Abs Clone Error'] = df['CF vs Std Error [%RDG]'].abs()

df_prod_results_verifi = df_prod_results[df_prod_results['Type'] == 'Verifi'].copy()
df_prod_results_verifi['Production Flow [sccm]'] = df_prod_results_verifi['Flow[SCCM]']
df_prod_results_verifi['Production MD Accuracy [%RDG]'] = (
    (df_prod_results_verifi['Production Flow [sccm]'] - df_prod_results_verifi['StdFlow[SCCM]']) 
    / df_prod_results_verifi['StdFlow[SCCM]'] * 100
)

# Rename df_benchmark columns for consistency
df_benchmark = df_benchmark.rename(columns={
    'SetPoint [%]': 'SetPoint[%]',
    'SetP2 [Torr]': 'SetP2[torr]'
})

df_results_sn = df.groupby(['SN', 'Num Points', 'Cm', 'Cn', 'Iteration'], as_index=False)['Abs Clone Error'].mean()
df_results = df.groupby(['Num Points', 'Cm', 'Cn', 'Iteration'], as_index=False)['Abs Clone Error'].mean()

# =============================================================================
# Filter Data Subsets
# =============================================================================
def filter_and_group(df, num_points, cm, cn, iteration):
    df_filtered = df[(df['Num Points'] == num_points) & 
                     (df['Cm'] == cm) & 
                     (df['Cn'] == cn) & 
                     (df['Iteration'] == iteration) & 
                     (df['Type'] == 'Verifi')]
    df_mean = df_filtered.groupby(['SetPoint[%]', 'SetP2[torr]', 'Type'], as_index=False)['Abs Clone Error'].mean()
    return df_filtered, df_mean

df_10_4_0_33, df_10_mean = filter_and_group(df, 10, 4, 0, 33)
df_15_5_1_15, df_15_mean = filter_and_group(df, 15, 5, 1, 15)
df_20_5_2_72, df_20_mean = filter_and_group(df, 20, 5, 2, 72)

df_results_10_4_0 = df_results_sn[(df_results_sn['Cm'] == 4) & (df_results_sn['Cn'] == 0)]
df_sample_10_4_0_33 = df_sample[(df_sample['Cm'] == 4) & (df_sample['Cn'] == 0) & (df_sample['Iteration'] == 33)].merge(
    df_results_10_4_0[['SN', 'Iteration', 'Num Points', 'Abs Clone Error']], 
    on=['SN', 'Iteration', 'Num Points'], how='left'
)

# =============================================================================
# Merge Production and Benchmark Results
# =============================================================================
def merge_prod_and_benchmark_results(df_filtered):
    df_merged = df_filtered.merge(
        df_prod_results_verifi[['SetPoint[%]', 'SetP2[torr]', 'SN', 'Production MD Accuracy [%RDG]']],
        on=['SetPoint[%]', 'SetP2[torr]', 'SN'],
        how='left'
    )

    df_merged = df_merged.merge(
        df_benchmark[['SetPoint[%]', 'SetP2[torr]', 'SN', 'MD vs. Production Accuracy [%RDG]']],
        on=['SetPoint[%]', 'SetP2[torr]', 'SN'],
        how='left'
    )

    return df_merged

df_10_4_0_33 = merge_prod_and_benchmark_results(df_10_4_0_33)
df_15_5_1_15 = merge_prod_and_benchmark_results(df_15_5_1_15)
df_20_5_2_72 = merge_prod_and_benchmark_results(df_20_5_2_72)

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_md_vs_clone_accuracy_boxplot(df, base_title, save_dir):
    """
    Creates and saves separate boxplots for each SetPoint [%]:
    - x-axis: SetP2[torr]
    - y-axis: Error [%RDG]
    - 3 error types: Clone Fit, Production, MD Benchmark
    """

    os.makedirs(save_dir, exist_ok=True)

    sns.set_context("talk")

    for setpoint in sorted(df['SetPoint[%]'].unique()):
        df_subset = df[df['SetPoint[%]'] == setpoint].copy()

        df_long = df_subset.melt(
            id_vars=['SetP2[torr]'],
            value_vars=[
                'CF vs Std Error [%RDG]',
                'Production MD Accuracy [%RDG]',
                'MD vs. Production Accuracy [%RDG]'
            ],
            var_name='Error Type',
            value_name='Error [%RDG]'
        )

        error_labels = {
            'CF vs Std Error [%RDG]': 'Simulated Clone Fit Accuracy',
            'Production MD Accuracy [%RDG]': 'Production MD Accuracy',
            'MD vs. Production Accuracy [%RDG]': 'Simulated MD Accuracy'
        }
        df_long['Error Type'] = df_long['Error Type'].map(error_labels)

        # Large Figure Size
        plt.figure(figsize=(20, 12))

        sns.boxplot(
            data=df_long,
            x='SetP2[torr]',
            y='Error [%RDG]',
            hue='Error Type',
            palette='deep'
        )

        plt.title(f'{base_title}\nSet Point = {setpoint}%', fontsize=20)
        plt.xlabel('Set Point [Torr]', fontsize=16)
        plt.ylabel('Accuracy [%RDG]', fontsize=16)
        # plt.ylim([-2, 2])
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend(title='', fontsize=14)
        plt.tight_layout()

        filename = f"{base_title.replace(' ', '_')}_SetPoint_{setpoint}pct.png"
        filepath = os.path.join(save_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filepath}")





def plot_min_clone_error(df):
    if df.empty:
        print("Warning: DataFrame is empty. No data to plot.")
        return

    df_min_error = df.loc[df.groupby(['SN', 'Num Points'])['Abs Clone Error'].idxmin()]
    fig = px.scatter(
        df_min_error,
        x="Num Points",
        y="Abs Clone Error",
        color="Iteration",
        hover_data={"SN": True, "Iteration": True, "Num Points": True, "Abs Clone Error": True},
        title="Minimum Clone Fit Error per SN and Num Points",
        labels={"Num Points": "Number of Calibration Points", "Abs Clone Error": "Minimum Absolute Clone Fit Error"}
    )
    fig.show()


def rank_iterations_by_error(df):
    df_avg_error = df.groupby(['Iteration', 'Num Points'], as_index=False)['Abs Clone Error'].mean()
    df_avg_error['Rank'] = df_avg_error['Abs Clone Error'].rank(method="dense", ascending=True)
    return df_avg_error.sort_values('Rank')


# =============================================================================
# Example Usage
# =============================================================================
# plot_md_vs_clone_accuracy_boxplot(
#     df_20_5_2_72,
#     'Simulated vs Production Accuracy vs Benchmark MD Accuracy - 20 Calibration Points',
#     save_dir=r'D:\Desktop\FRC calibration\plots'
# )

# plot_md_vs_clone_accuracy_boxplot(
#     df_15_5_1_15,
#     'Simulated vs Production Accuracy vs Benchmark MD Accuracy - 15 Calibration Points',
#     save_dir=r'D:\Desktop\FRC calibration\plots'
# )

# plot_md_vs_clone_accuracy_boxplot(
#     df_10_4_0_33,
#     'Simulated vs Production Accuracy vs Benchmark MD Accuracy - 10 Calibration Points',
#     save_dir=r'D:\Desktop\FRC calibration\plots'
# )

df_ranked_iterations = rank_iterations_by_error(df_results_10_4_0)
