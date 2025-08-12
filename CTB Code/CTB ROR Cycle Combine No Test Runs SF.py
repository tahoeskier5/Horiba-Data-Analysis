# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:35:25 2025

@author: SFerneyhough
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def load_processed_dataframes(folder_path):
    """
    Walk `folder_path` for any files with 'Processed' in the filename,
    read each into a DataFrame, and return a dict {file_stem: df}.
    Supports .csv, .xls, .xlsx.
    """
    processed_dfs = {}
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if 'Processed' in fname:
                file_path = os.path.join(root, fname)
                stem, ext = os.path.splitext(fname)
                ext = ext.lower()
                try:
                    if ext == '.csv':
                        df = pd.read_csv(file_path, low_memory=False)
                    elif ext in ('.xls', '.xlsx'):
                        df = pd.read_excel(file_path, engine='openpyxl')
                    else:
                        # skip unsupported file types
                        continue
                except Exception as e:
                    print(f"⚠️  Skipping {fname}: {e}")
                    continue

                processed_dfs[stem] = df
    return processed_dfs

# Usage
folder = r'D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\1st soak\Results'
dfs_dict = load_processed_dataframes(folder)

for name, df in dfs_dict.items():
    # 1) Strip any leading “DUT<digits> ” from all column names
    df.columns = df.columns.str.replace(r'^DUT\d+\s*', '', regex=True)

    # 2) If “DUT16” was in the filename, rename “Pressure Reading [REAL]” → “P0 Pressure Reading [REAL]”
    if "DUT16" in name:
        df.rename(
            columns={"Pressure Reading [REAL]": "P0 Pressure Reading [REAL]"},
            inplace=True
        )

    # 3) Extract the digits after “TESTRUN” as an integer (or pd.NA if no match)
    match = re.search(r'TESTRUN(\d+)', name, re.IGNORECASE)
    if match:
        test_run_num = int(match.group(1))
    else:
        test_run_num = pd.NA

    # 4) Assign it to every row in a new column, then cast to nullable Int64
    df['Test Run'] = test_run_num
    df['Test Run'] = df['Test Run'].astype("Int64")

concat_df = pd.concat(dfs_dict.values(), ignore_index=True)


concat_df = concat_df[concat_df['ror_flow_iterative'].notna()]
concat_df = concat_df[concat_df['r_squared'] > 0.999]

ts = pd.to_datetime(concat_df['Time'].iloc[0])
mean_time = ts.strftime('%Y-%m-%d %H:%M')

# keep only int/float columns
concat_df = concat_df.select_dtypes(include=['number'])



def compute_ror_stats(concat_df, mean_time):
    """
    Compute ROR statistics and return two DataFrames:
      1) stats_df: intermediate cycle‐level stats
      2) results_df_ror: DUT‐ and Setpoint‐level stats with repeatability and uncertainty

    Parameters
    ----------
    concat_df : pandas.DataFrame
        Must include columns ['Cycle', 'DUT', 'ROR Setpoint',
        'ror_flow_iterative', 'ror_flow_global', ...].
    mean_time : str or pandas.Timestamp
        A label (e.g. date) to add into the intermediate stats DataFrame.

    Returns
    -------
    stats_df : pandas.DataFrame
        Flattened cycle‐level statistics, including count/mean/min/max/median/std
        for both ror_flow_iterative and ror_flow_global.
    results_df_ror : pandas.DataFrame
        Contains:
          - 'DUT' and 'ROR Setpoint'
          - All ror_flow_* aggregated stats (mean, std, min, max, count)
          - 'ROR Iterative Repeatability [%RDG]' (2σ)
          - 'ROR Global Repeatability [%RDG]'    (2σ)
          - 'ROR Iterative Uncertainty [%RDG]'   (max_repeatability_per_DUT / √count)
          - 'ROR Global Uncertainty [%RDG]'      (max_repeatability_per_DUT / √count)
    """
    # 1) (Optional) describe ror_flow_iterative by Cycle, DUT, ROR Setpoint
    _ = concat_df.groupby(
        ['Cycle', 'DUT', 'ROR Setpoint'], as_index=False
    )['ror_flow_iterative'].describe()

    # 2) Perform custom aggregation on ['count','mean','min','max','median','std']
    stats_df = (
        concat_df
        .groupby(['Cycle', 'DUT', 'ROR Setpoint'], as_index=False)
        .agg(['count', 'mean', 'min', 'max', 'median', 'std'])
    )

    # 3) Flatten MultiIndex columns of stats_df
    stats_df.columns = [
        f"{col_name}_{agg}" if agg else f"{col_name}"
        for col_name, agg in stats_df.columns.to_flat_index()
    ]
    #    Now stats_df has columns like:
    #    ['Cycle', 'DUT', 'ROR Setpoint',
    #     'ror_flow_iterative_count', 'ror_flow_iterative_mean', …,
    #     'ror_flow_global_std', …, etc.]

    # 4) Add the provided mean_time as a new column
    stats_df['Date'] = mean_time

    # 5) Keep only the grouping keys + any column that ends with '_mean'
    keep_mean_cols = ['DUT', 'ROR Setpoint', 'Cycle'] + [
        col for col in stats_df.columns if col.endswith('_mean')
    ]
    stats_mean = stats_df[keep_mean_cols]

    # 6) Group stats_mean by DUT and ROR Setpoint (as_index=False to keep them as columns)
    results_df = (
        stats_mean
        .groupby(['DUT', 'ROR Setpoint'], as_index=False)
        .agg(['mean', 'std', 'min', 'max', 'count'])
    )

    # 7) Flatten MultiIndex columns in results_df
    results_df.columns = [
        f"{col_name}_{agg}" if agg else f"{col_name}"
        for col_name, agg in results_df.columns.to_flat_index()
    ]
    #    After flattening, results_df columns look like:
    #    ['DUT', 'ROR Setpoint',
    #     'ror_flow_iterative_mean_mean', 'ror_flow_iterative_mean_std',
    #     'ror_flow_iterative_mean_min', …,
    #     'ror_flow_global_mean_count', …, etc.]

    # 8) Identify all columns that contain 'ror_flow'
    ror_cols = [c for c in results_df.columns if 'ror_flow' in c]

    # 9) Build results_df_ror with DUT, ROR Setpoint, plus all ror_flow_* columns
    results_df_ror = results_df[['DUT', 'ROR Setpoint'] + ror_cols].copy()

    # 10) Compute 2σ percent‐RDG for iterative ROR
    results_df_ror.loc[:, 'ROR Iterative Repeatability [%RDG]'] = (
        results_df_ror['ror_flow_iterative_mean_std']
        / results_df_ror['ror_flow_iterative_mean_mean']
    ) * 100 * 2

    # 11) Compute 2σ percent‐RDG for global ROR
    results_df_ror.loc[:, 'ROR Global Repeatability [%RDG]'] = (
        results_df_ror['ror_flow_global_mean_std']
        / results_df_ror['ror_flow_global_mean_mean']
    ) * 100 * 2

    # 12) Determine the maximum repeatability per DUT
    max_iter_repeat_per_dut = results_df_ror.groupby('DUT')['ROR Iterative Repeatability [%RDG]'] \
                                            .transform('max')
    max_glob_repeat_per_dut = results_df_ror.groupby('DUT')['ROR Global Repeatability [%RDG]'] \
                                            .transform('max')

    # 13) Compute uncertainty using the max repeatability for that DUT
    #     divided by sqrt(count)
    results_df_ror.loc[:, 'ROR Iterative Uncertainty [%RDG]'] = (
        max_iter_repeat_per_dut
        / results_df_ror['ror_flow_iterative_mean_count'].pow(0.5)
    )

    results_df_ror.loc[:, 'ROR Global Uncertainty [%RDG]'] = (
        max_glob_repeat_per_dut
        / results_df_ror['ror_flow_global_mean_count'].pow(0.5)
    )

    return stats_df, results_df_ror


stats_df, results_df_ror = compute_ror_stats(concat_df, mean_time)


def plot_ror_iterative(results_df_ror):
    """
    Plot ROR Iterative Repeatability[%RDG] vs. ROR Setpoint, colored by DUT,
    with distinct line styles and markers for each DUT.
    """
    required_cols = {'DUT', 'ROR Setpoint', 'ROR Global Repeatability [%RDG]'}
    if not required_cols.issubset(results_df_ror.columns):
        raise KeyError(f"DataFrame must contain columns: {required_cols}")

    unique_duts = sorted(results_df_ror['DUT'].unique())
    cmap       = plt.cm.get_cmap('tab20', len(unique_duts))
    linestyles = ['-', '--', '-.', ':']
    markers    = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, dut in enumerate(unique_duts):
        sub = results_df_ror[results_df_ror['DUT'] == dut].sort_values('ROR Setpoint')
        x, y = sub['ROR Setpoint'], sub['ROR Global Repeatability [%RDG]']
        color = cmap(idx)
        ax.plot(x, y, color=color, linestyle=linestyles[idx % len(linestyles)], linewidth=2, alpha=0.8)
        ax.scatter(x, y, label=f"DUT {int(dut)}", color=color, marker=markers[idx % len(markers)], edgecolors='w', s=60)

    ax.set_xlabel("ROR Setpoint", fontsize=14)
    ax.set_ylabel("ROR Global 2-σ Repeatability [%RDG]", fontsize=14)
    ax.set_title("First HF Soak ROR Repeatability", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Legend outside
    legend = ax.legend(title="DUT", title_fontsize=12, fontsize=12,
                       bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    


def plot_ror_uncertainty(results_df_ror):
    """
    Plot the maximum ROR Iterative Uncertainty [%RDG] per DUT,
    with DUT on the x-axis. No legend is needed since there's one point per DUT.

    Parameters
    ----------
    results_df_ror : pandas.DataFrame
        Must contain columns:
          - 'DUT'
          - 'ROR Iterative Uncertainty [%RDG]'
    """
    required_cols = {'DUT', 'ROR Iterative Uncertainty [%RDG]'}
    if not required_cols.issubset(results_df_ror.columns):
        raise KeyError(f"DataFrame must contain columns: {required_cols}")

    # Compute the maximum iterative uncertainty per DUT
    df_max = (
        results_df_ror
        .groupby('DUT', as_index=False)['ROR Iterative Uncertainty [%RDG]']
        .max()
    )

    # Extract x and y
    x = df_max['DUT']
    y = df_max['ROR Iterative Uncertainty [%RDG]']

    plt.figure(figsize=(8, 5))
    plt.scatter(
        x, y,
        color='tab:blue',
        edgecolors='w',
        s=80
    )

    plt.xlabel("DUT", fontsize=14)
    plt.ylabel("Max ROR Iterative Uncertainty [%RDG]", fontsize=14)
    plt.title("Max ROR Iterative Uncertainty per DUT", fontsize=16)

    # Use the DUT values as tick labels
    plt.xticks(x, fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()
    
    
plot_ror_iterative(results_df_ror)
# plot_ror_uncertainty(results_df_ror)



def extract_flow_means(stats_df):
    """
    Given a DataFrame `stats_df` (e.g., the output of a .groupby(...).describe() 
    or similar), return a new DataFrame containing only the columns:
      - 'Cycle'
      - 'DUT'
      - 'ROR Setpoint'
      - 'Test Run'
      - 'ror_flow_global_mean'
      - 'ror_flow_iterative_mean'

    It is assumed that `stats_df` already has those columns (for example,
    if you previously did something like `df_stats = df_trimmed.groupby(
    ["Cycle", "DUT", "ROR Setpoint", "Test Run"]).describe()`).

    Parameters
    ----------
    stats_df : pandas.DataFrame
        A DataFrame whose columns include at least:
        ['Cycle', 'DUT', 'ROR Setpoint', 'Test Run',
         'ror_flow_global_mean', 'ror_flow_iterative_mean', ...].

    Returns
    -------
    pandas.DataFrame
        A copy of `stats_df` containing only the six specified columns.
    """
    # List of columns we want to keep
    keep_cols = [
        'Cycle',
        'DUT',
        'ROR Setpoint',
        'ror_flow_global_mean',
        'ror_flow_iterative_mean'
    ]

    # Check that all required columns exist
    missing = [c for c in keep_cols if c not in stats_df.columns]
    if missing:
        raise KeyError(f"extract_flow_means: missing columns in stats_df: {missing}")

    # Create and return the new DataFrame
    summary_df = stats_df[keep_cols].copy()
    return summary_df


summary = extract_flow_means(stats_df)

def compute_two_sigma_with_uncertainty(summary_df, max_n=25, random_state=None):
    """
    For each combination of DUT and ROR Setpoint, compute repeatability (unc_g, unc_i)
    for n = 1..max_n by sampling n points, then derive uncertainties using the maximum
    repeatability across all n values.

    Steps for each (DUT, ROR Setpoint):
      1. For n = 1..max_n:
         • unc_g[n] = (2 * std(sample_global_n) / mean(sample_global_n)) * 100
         • unc_i[n] = (2 * std(sample_iterative_n) / mean(sample_iterative_n)) * 100
         (For n=1, unc_g[1]=unc_i[1]=0.0 by definition.)
      2. Let max_unc_g = max(unc_g[1..max_n]) and max_unc_i = max(unc_i[1..max_n])
      3. For each n, 
         • Global Uncertainty [%RDG]    = max_unc_g / sqrt(n)
         • Iterative Uncertainty [%RDG] = max_unc_i / sqrt(n)

    If a group has fewer than n rows, unc_g[n] and unc_i[n] are NaN, and both
    uncertainties are NaN for that n.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Must contain at least these columns:
          - 'DUT'
          - 'ROR Setpoint'
          - 'ror_flow_global_mean'
          - 'ror_flow_iterative_mean'
    max_n : int, default=25
        Compute repeatability/uncertainties for n = 1..max_n.
    random_state : int or None, default=None
        Base seed for reproducible sampling. Each group uses (random_state + group_index).

    Returns
    -------
    pandas.DataFrame
        Columns:
          - 'DUT'
          - 'ROR Setpoint'
          - 'n'
          - 'Global Repeatability [%RDG]'
          - 'Iterative Repeatability [%RDG]'
          - 'Global Uncertainty [%RDG]'
          - 'Iterative Uncertainty [%RDG]'
    """
    required = ['DUT', 'ROR Setpoint', 'ror_flow_global_mean', 'ror_flow_iterative_mean']
    missing = [col for col in required if col not in summary_df.columns]
    if missing:
        raise KeyError(f"Missing columns in summary_df: {missing}")

    all_records = []
    # Enumerate groups for a stable seed offset
    for idx, ((dut, setpoint), grp) in enumerate(
        summary_df.groupby(['DUT', 'ROR Setpoint'])
    ):
        grp_global = grp['ror_flow_global_mean']
        grp_iter   = grp['ror_flow_iterative_mean']
        size = len(grp)
        group_seed = None if random_state is None else random_state + idx

        # First pass: compute unc_g and unc_i for n = 1..max_n and store
        temp = []
        for n in range(1, max_n + 1):
            if size < n:
                # Not enough samples
                unc_g = np.nan
                unc_i = np.nan
            elif n == 1:
                # By definition for a single point
                unc_g = 0.0
                unc_i = 0.0
            else:
                # Sample n points (with reproducible seed if provided)
                if group_seed is not None:
                    samp_g = grp_global.sample(n, random_state=group_seed)
                    samp_i = grp_iter.sample(n, random_state=group_seed + 1)
                else:
                    samp_g = grp_global.sample(n)
                    samp_i = grp_iter.sample(n)

                std_g  = samp_g.std(ddof=1)
                mean_g = samp_g.mean()
                unc_g  = (2 * std_g / mean_g) * 100.0

                std_i  = samp_i.std(ddof=1)
                mean_i = samp_i.mean()
                unc_i  = (2 * std_i / mean_i) * 100.0

            temp.append({'n': n, 'unc_g': unc_g, 'unc_i': unc_i})

        # Determine maxima across all n
        unc_g_vals = [entry['unc_g'] for entry in temp if not pd.isna(entry['unc_g'])]
        unc_i_vals = [entry['unc_i'] for entry in temp if not pd.isna(entry['unc_i'])]
        max_unc_g = max(unc_g_vals) if unc_g_vals else np.nan
        max_unc_i = max(unc_i_vals) if unc_i_vals else np.nan

        # Second pass: build final records, using max_unc_g and max_unc_i
        for entry in temp:
            n = entry['n']
            unc_g = entry['unc_g']
            unc_i = entry['unc_i']

            if size < n:
                glob_unc = np.nan
                iter_unc = np.nan
            else:
                factor = np.sqrt(n)
                glob_unc = max_unc_g / factor if not pd.isna(max_unc_g) else np.nan
                iter_unc = max_unc_i / factor if not pd.isna(max_unc_i) else np.nan

            all_records.append({
                'DUT': dut,
                'ROR Setpoint': setpoint,
                'n': n,
                'Global Repeatability [%RDG]': unc_g,
                'Iterative Repeatability [%RDG]': unc_i,
                'Global Uncertainty [%RDG]': glob_unc,
                'Iterative Uncertainty [%RDG]': iter_unc
            })

    return pd.DataFrame.from_records(all_records)


unc_df = compute_two_sigma_with_uncertainty(summary, max_n=25, random_state=42)

def plot_repeatability_and_uncertainty(df):
    """
    For each DUT in df, create a single figure window with a 2x2 grid of subplots:
      - Top-left:  Global Repeatability [%RDG] vs. n
      - Top-right: Iterative Repeatability [%RDG] vs. n
      - Bottom-left:  Global Uncertainty [%RDG] vs. n
      - Bottom-right: Iterative Uncertainty [%RDG] vs. n

    Within each subplot, lines are colored by 'ROR Setpoint'.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
          - 'DUT'
          - 'ROR Setpoint'
          - 'n'
          - 'Global Repeatability [%RDG]'
          - 'Iterative Repeatability [%RDG]'
          - 'Global Uncertainty [%RDG]'
          - 'Iterative Uncertainty [%RDG]'
    """
    # Ensure required columns exist
    required = [
        'DUT', 'ROR Setpoint', 'n',
        'Global Repeatability [%RDG]',
        'Iterative Repeatability [%RDG]',
        'Global Uncertainty [%RDG]',
        'Iterative Uncertainty [%RDG]'
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input DataFrame: {missing}")

    # Group by DUT
    for dut, dfg in df.groupby('DUT'):
        # Sort by n to ensure lines are drawn in order
        dfg = dfg.sort_values('n')

        # Create figure with larger fonts
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        # Prepare colors for each unique ROR Setpoint
        setpoints = sorted(dfg['ROR Setpoint'].unique())
        cmap = plt.get_cmap('tab10')
        colors = {sp: cmap(i % 10) for i, sp in enumerate(setpoints)}

        # Define metrics: (subplot position, column name, title)
        metrics = [
            ((0, 0), 'Global Repeatability [%RDG]',   'Global Repeatability'),
            ((0, 1), 'Iterative Repeatability [%RDG]', 'Iterative Repeatability'),
            ((1, 0), 'Global Uncertainty [%RDG]',     'Global Uncertainty'),
            ((1, 1), 'Iterative Uncertainty [%RDG]',  'Iterative Uncertainty')
        ]

        for (row, col), colname, title in metrics:
            ax = axes[row][col]
            # Plot each ROR setpoint as a separate line
            for sp in setpoints:
                sub = dfg[dfg['ROR Setpoint'] == sp]
                ax.plot(
                    sub['n'],
                    sub[colname],
                    label=f"Setpoint {sp}",
                    color=colors[sp],
                    marker='o',
                    markersize=5
                )
            ax.set_xlabel('n Random Samples', fontsize=14)
            ax.set_ylabel(f"{title} [%RDG]", fontsize=14)
            ax.set_title(title, fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True)

        # Add legend to top-right subplot with larger font
        axes[0][1].legend(title='ROR Setpoint', loc='best', fontsize=12, title_fontsize=12)

        # Set super-title with larger font
        fig.suptitle(f"DUT {dut}: Ambient 7-Day 2σ Repeatability & Uncertainty", fontsize=18)

        plt.show()


# plot_repeatability_and_uncertainty(unc_df)
