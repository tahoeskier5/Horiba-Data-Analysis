# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 13:54:46 2025

@author: SFerneyhough
"""

import os
import re
import pandas as pd
import numpy as np
from d500sim.linalgmethods import *
import matplotlib.pyplot as plt
import math

# ─── User inputs ───────────────────────────────────────────────────────────────
DEFAULT_DIR = r"D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\1st soak\Results"
SHIFTED_DIR = r"D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\1st soak\corrected_ROR\Results"

# MDoutput coefficients
PowX = 1.91
Tcoeffs  = [-26.45169198, 39.63130225, -20.93241902, 4.887365797, -.4248112]     # your temp coeffs
MDcoeffs = [
    5, 5,
    -4.55081,
     0.888211,
    -0.044105,
     0.0146654,
    -0.00107361,
     0,
    -0.0211126,
     0.0150843,
    -0.00408569,
     0.000496255,
    -2.27499e-05,
     0,
     8.12989e-05,
    -5.91851e-05,
     1.6391e-05,
    -2.03683e-06,
     9.54491e-08,
     0,
    -1.09628e-07,
     8.05491e-08,
    -2.24987e-08,
     2.81768e-09,
    -1.32974e-10,
     0,
     4.83421e-11,
    -3.57499e-11,
     1.0043e-11,
    -1.26417e-12,
     5.99272e-14,
     0,
     0,
     0,
     0,
     0,
     0,
     0
]

# ────────────────────────────────────────────────────────────────────────────────

def MDoutput(Tcoeffs, MDcoeffs, P1, P2, T, df, PowX):
    MFC2 = d500sim(df['P1[torr]'], df['P2[torr]'],
                   df['Temp[degC]'], df['Flow[SCCM]'], PowX=PowX)
    MFC2.Tempcoeffs = Tcoeffs
    LBX = np.log10(np.power(P1, PowX) - np.power(P2, PowX))
    arr = createMatrixA(LBX, P2, int(MDcoeffs[0]), int(MDcoeffs[1]))
    MDout = 10.0**np.matmul(arr, MDcoeffs[2:])
    if T.iloc[0] != 298.15:
        TempRatio = 298.15 / (T + 273.15)
        Tfactor   = MFC2.T_fit_output(TempRatio, LBX)
        MDout     = MDout * Tfactor
    return MDout * 72

def compute_all_MDoutputs(default_dir, shifted_dir,
                          Tcoeffs, MDcoeffs, PowX):
    results = {'default': {}, 'shifted': {}}

    for label, folder in (('default', default_dir),
                          ('shifted', shifted_dir)):
        for fname in os.listdir(folder):
            base, ext = os.path.splitext(fname)
            # only process files whose name ends with 'Processed'
            if not base.endswith('Processed') or ext.lower() not in ('.csv','.xlsx'):
                continue

            path = os.path.join(folder, fname)
            # read in CSV or Excel
            df = pd.read_csv(path) if ext.lower()=='.csv' else pd.read_excel(path)

            # find the four DUT columns
            p1_col   = next(c for c in df.columns if 'Pressure 1 Reading'   in c)
            p2_col   = next(c for c in df.columns if 'Pressure 2 Reading'   in c)
            flow_col = next(c for c in df.columns if 'Flow Reading'         in c)
            temp_col = next(c for c in df.columns if 'Temperature Reading'  in c)

            # extract DUT name
            dut_match = re.search(r'(DUT\d+)', p1_col)
            dut = dut_match.group(1) if dut_match else base

            # rename to MDoutput inputs
            df = df.rename(columns={
                p1_col:   'P1[torr]',
                p2_col:   'P2[torr]',
                flow_col: 'Flow[SCCM]',
                temp_col: 'Temp[degC]'
            })

            # compute and append
            df['MDoutput'] = MDoutput(
                Tcoeffs, MDcoeffs,
                df['P1[torr]'], df['P2[torr]'],
                df['Temp[degC]'], df, PowX
            )

            results[label][dut] = df

    return results

def compute_MDoutput_differences(all_results):
    """
    Given the dict from compute_all_MDoutputs, returns:
      1) diffs: dict mapping each DUT to the row-wise DataFrame with columns:
         P1_default, P2_default, MD_default,
         P1_shifted, P2_shifted, MD_shifted,
         ROR Setpoint,
         MD_diff_torr, MD_diff_percent
      2) diffs_mean_df: single concatenated DataFrame of the groupby('ROR Setpoint').mean()
         summaries with '_mean' suffixes and a 'DUT' column.
    """
    diffs = {}

    for dut, df_def in all_results['default'].items():
        df_sh = all_results['shifted'].get(dut)
        if df_sh is None:
            continue

        # build slim DataFrame (assumes same row order)
        df = pd.DataFrame({
            'P1_default':  df_def['P1[torr]'],
            'P2_default':  df_def['P2[torr]'],
            'MD_default':  df_def['MDoutput'],
            'P1_shifted':  df_sh['P1[torr]'],
            'P2_shifted':  df_sh['P2[torr]'],
            'MD_shifted':  df_sh['MDoutput'],
            'ROR Setpoint': df_def['ROR Setpoint']
        }).reset_index(drop=True)

        # compute diffs
        df['MD_diff_torr'] = df['MD_default'] - df['MD_shifted']
        df['MD_diff_percent'] = (
            np.divide(
                df['MD_diff_torr'],
                df['MD_default'],
                out=np.full_like(df['MD_diff_torr'], np.nan, dtype=float),
                where=df['MD_default'] != 0
            ) * 100
        )

        diffs[dut] = df

    # build concatenated mean DataFrame
    summary_frames = []
    for dut, df in diffs.items():
        grouped = (
            df.groupby('ROR Setpoint', as_index=False)
              .agg({
                  'P1_default': 'mean',
                  'P2_default': 'mean',
                  'MD_default': 'mean',
                  'P1_shifted': 'mean',
                  'P2_shifted': 'mean',
                  'MD_shifted': 'mean',
                  'MD_diff_torr': 'mean',
                  'MD_diff_percent': 'mean',
              })
              .rename(columns=lambda c: c if c == 'ROR Setpoint' else f"{c}_mean")
        )
        grouped.insert(0, 'DUT', dut)
        summary_frames.append(grouped)

    if summary_frames:
        diffs_mean_df = pd.concat(summary_frames, ignore_index=True)
    else:
        diffs_mean_df = pd.DataFrame(
            columns=[
                'DUT', 'ROR Setpoint',
                'P1_default_mean', 'P2_default_mean', 'MD_default_mean',
                'P1_shifted_mean', 'P2_shifted_mean', 'MD_shifted_mean',
                'MD_diff_torr_mean', 'MD_diff_percent_mean'
            ]
        )

    return diffs, diffs_mean_df

def plot_MD_diff_batches(diff_results, batch_size=4):
    """
    For each batch of up to `batch_size` DUTs, create one figure with
    a grid of individual scatter plots (P1_default vs MD_diff_percent).
    
    Parameters
    ----------
    diff_results : dict
        Mapping DUT → DataFrame with 'P1_default' and 'MD_diff_percent'.
    batch_size : int, optional
        Number of DUTs per figure (default 4).
    """
    duts = list(diff_results.keys())
    n_batches = math.ceil(len(duts) / batch_size)
    
    for b in range(n_batches):
        batch = duts[b*batch_size:(b+1)*batch_size]
        cols = 2
        rows = math.ceil(len(batch) / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)
        fig.suptitle(f"MDoutput % Difference vs P1 (Panel {b+1})", fontsize=16)
        
        for idx, dut in enumerate(batch):
            r = idx // cols
            c = idx % cols
            ax = axes[r][c]
            df = diff_results[dut]
            ax.scatter(
                df['P1_default'],
                df['MD_diff_percent'],
                marker='o'
            )
            ax.set_title(dut, fontsize=14)
            ax.set_xlabel('P1 [Torr]', fontsize=12)
            ax.set_ylabel('% Diff', fontsize=12)
            ax.grid(True)
        
        # hide any unused subplots
        total_plots = rows * cols
        for empty_idx in range(len(batch), total_plots):
            r = empty_idx // cols
            c = empty_idx % cols
            fig.delaxes(axes[r][c])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

if __name__ == '__main__':
    all_results = compute_all_MDoutputs(
        DEFAULT_DIR, SHIFTED_DIR,
        Tcoeffs, MDcoeffs, PowX
    )
    diff_results, diff_mean = compute_MDoutput_differences(all_results)
    # plot_MD_diff_batches(diff_results, batch_size=4)
