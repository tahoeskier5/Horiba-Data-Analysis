# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 09:48:05 2025

@author: SFerneyhough
"""

import os
import re
import pandas as pd
import numpy as np
from d500sim.linalgmethods import *
import matplotlib.pyplot as plt
import math
from scipy.optimize import least_squares

# ─── User inputs ───────────────────────────────────────────────────────────────
DEFAULT_DIR = r"D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\2nd soak\Results"
SHIFTED_DIR = r"D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\2nd soak\corrected_ROR\Results"

ROR_DIR = r'D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\ROR Drift'

def _lines_to_array(lines, path, dut, name):
    try:
        rows = []
        for l in lines:
            parts = [p for p in re.split(r',\s*', l) if p != '']
            rows.append([float(p) for p in parts])
        return np.vstack(rows)
    except Exception as e:
        raise ValueError(
            f"Failed to parse numeric block for DUT {dut} '{name}' in file {path!r}. "
            f"Raw lines: {lines!r}\nUnderlying error: {e}") from e


def parse_coeff_file(filepath):
    """
    Parse a coefficient text file into a dict:
      { dut_int: { coeff_key: [numpy_array_block1, ...] } }
    Supports multiple blocks per name.
    Blocks are expected in sections headed by lines like:
      DUT <id> <name>
    Followed by numeric lines, until next header or EOF.

    Coeff_key is normalized to the first word of <name>, e.g. 'Master', 'Temperature', 'Clone'.
    """
    data = {}
    current_dut = None
    current_name = None
    block_lines = []

    # Header: capture DUT number and full name
    header_re = re.compile(r"^DUT\s*(\d+)\s+(.+)$")

    def normalize(name_full):
        return name_full.split()[0]

    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            # header line?
            m = header_re.match(line)
            if m:
                # flush previous block
                if current_dut is not None and block_lines:
                    key = normalize(current_name)
                    arr = _lines_to_array(block_lines, filepath, current_dut, key)
                    data.setdefault(current_dut, {}).setdefault(key, []).append(arr)
                    block_lines = []
                # start new section
                current_dut = int(m.group(1))
                current_name = m.group(2)
                continue

            # numeric data line?
            if current_dut is not None and line and re.match(r"^[\d\-\.E]+(,\s*[\d\-\.E]+)*$", line):
                block_lines.append(line)
                continue

            # other line → flush block
            if current_dut is not None and block_lines:
                key = normalize(current_name)
                arr = _lines_to_array(block_lines, filepath, current_dut, key)
                data.setdefault(current_dut, {}).setdefault(key, []).append(arr)
                block_lines = []

    # EOF flush
    if current_dut is not None and block_lines:
        key = normalize(current_name)
        arr = _lines_to_array(block_lines, filepath, current_dut, key)
        data.setdefault(current_dut, {}).setdefault(key, []).append(arr)

    return data


def load_all_dut_coefficients(folder_path):
    """
    Walk the folder, parse all .txt coefficient files, and return:
      { dut_int: { coeff_key: concatenated_numpy_array } }
    Requires at least a 'Master' block per DUT; missing ones will be warned.
    """
    master = {}
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path!r}")

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith('.txt'):
            continue
        full = os.path.join(folder_path, fname)
        print(f"[+] Parsing file: {full}")
        file_data = parse_coeff_file(full)
        for dut, coeffs in file_data.items():
            if 'Master' not in coeffs:
                print(f"[!] Missing 'Master' block for DUT {dut} in {fname}; skipping DUT {dut}")
                continue
            for key, blocks in coeffs.items():
                concatenated = np.vstack(blocks) if len(blocks) > 1 else blocks[0]
                master.setdefault(dut, {})[key] = concatenated
                print(f"    -> DUT {dut}, '{key}' coeffs shape: {concatenated.shape}")
    return master


def prepend_master_mn(coeff_array, m=5, n=5):
    if coeff_array is None:
        raise ValueError("Coefficient array is missing.")
    flat = np.ravel(coeff_array)
    # Build list with Python int for m,n and Python floats for the rest, force object dtype
    items = [int(m), int(n)] + [float(v) for v in flat.tolist()]
    return np.array(items, dtype=object)

def MDoutput(Tcoeffs, MDcoeffs, P1, P2, T, df, PowX):
    MFC2 = d500sim(df['P1[torr]'], df['P2[torr]'],
                   df['Temp[degC]'], df['Flow[SCCM]'], PowX=PowX)
    MFC2.Tempcoeffs = Tcoeffs
    LBX = np.log10(np.power(P1, PowX) - np.power(P2, PowX))
    arr = createMatrixA(LBX, P2, int(MDcoeffs[0]), int(MDcoeffs[1]))
    MDout = 10.0 ** np.matmul(arr, MDcoeffs[2:])
    if T.iloc[0] != 298.15:
        TempRatio = 298.15 / (T + 273.15)
        Tfactor = MFC2.T_fit_output(TempRatio, LBX)
        MDout = MDout * Tfactor
    return MDout * 72


def cf_output(P1, P2, T, CF_Coeffs, PowX=1.9099999666214):
    """
    Simplified clone flow output using preprocessed CF_Coeffs DataFrame from std_vector.
    """
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    # T is not used in current formula but kept for signature compatibility
    LBX = np.log10(np.power(P1, PowX) - np.power(P2, PowX))
    arr = createMatrixA(LBX, P2,
                        int(CF_Coeffs['Coefficients'][0]),
                        int(CF_Coeffs['Coefficients'][1]))
    CFoutput = np.matmul(arr, CF_Coeffs['Coefficients'][2:])
    return CFoutput


def compute_all_MDoutputs(default_dir, shifted_dir, coeff_dict, PowX, PowY=None):
    """
    PowY is unused in this simplified cf_output version but retained for interface compatibility.
    """
    results = {'default': {}, 'shifted': {}}

    for label, folder in (('default', default_dir),
                          ('shifted', shifted_dir)):
        for fname in sorted(os.listdir(folder)):
            base, ext = os.path.splitext(fname)
            if not base.endswith('Processed') or ext.lower() not in ('.csv', '.xlsx'):
                continue

            path = os.path.join(folder, fname)
            df = pd.read_csv(path) if ext.lower() == '.csv' else pd.read_excel(path)

            # locate required columns
            try:
                p1_col = next(c for c in df.columns if 'Pressure 1 Reading' in c)
                p2_col = next(c for c in df.columns if 'Pressure 2 Reading' in c)
                flow_col = next(c for c in df.columns if 'Flow Reading' in c)
                temp_col = next(c for c in df.columns if 'Temperature Reading' in c)
            except StopIteration as e:
                print(f"[!] Required column not found in {fname}: {e}; skipping.")
                continue

            # extract DUT number
            dut_int = None
            m_match = re.search(r'DUT\s*(\d+)', p1_col, re.IGNORECASE)
            if m_match:
                dut_int = int(m_match.group(1))
            else:
                m2 = re.search(r'(\d+)', base)
                if m2:
                    dut_int = int(m2.group(1))
            if dut_int is None:
                print(f"[!] Could not determine DUT for file {fname}; skipping.")
                continue

            # lookup coefficient blocks
            master_block = coeff_dict.get(dut_int, {}).get("Master")
            temp_block = coeff_dict.get(dut_int, {}).get("Temperature")
            clone_block = coeff_dict.get(dut_int, {}).get("Clone")  # optional

            if master_block is None:
                print(f"[!] Missing 'Master' block for DUT {dut_int}; skipping {fname}.")
                continue
            if temp_block is None:
                print(f"[!] Missing 'Temperature' block for DUT {dut_int}; skipping {fname}.")
                continue

            # prepare MDcoeffs
            raw_mdvec = prepend_master_mn(master_block, 5, 5)
            md_df = std_vector(raw_mdvec)
            MDcoeffs_vec = md_df['Coefficients'].to_numpy()

            # prepare Tcoeffs
            Tcoeffs_vec = np.ravel(temp_block)

            # rename columns for MDoutput
            df = df.rename(columns={
                p1_col: 'P1[torr]',
                p2_col: 'P2[torr]',
                flow_col: 'Flow[SCCM]',
                temp_col: 'Temp[degC]'
            })

            # compute MDoutput
            df['MDoutput'] = MDoutput(
                Tcoeffs_vec, MDcoeffs_vec,
                df['P1[torr]'], df['P2[torr]'],
                df['Temp[degC]'], df, PowX
            )

            # apply simplified clone fit if available
            if clone_block is not None:
                raw_clone_vec = prepend_master_mn(clone_block, 5, 5)
                clone_df = std_vector(raw_clone_vec)
                df['Clonefitoutput'] = cf_output(
                    df['P1[torr]'], df['P2[torr]'], df['Temp[degC]'],
                    clone_df,  # std_vector output
                )
                df['Clone Flow'] = df['Clonefitoutput'] * df['MDoutput']
                df['Clonefiterror'] = df['Clone Flow'] / df['Flow[SCCM]'] - 1
            else:
                df['Clonefitoutput'] = np.nan
                df['Clonefiterror'] = np.nan
                
          
            results[label][f"DUT{dut_int}"] = df

    return results

def compute_cloneflow_differences(all_results):
    """
    Given the dict from compute_all_MDoutputs, returns:
      1) diffs: dict mapping each DUT to the row-wise DataFrame with columns:
         P1_default, P2_default, CloneFlow_default,
         P1_shifted, P2_shifted, CloneFlow_shifted,
         ROR Setpoint,
         CloneFlow_diff, CloneFlow_diff_percent
      2) diffs_mean_df: single concatenated DataFrame of the groupby('ROR Setpoint').mean()
         summaries with '_mean' suffixes and a 'DUT' column.
    """
    diffs = {}

    for dut, df_def in all_results['default'].items():
        df_sh = all_results['shifted'].get(dut)
        if df_sh is None:
            continue

        # require Clone Flow in both
        if 'Clone Flow' not in df_def.columns or 'Clone Flow' not in df_sh.columns:
            continue

        # build slim DataFrame (assumes same row order)
        df = pd.DataFrame({
            'P1_default':        df_def['P1[torr]'],
            'P2_default':        df_def['P2[torr]'],
            'CloneFlow_default': df_def['Clone Flow'],
            'P1_shifted':        df_sh['P1[torr]'],
            'P2_shifted':        df_sh['P2[torr]'],
            'CloneFlow_shifted': df_sh['Clone Flow'],
            'ROR Setpoint':      df_def['ROR Setpoint']
        }).reset_index(drop=True)

        # compute diffs
        df['CloneFlow_diff'] = df['CloneFlow_default'] - df['CloneFlow_shifted']
        df['CloneFlow_diff_percent'] = (
            np.divide(
                df['CloneFlow_diff'],
                df['CloneFlow_default'],
                out=np.full_like(df['CloneFlow_diff'], np.nan, dtype=float),
                where=df['CloneFlow_default'] != 0
            ) * 100
        )

        diffs[dut] = df

    # build concatenated mean DataFrame
    summary_frames = []
    for dut, df in diffs.items():
        grouped = (
            df.groupby('ROR Setpoint', as_index=False)
              .agg({
                  'P1_default':         'mean',
                  'P2_default':         'mean',
                  'CloneFlow_default':  'mean',
                  'P1_shifted':         'mean',
                  'P2_shifted':         'mean',
                  'CloneFlow_shifted':  'mean',
                  'CloneFlow_diff':     'mean',
                  'CloneFlow_diff_percent': 'mean',
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
                'P1_default_mean', 'P2_default_mean', 'CloneFlow_default_mean',
                'P1_shifted_mean', 'P2_shifted_mean', 'CloneFlow_shifted_mean',
                'CloneFlow_diff_mean', 'CloneFlow_diff_percent_mean'
            ]
        )

    return diffs, diffs_mean_df

def load_flow_drift_global(ROR_DIR, hf_hours=564):
    """
    Read every .xlsx in ROR_DIR, iterate all sheets (sheet name = DUT),
    filter by HF_Soak_Hours == hf_hours, and return a combined DataFrame
    with columns:
      ['DUT', 'time', 'setpoint',
       'Flow Drift Global (sccm)',
       'Flow Drift Global (%RS)',
       'Flow Drift Average (%RS)']
    """
    records = []
    for fname in sorted(os.listdir(ROR_DIR)):
        if not fname.lower().endswith('.xlsx'):
            continue

        path = os.path.join(ROR_DIR, fname)
        # read all sheets into a dict: { sheet_name: DataFrame, ... }
        sheets = pd.read_excel(path, sheet_name=None)
        for sheet_name, df in sheets.items():
            # assume sheet_name like "DUT14" or just "14"
            dut = sheet_name

            # filter soak-hours
            dff = df[df['HF_Soak_Hours'] == hf_hours]
            if dff.empty:
                continue

            # collect only the needed columns
            for _, row in dff.iterrows():
                records.append({
                    'DUT':                     dut,
                    'time':                    row['time'],
                    'setpoint':                row['setpoint'],
                    'Flow Drift Global (sccm)':   row['Flow Drift Global (sccm)'],
                    'Flow Drift Global (%RS)':    row['Flow Drift Global (%RS)'],
                    'Flow Drift Average (%RS)':   row['Flow Drift Average (%RS)']
                })

    combined_df = pd.DataFrame(records)
    
    return combined_df


def plot_flow_drift(df):
    """
    Scatter plot of Flow Drift Average and Calculated Drift vs. setpoint.
    Four DUTs per figure, independent y-axes, with larger fonts.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
          - 'DUT'
          - 'setpoint'
          - 'Flow Drift Global (%RS)'
          - 'Calculated Drift [%RDG]'
    """
    # Identify unique DUTs
    duts = df['DUT'].unique()
    # Split into panels of 4
    panels = [duts[i:i+4] for i in range(0, len(duts), 4)]
    
    for panel_idx, panel_duts in enumerate(panels, start=1):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for ax, dut in zip(axes, panel_duts):
            sub = df[df['DUT'] == dut]
            ax.scatter(sub['setpoint'], sub['Flow Drift Global (%RS)'],
                       label='Measured Drift', marker='o')
            ax.scatter(sub['setpoint'], sub['Pressure Sensor Induced Drift [%RDG]'],
                       label='Pressure Compensated Drift', marker='x')
            ax.set_title(dut, fontsize=16)
            ax.set_xlabel('Setpoint', fontsize=14)
            ax.set_ylabel('Drift [%]', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(fontsize=12)
            ax.grid(True)
        
        # Remove unused subplots if fewer than 4 DUTs in this panel
        for ax in axes[len(panel_duts):]:
            fig.delaxes(ax)
        
        fig.suptitle(f'Measured Drift - Pressure Sensor Induced Drift, Panel {panel_idx}, 564 HF Hours',
                     fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        
# Example usage:
if __name__ == "__main__":
    folder = r"\\shr-eng-srv\RnD-Project\HORIBA_Test_and_Dev_Systems\Corrosion_Test_Bench\2025-05_MFC_HF_Drift2\DUT Flow Coefficients"
    coeff_dict = load_all_dut_coefficients(folder)

    all_results = compute_all_MDoutputs(DEFAULT_DIR, SHIFTED_DIR, coeff_dict, PowX=1.91,PowY=None)
    
    diff_results, diff_mean = compute_cloneflow_differences(all_results)
    
    combined = load_flow_drift_global(ROR_DIR, hf_hours=564)
    
    combined['Calculated Drift [%RDG]'] = diff_mean['CloneFlow_diff_percent_mean'] 
    
    combined['Pressure Sensor Induced Drift [%RDG]'] = combined['Flow Drift Average (%RS)'] - combined['Calculated Drift [%RDG]']
    
    plot_flow_drift(combined)
