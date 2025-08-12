# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 12:31:14 2025

@author: SFerneyhough
"""

from pathlib import Path
import pandas as pd
import re
import math
import os
import matplotlib.pyplot as plt

FOLDER_PATH = r'D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\N2 Baseline\2025.06.04 N2 Baseline\Results'

def load_processed_data(folder_path):
    """
    Load files ending with 'Processed' into a dict, then
    make a df_temperature with:
      - 'PT3 Temperature'
      - 'PT4 Temperature'
      - 'DUT Temperature Reading'
      - 'DUT'
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a valid directory")

    def _read_file(p: Path):
        suf = p.suffix.lower()
        if suf in {".csv", ".txt"}:
            return pd.read_csv(p)
        elif suf in {".xlsx", ".xls"}:
            return pd.read_excel(p)
        else:
            return None

    processed_files = [
        p for p in folder.iterdir()
        if p.is_file() and p.stem.endswith("Processed")
    ]

    files_dict = {}
    temp_frames = []

    for p in processed_files:
        df = _read_file(p)
        if df is None:
            continue

        files_dict[p.stem] = df

        # Base columns if they exist
        base_cols = [c for c in ["PT3 Temperature", "PT4 Temperature"] if c in df.columns]
        base_data = df[base_cols].copy() if base_cols else pd.DataFrame()

        # Find DUT temperature columns like "DUT1 Temperature Reading ..."
        dut_temp_cols = [c for c in df.columns if "Temperature Reading" in c]

        for col in dut_temp_cols:
            temp_df = base_data.copy()

            # Extract DUT number from the column name
            m = re.search(r"DUT\s*(\d+)", col, flags=re.IGNORECASE)
            if m:
                dut_num = int(m.group(1))
            else:
                # If no number in column name, try filename or NA
                m2 = re.search(r"DUT[_\s-]?(\d+)", p.stem, flags=re.IGNORECASE)
                dut_num = int(m2.group(1)) if m2 else pd.NA

            temp_df["DUT Temperature Reading"] = df[col]
            temp_df["DUT"] = dut_num
            temp_frames.append(temp_df)

    # Combine all DUT temperature data
    df_temperature = pd.concat(temp_frames, ignore_index=True) if temp_frames else pd.DataFrame(
        columns=["PT3 Temperature", "PT4 Temperature", "DUT Temperature Reading", "DUT"]
    )

    return files_dict, df_temperature

# Example:
files_dict, df_temperature = load_processed_data(FOLDER_PATH)

df_temp = df_temperature.iloc[::100,]
# print(df_temperature.head())

def plot_df_temp(df_temp: pd.DataFrame, x_col: str | None = None, save_dir: str | None = None):
    """
    Plot PT3, PT4, and DUT Temperature for each DUT, with 4 DUTs per figure.

    Parameters
    ----------
    df_temp : pd.DataFrame
        Must include columns:
          - 'PT3 Temperature'
          - 'PT4 Temperature'
          - 'DUT Temperature Reading'
          - 'DUT'
        Index will be used as x-axis if x_col is None.
    x_col : str or None
        Optional column to use for the x-axis (e.g., 'Time'). If None, uses the index.
    save_dir : str or None
        If provided, PNGs will be saved here. Directory is created if it doesn't exist.
    """
    required = ['PT3 Temperature', 'PT4 Temperature', 'DUT Temperature Reading', 'DUT']
    missing = [c for c in required if c not in df_temp.columns]
    if missing:
        raise ValueError(f"df_temp is missing required columns: {missing}")

    # Prepare x values
    if x_col is not None and x_col in df_temp.columns:
        x_vals = df_temp[x_col]
        x_label = x_col
    else:
        x_vals = df_temp.index
        x_label = "Sample"

    # Ensure clean types
    work = df_temp.copy()
    work = work.sort_index()
    work = work.dropna(subset=['DUT'])  # ensure we can group

    duts = sorted(pd.unique(work['DUT']))
    if len(duts) == 0:
        raise ValueError("No DUT values found to plot.")

    # Saving
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Chunk DUTs into groups of 4
    panels = [duts[i:i+4] for i in range(0, len(duts), 4)]

    for page_idx, dut_group in enumerate(panels, start=1):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.flatten()

        for ax, dut in zip(axes, dut_group):
            sub = work[work['DUT'] == dut]

            # x data per subset (keeps alignment even if we used index)
            x = sub[x_col] if (x_col is not None and x_col in sub.columns) else sub.index

            # scatter plots
            ax.scatter(x, sub['PT3 Temperature'], s=12, label='PT3 Temperature', marker='o')
            ax.scatter(x, sub['PT4 Temperature'], s=12, label='PT4 Temperature', marker='^')
            ax.scatter(x, sub['DUT Temperature Reading'], s=12, label='DUT Temperature', marker='s')

            ax.set_title(f"DUT {int(dut) if pd.notna(dut) else dut}", fontsize=12)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel("Temperature (°C)", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc="best")

        # Turn off any unused axes on the last page
        for j in range(len(dut_group), 4):
            axes[j].set_visible(False)

        fig.suptitle(f"Temperatures by DUT (Page {page_idx})", fontsize=14)

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"df_temp_page_{page_idx}.png")
            fig.savefig(out_path, dpi=200)

        plt.show()
        
# plot_df_temp(df_temp)