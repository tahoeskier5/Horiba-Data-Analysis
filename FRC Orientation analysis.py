# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:51:31 2025

@author: SFerneyhough
"""

import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt

# 1. Point to your folder
folder = r"D:\Desktop\FRC Orientation Analysis\Orientation 1"
pattern = os.path.join(folder, "*.csv")

# 2. Read, tag, and early-assign “Start”
dfs = []
for path in glob.glob(pattern):
    df = pd.read_csv(path)
    
    
    # now extract the Run # from the filename
    fname = os.path.basename(path)
    m = re.search(r"Run\s*(\d+)", fname, re.IGNORECASE)
    run = int(m.group(1)) if m else None
    df['Run #'] = run
    
    dfs.append(df)

# 3. Combine into one DataFrame
combined = pd.concat(dfs, ignore_index=True)

# 4. Extract numeric molbox flow, rename & drop original
molcol = 'COM8-1:molbox1+-0x01-1:Flow (sccm)-1'
combined['Molbox Flow [sccm]'] = (
    combined[molcol]
      .astype(str)
      .str.extract(r'([0-9]+(?:\.[0-9]+)?)')[0]
      .astype(float)
)
combined.drop(columns=[molcol], inplace=True)
combined = combined[combined['Molbox Flow [sccm]'] != 9]

# 5. Define the common and per-channel columns
common = [
    'Time',
    'Relative Time (ms)',
    'Molbox Flow [sccm]',
    'Run #'
]

channel_map = {
    1: ['COM10-0:FRC-PROTO-4CH-0x01-1:Ch.1  Flow-1',  'Channel 1'],
    2: ['COM10-0:FRC-PROTO-4CH-0x01-1:Ch. 2 Flow-2','Channel 2'],
    3: ['COM10-0:FRC-PROTO-4CH-0x01-1:Ch. 3 Flow-3','Channel 3'],
    4: ['COM10-0:FRC-PROTO-4CH-0x01-1:Ch. 4 Flow-54','Channel 4'],
}

# 5. Build one DataFrame per channel
df_ch1 = combined[common + channel_map[1]].copy()
df_ch2 = combined[common + channel_map[2]].copy()
df_ch3 = combined[common + channel_map[3]].copy()
df_ch4 = combined[common + channel_map[4]].copy()



# 6. Forward‐ and back‐fill all four dataframes
for df in (df_ch1, df_ch2, df_ch3, df_ch4):
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
# 7. Filter each one for rows where the “Channel #” column contains “Start”
df_ch1 = df_ch1[df_ch1['Channel 1'].str.contains('Start', na=False)].reset_index()
df_ch2 = df_ch2[df_ch2['Channel 2'].str.contains('Start', na=False)].reset_index()
df_ch3 = df_ch3[df_ch3['Channel 3'].str.contains('Start', na=False)].reset_index()
df_ch4 = df_ch4[df_ch4['Channel 4'].str.contains('Start', na=False)].reset_index()

for ch, df in [(1, df_ch1), (2, df_ch2), (3, df_ch3), (4, df_ch4)]:
    flow_col = channel_map[ch][0]
    df['Flow Accuracy [%RDG]'] = (
        (df[flow_col] - df['Molbox Flow [sccm]'])
        / df['Molbox Flow [sccm]']
    ) * 100
    
# 8. Rename the per‐channel cols to generic names and tag Channel ID
to_concat = []
for ch, df in [(1, df_ch1), (2, df_ch2), (3, df_ch3), (4, df_ch4)]:
    flow_col, ch_col = channel_map[ch]
    tmp = df.rename(columns={
        flow_col:    'Channel Flow [sccm]',
        ch_col:      'Channel #'
    }).copy()
    tmp['Channel ID'] = ch
    to_concat.append(tmp)

# 9. Combine all into one DataFrame
df_all = pd.concat(to_concat, ignore_index=True)

df_all.rename(columns={'Channel #': 'Setpoint'}, inplace=True)

# Extract the numeric part (e.g. "250 Start" → 250) and make it float/int
df_all['Setpoint'] = (
    df_all['Setpoint']
      .astype(str)
      .str.extract(r'(\d+\.?\d*)')[0]
      .astype(float)
)

# 10. (Optional) reorder columns
cols = [
    'Channel ID', 'Setpoint', 'Channel Flow [sccm]',
    'Molbox Flow [sccm]', 'Flow Accuracy [%RDG]',
    'Time', 'Relative Time (ms)', 'Run #'
]
df_all = df_all[cols]

df_group = df_all.groupby(['Channel ID','Setpoint','Run #']).tail(20)
df_desc = df_group.groupby(['Channel ID','Setpoint','Run #']).describe()

# Flatten any MultiIndex columns
df_desc.columns = [
    f"{col_name}_{agg}" if agg else f"{col_name}"
    for col_name, agg in df_desc.columns.to_flat_index()
]

df_group_repeat = df_all.groupby(['Channel ID','Setpoint'],as_index=False).tail(20)
df_desc_repeat = df_group_repeat.groupby(['Channel ID','Setpoint'],as_index=False).describe()

# Flatten any MultiIndex columns
df_desc_repeat.columns = [
    f"{col_name}_{agg}" if agg else f"{col_name}"
    for col_name, agg in df_desc_repeat.columns.to_flat_index()
]


def plot_accuracy_std(df):
    """
    Scatter plot of Flow Accuracy [%RDG] standard deviation vs. Setpoint, colored by Channel ID.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
          - 'Setpoint'
          - 'Flow Accuracy [%RDG]_std'
          - 'Channel ID'
    """
    fig, ax = plt.subplots()
    # plot one scatter per channel
    for channel_id, group in df.groupby('Channel ID'):
        ax.scatter(
            group['Setpoint'],
            group['Flow Accuracy [%RDG]_std'],
            label=f"Channel {int(channel_id)}"
        )

    ax.set_xlabel('Setpoint [sccm]')
    ax.set_ylabel('1σ Flow Repeatability [%RDG]')
    ax.set_title('DF-100 Flow Repeatability, Orientation 1')
    ax.legend(title='Channel ID')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
plot_accuracy_std(df_desc_repeat)
