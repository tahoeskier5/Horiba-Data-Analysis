# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:51:58 2025
@author: SFerneyhough
"""

import os
import pandas as pd

def compare_values(def_val, actual, tol=1e-6):
    if actual is None:
        return False
    s_def = str(def_val).strip()
    s_act = str(actual).strip()
    if ' ' in s_def and ' ' in s_act:
        toks_def = s_def.split(); toks_act = s_act.split()
        if len(toks_def) != len(toks_act):
            return False
        try:
            nums_def = [float(t) for t in toks_def]
            nums_act = [float(t) for t in toks_act]
            return all(abs(a - d) <= tol for a, d in zip(nums_act, nums_def))
        except ValueError:
            return toks_act == toks_def
    try:
        return abs(float(s_act) - float(s_def)) <= tol
    except ValueError:
        return s_act == s_def

# 1) load local CSVs
print("Loading local CSVs...")
local_dir = r'D:\Desktop\AFS Data Walk'
files = [
    'flagged_df_all.csv',
    'AFS 1.6.3 Default Parameter Values.csv',
    'AFS 1.7.0 Default Parameter Values.csv'
]
dfs = { os.path.splitext(f)[0]: pd.read_csv(os.path.join(local_dir, f)) for f in files }
df_flagged = dfs['flagged_df_all']
dp_163     = dfs['AFS 1.6.3 Default Parameter Values']
dp_170     = dfs['AFS 1.7.0 Default Parameter Values']
print(f"  flagged devices: {len(df_flagged)}")
print(f"  defaults 1.6.3:   {len(dp_163)}, 1.7.0: {len(dp_170)}")

# ensure Serial Number is string and build firmware lookup
df_flagged['Serial Number'] = df_flagged['Serial Number'].astype(str)
firmware_lookup = df_flagged.set_index('Serial Number')['Firmware Version'].to_dict()

# 2) build default lookups per firmware
def build_defaults(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df['Name']  = df['Name'].str.strip()
    df['Value'] = df['Value'].str.strip()
    return df.set_index('Name')['Value'].to_dict()

defaults = {
    '1.6.3': build_defaults(dp_163),
    '1.7.0': build_defaults(dp_170),
}

# 3) prepare outputs & detect already-done
match_csv    = os.path.join(local_dir, 'df_matches.csv')
mismatch_csv = os.path.join(local_dir, 'df_mismatches.csv')
processed = set()
for path in (match_csv, mismatch_csv):
    if os.path.exists(path):
        processed |= set(pd.read_csv(path)['Serial Number'].astype(str).unique())
print(f"Already processed: {len(processed)} devices")

# 4) batch settings
batch_size     = 50   # flush every 50 devices
batch_processed= 0
matches_batch  = []
mismatches_batch = []

server_dir = r'\\shr-eng-srv\Prod-Data-Copy\DATA'
print("Starting full server walk...")

for root, dirs, files in os.walk(server_dir):
    serial = os.path.basename(root)
    if serial not in df_flagged['Serial Number'].values:
        continue
    if serial in processed:
        continue

    fw = firmware_lookup.get(serial, '1.6.3')
    default_dict = defaults.get(fw, defaults['1.6.3'])
    print(f"Processing SN={serial} (FW={fw})")

    # find FinalChecker
    for d in dirs:
        if not d.endswith('FinalChecker'):
            continue
        final_dir = os.path.join(root, d)

        # read first CSV
        for fname in os.listdir(final_dir):
            if not fname.lower().endswith('.csv'):
                continue
            df_csv = pd.read_csv(
                os.path.join(final_dir, fname),
                sep=',', comment='#',
                header=None,
                names=['Address','Flag','Name','Value'],
                usecols=['Name','Value'],
                dtype=str
            )
            df_csv['Name']  = df_csv['Name'].str.strip()
            df_csv['Value'] = df_csv['Value'].str.strip()
            actual = df_csv.set_index('Name')['Value'].to_dict()

            # split into matches/mismatches
            match_recs, mismatch_recs = [], []
            for p, dv in default_dict.items():
                val = actual.get(p)
                rec = {
                    'Serial Number':    serial,
                    'Firmware Version': fw,
                    'Parameter':        p,
                    'Default':          dv,
                    'Value':            val
                }
                if compare_values(dv, val):
                    match_recs.append(rec)
                else:
                    mismatch_recs.append(rec)

            # add to batch lists
            matches_batch.extend(match_recs)
            mismatches_batch.extend(mismatch_recs)

            processed.add(serial)
            batch_processed += 1
            break
        break

    # flush batch?
    if batch_processed >= batch_size:
        print(f"Flushing batch of {batch_processed} devices...")
        # write matches
        df_m = pd.DataFrame(matches_batch)
        df_m.to_csv(match_csv, mode='a', index=False,
                    header=not os.path.exists(match_csv))
        # write mismatches
        df_mm = pd.DataFrame(mismatches_batch)
        df_mm.to_csv(mismatch_csv, mode='a', index=False,
                     header=not os.path.exists(mismatch_csv))
        print(f"  Wrote {len(matches_batch)} matches, {len(mismatches_batch)} mismatches")
        # reset batch
        batch_processed = 0
        matches_batch.clear()
        mismatches_batch.clear()

# final flush
if matches_batch or mismatches_batch:
    print("Final flush...")
    pd.DataFrame(matches_batch).to_csv(
        match_csv, mode='a', index=False, header=not os.path.exists(match_csv))
    pd.DataFrame(mismatches_batch).to_csv(
        mismatch_csv, mode='a', index=False, header=not os.path.exists(mismatch_csv))
    print(f"  Wrote {len(matches_batch)} matches, {len(mismatches_batch)} mismatches")

print("All done.")
print(f"Results in:\n  {match_csv}\n  {mismatch_csv}")
