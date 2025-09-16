# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:05:21 2025

@author: SFerneyhough
"""
import os
import pandas as pd

def extract_metadata_and_thresholds(csv_path, debug: bool = False):
    check_fields = {
        'Channel P2 Pressure High Error - Threshold',
        'Channel P2 Pressure High Warning - Threshold',
        'Channel P1 Pressure High Error - Threshold',
        'Channel P1 Pressure High Warning - Threshold',
    }
    meta_fields = {
        'Model Name', 'Serial Number', 'Build Date', 'Firmware Version'
    }

    metadata   = {}
    thresholds = {}
    parsed     = []   # for debug: list of (field, val)

    # sniff delimiter
    with open(csv_path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            delim = ',' if ',' in line else '\t'
            break

    # parse
    with open(csv_path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue

            parts = [p.strip() for p in line.split(delim, 3)]
            if len(parts) < 4:
                continue

            field, val = parts[2], parts[3]
            parsed.append(field)

            # only grab metadata once per key
            if field in meta_fields and field not in metadata:
                metadata[field] = val
                if debug:
                    print(f"  → metadata[{field!r}] = {val!r}")

            if field in check_fields:
                num = float(val.split()[0])
                thresholds[field] = num
                if debug:
                    print(f"  → thresholds[{field!r}] = {num!r}")

    if debug:
        seen = set(parsed)
        print(f"\n‣ DEBUG: Fields seen in this file: {sorted(seen)}")
        for mf in sorted(meta_fields):
            ok = "✓" if mf in metadata else "✗"
            print(f"    {ok} metadata field {mf!r}")
        for cf in sorted(check_fields):
            ok = "✓" if cf in thresholds else "✗"
            print(f"    {ok} threshold field {cf!r}")

    # now check only *metadata* is required; thresholds may legitimately be empty
    missing_meta = meta_fields - set(metadata)
    if missing_meta:
        raise KeyError(f"Missing metadata fields {missing_meta} in {csv_path!r}")

    return metadata, thresholds



def find_flagged_final_checkers(
    base_dir: str,
    max_to_test: int = 50,
    batch_size: int = 10,
    output_csv: str = None
) -> pd.DataFrame:
    """
    Scans up to max_to_test 'FinalChecker' dirs in batches of batch_size.
    Returns a DataFrame, flagged_df, containing only those units where
    all 4 thresholds exist (no value filtering).

    Columns:
      - Serial Number
      - Model Name
      - Build Date
      - Firmware Version
      - Channel P2 Pressure High Error - Threshold
      - Channel P2 Pressure High Warning - Threshold
      - Channel P1 Pressure High Error - Threshold
      - Channel P1 Pressure High Warning - Threshold

    If output_csv is provided, appends flagged_df to that file (no header).
    """
    # gather all FinalChecker directories
    checker_dirs = []
    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            if d.endswith('FinalChecker'):
                checker_dirs.append(os.path.join(root, d))
                if len(checker_dirs) >= max_to_test:
                    break
        if len(checker_dirs) >= max_to_test:
            break

    check_fields = [
        'Channel P2 Pressure High Error - Threshold',
        'Channel P2 Pressure High Warning - Threshold',
        'Channel P1 Pressure High Error - Threshold',
        'Channel P1 Pressure High Warning - Threshold',
    ]

    flagged_rows = []

    for chk_dir in checker_dirs:
        # find the first CSV in that FinalChecker dir
        csv_files = [f for f in os.listdir(chk_dir) if f.lower().endswith('.csv')]
        if not csv_files:
            continue

        csv_path = os.path.join(chk_dir, csv_files[0])
        try:
            meta, thr = extract_metadata_and_thresholds(csv_path)
        except KeyError:
            continue  # skip parse errors

        # only keep units where *all* 4 thresholds were found
        if all(field in thr for field in check_fields):
            row = {
                'Serial Number':    meta['Serial Number'],
                'Model Name':       meta['Model Name'],
                'Build Date':       meta['Build Date'],
                'Firmware Version': meta['Firmware Version'],
            }
            # add each threshold value, no filtering by value
            for fld in check_fields:
                row[fld] = thr[fld]
            flagged_rows.append(row)

    cols = [
        'Serial Number',
        'Model Name',
        'Build Date',
        'Firmware Version',
    ] + check_fields

    flagged_df = pd.DataFrame(flagged_rows, columns=cols)

    if output_csv and not flagged_df.empty:
        flagged_df.to_csv(output_csv, index=False, mode='a', header=False)

    return flagged_df


if __name__ == '__main__':
    base_dir = r'\\shr-eng-srv\Prod-Data-Copy\DATA'
    flagged_df = find_flagged_final_checkers(
        base_dir,
        max_to_test=50,    # smaller for debug
        batch_size=50,
        output_csv=None
    )


    print("\n=== Flagged Units ===")
    # print(flagged_df.to_string(index=False))
