import os
import re
import pandas as pd
from sklearn.linear_model import LinearRegression

# ─ User configuration ────────────────────────────────────────────────────────
drift_folder    = r"D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\2nd soak"
drift_filename  = "Pressure_Drift_Data_2025-07-30.xlsx"
output_folder   = r"D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\2nd soak\corrected_ROR"

hf_hours = 564
# ───────────────────────────────────────────────────────────────────────────────

def compute_drift_coeffs():
    """Read the Excel, filter by HF_Soak_Hours, and return dict of
       DUT → DataFrame(index=['P1','P2'], columns=['slope','intercept'])."""
    filepath = os.path.join(drift_folder, drift_filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No file found at {filepath!r}")

    xls = pd.ExcelFile(filepath)
    results = {}

    for dut in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=dut)

        # filter by soak‑hours
        if 'HF_Soak_Hours' not in df.columns:
            print(f"⚠️ Sheet '{dut}' missing HF_Soak_Hours → skipping")
            continue
        sub = df[df['HF_Soak_Hours'] == hf_hours]
        if sub.empty:
            print(f"⚠️ No rows in '{dut}' for HF_Soak_Hours={hf_hours} → skipping")
            continue

        regs = {}
        for i in (1, 2):
            drift_col = f'Pressure {i} Drift (torr)'
            # find any matching “mean” column
            mean_cols = [c for c in sub.columns if f"Pressure {i} Reading [REAL]_mean" in c]
            if drift_col not in sub.columns or not mean_cols:
                print(f"⚠️ '{dut}' missing '{drift_col}' or P{i} mean → skipping P{i}")
                continue

            mean_col = mean_cols[0]
            X = sub[[mean_col]].values   # reading on X
            y = sub[drift_col].values    # drift on Y

            model = LinearRegression().fit(X, y)
            regs[f'P{i}'] = {
                'slope':     float(model.coef_[0]),
                'intercept': float(model.intercept_)
            }

        if regs:
            results[dut.upper()] = pd.DataFrame.from_dict(regs, orient='index')

    return results


def apply_drift_to_ror(results):
    """For each ROR CSV in drift_folder, extract DUT#, adjust readings and save."""
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(drift_folder):
        if not fname.lower().endswith('.csv') or 'ror' not in fname.lower():
            continue

        # extract DUT identifier, e.g. 'DUT16'
        m = re.search(r'(DUT\d+)', fname, re.IGNORECASE)
        if not m:
            print(f"⚠️ Could not find DUT in '{fname}' → skipping")
            continue
        dut = m.group(1).upper()

        if dut not in results:
            print(f"⚠️ No drift coeffs for {dut}, skipping '{fname}'")
            continue

        # load the ROR file
        ror_df = pd.read_csv(os.path.join(drift_folder, fname))

        # pull out slopes/intercepts
        regs = results[dut]
        slope1, int1 = regs.at['P1','slope'], regs.at['P1','intercept']
        slope2, int2 = regs.at['P2','slope'], regs.at['P2','intercept']

        # find the reading columns by substring
        p1_cols = [c for c in ror_df.columns if 'Pressure 1 Reading' in c]
        p2_cols = [c for c in ror_df.columns if 'Pressure 2 Reading' in c]

        if not p1_cols or not p2_cols:
            print(f"⚠️ '{fname}' missing P1 or P2 Reading cols → skipping")
            continue

        # apply correction: new = orig - (slope*orig + intercept)
        for c in p1_cols:
            ror_df[c] = ror_df[c] - (slope1 * ror_df[c] + int1)
        for c in p2_cols:
            ror_df[c] = ror_df[c] - (slope2 * ror_df[c] + int2)

        # save out
        out_path = os.path.join(output_folder, fname)
        ror_df.to_csv(out_path, index=False)
        print(f"✅ Corrected '{fname}' → '{out_path}'")


def main():
    # 1) compute drift regressions
    results = compute_drift_coeffs()

    print("\nDrift coefficients by DUT:")
    for dut, df_regs in results.items():
        print(f"\n{dut}")
        print(df_regs)

    # 2) apply to all ROR CSVs
    apply_drift_to_ror(results)


if __name__ == "__main__":
    main()
