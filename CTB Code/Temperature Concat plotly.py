# -*- coding: utf-8 -*-
"""
Temperature plots: Matplotlib (4 DUTs/fig) + Plotly HTML (1 DUT per plot, inline JS)
Now uses TWO separate Plotly figures per DUT stacked vertically:
  1) Temperature Trace
  2) % Difference = 100 * (PT3 - DUT) / DUT
Phase lines + labels render on each figure independently.

Created on Fri Aug  8 12:49:41 2025
@author: SFerneyhough
"""

import os
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH = r"D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\Temperature Concat.csv"
OUT_DIR  = r"D:\Desktop\Corrosion Test Bench Data\Test 2 June 2025\plots"

# ── Load + sanitize ──────────────────────────────────────────────────────────
order = ["Baseline", "Post Passivation", "Soak 1", "Soak 2"]

df = pd.read_csv(CSV_PATH)

# Trim column names and normalize expected headers
df.columns = [c.strip() for c in df.columns]

# Ensure expected columns exist (rename here if your CSV uses variants)
required_cols = ["PT3 Temperature", "PT4 Temperature", "DUT Temperature Reading", "DUT", "Filename"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}\nFound: {df.columns.tolist()}")

# Coerce numerics (bad strings -> NaN)
for c in ["PT3 Temperature", "PT4 Temperature", "DUT Temperature Reading", "DUT"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows without a DUT and cast to int
df = df.dropna(subset=["DUT"]).copy()
df["DUT"] = df["DUT"].astype(int)

# Clean Filename values and force ordered categorical
df["Filename"] = df["Filename"].astype(str).str.strip()
df["Filename"] = pd.Categorical(df["Filename"], categories=order, ordered=True)

# Preserve within-phase sequence per DUT, then globally sort
df["_orig_order"]   = np.arange(len(df))
df["_row_in_file"]  = (
    df.sort_values("_orig_order")
      .groupby(["DUT", "Filename"], observed=True)["_orig_order"]
      .rank(method="first").astype(int) - 1
)
df = df.sort_values(["DUT", "Filename", "_row_in_file"]).reset_index(drop=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def _phase_counts_in_order(sub: pd.DataFrame):
    """Return (labels, counts) for phases present in 'sub' in categorical order."""
    sizes = (sub.groupby("Filename", observed=True).size()
                .reindex(sub["Filename"].cat.categories, fill_value=0))
    sizes = sizes[sizes > 0]  # keep only phases present in this DUT slice
    return sizes.index.tolist(), sizes.values

# ── Matplotlib: 4 DUTs per figure (unchanged) ────────────────────────────────
def plot_temps_grouped(df: pd.DataFrame, title_prefix="Temperature Trace"):
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    all_duts = sorted(df["DUT"].unique())
    dut_groups = [all_duts[i:i+4] for i in range(0, len(all_duts), 4)]

    for group_idx, duts in enumerate(dut_groups, start=1):
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        axes = axes.flatten()

        for ax, dut in zip(axes, duts):
            sub = (df[df["DUT"] == dut]
                    .sort_values(["Filename", "_row_in_file"])
                    .reset_index(drop=True)).copy()
            if sub.empty:
                ax.axis("off")
                continue

            x = np.arange(len(sub))
            phase_labels, counts = _phase_counts_in_order(sub)
            boundaries = np.cumsum(counts)[:-1]

            ax.plot(x, sub["PT3 Temperature"], label="PT3")
            ax.plot(x, sub["PT4 Temperature"], label="PT4")
            ax.plot(x, sub["DUT Temperature Reading"], label="DUT Temp")

            ax.set_title(f"DUT {dut}")
            ax.set_ylabel("Temperature (°C)")
            ax.grid(True)

            for b in boundaries:
                ax.axvline(b, linestyle="--", alpha=0.7)

            ymin, ymax = ax.get_ylim()
            left = 0
            for lab, n in zip(phase_labels, counts):
                center = left + n / 2
                ax.text(center, ymax, lab, ha="center", va="bottom", fontsize=8)
                left += n

            ax.legend()

        for ax in axes[len(duts):]:
            ax.axis("off")

        fig.suptitle(f"{title_prefix} — DUTs {', '.join(map(str, duts))}",
                     fontsize=16, weight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

# ── Plotly: two separate figures per DUT ─────────────────────────────────────
def _make_temp_fig(sub: pd.DataFrame, dut: int) -> go.Figure:
    sub = sub.copy()
    for c in ["PT3 Temperature", "PT4 Temperature", "DUT Temperature Reading"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    x = np.arange(len(sub))
    phase_labels, counts = _phase_counts_in_order(sub)
    boundaries = np.cumsum(counts)[:-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sub["PT3 Temperature"], mode="lines", name="PT3"))
    fig.add_trace(go.Scatter(x=x, y=sub["PT4 Temperature"], mode="lines", name="PT4"))
    fig.add_trace(go.Scatter(x=x, y=sub["DUT Temperature Reading"], mode="lines", name="DUT Temp"))

    shapes = [dict(type="line", x0=b, x1=b, y0=0, y1=1, xref="x", yref="paper",
                   line=dict(dash="dash", width=1)) for b in boundaries]

    annotations = []
    left = 0
    for lab, n in zip(phase_labels, counts):
        center = left + n/2
        annotations.append(dict(x=center, xref="x", y=1.06, yref="paper",
                                text=str(lab), showarrow=False,
                                font=dict(size=12, color="#333")))
        left += n

    fig.update_layout(
        title=dict(
            text=f"Temperature Trace — DUT {dut}",
            x=0.5,               # center title
            xanchor="center"
        ),
        xaxis_title="Sample Index (ordered by phase)",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,              # lower so it’s under title
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=70, r=20, t=80, b=60),
        shapes=shapes,
        annotations=annotations,
        height=420,
    )
    return fig


def _make_pctdiff_fig(sub: pd.DataFrame, dut: int) -> go.Figure:
    sub = sub.copy()
    for c in ["PT3 Temperature", "PT4 Temperature", "DUT Temperature Reading"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    denom = sub["DUT Temperature Reading"].replace(0, np.nan)
    pct_diff = 100.0 * (sub["PT3 Temperature"] - sub["DUT Temperature Reading"]) / denom

    x = np.arange(len(sub))
    phase_labels, counts = _phase_counts_in_order(sub)
    boundaries = np.cumsum(counts)[:-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pct_diff, mode="lines", name="% Diff (PT3 vs DUT)"))

    # 0% reference
    fig.add_hline(y=0, line_width=1, line_dash="dot")

    shapes = [dict(type="line", x0=b, x1=b, y0=0, y1=1, xref="x", yref="paper",
                   line=dict(dash="dash", width=1)) for b in boundaries]

    annotations = []
    left = 0
    for lab, n in zip(phase_labels, counts):
        center = left + n/2
        annotations.append(dict(x=center, xref="x", y=1.09, yref="paper",
                                text=str(lab), showarrow=False,
                                font=dict(size=12, color="#333")))
        left += n

    fig.update_layout(
        title=dict(
            text=f"% Difference — DUT {dut}  (100·(PT3 − DUT)/DUT)",
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Sample Index (ordered by phase)",
        yaxis_title="% Difference",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=70, r=20, t=80, b=60),
        shapes=shapes,
        annotations=annotations,
        height=340,
    )
    return fig


def write_plotly_html_per_dut_inline(df: pd.DataFrame, out_dir: str,
                                     page_title="Interactive Temperature Traces") -> str:
    """
    Build a single HTML that embeds Plotly.js inline (offline) and
    includes TWO interactive charts per DUT, stacked with horizontal
    separators between DUT sections.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_duts = sorted(df["DUT"].unique())

    fragments = []
    plotly_included = False
    for idx, dut in enumerate(all_duts):
        sub = (df[df["DUT"] == dut]
                .sort_values(["Filename", "_row_in_file"])
                .reset_index(drop=True)).copy()
        if sub.empty:
            continue

        fig_top = _make_temp_fig(sub, dut)
        fig_bot = _make_pctdiff_fig(sub, dut)

        if not plotly_included:
            frag_top = pio.to_html(fig_top, full_html=False, include_plotlyjs="inline")
            frag_bot = pio.to_html(fig_bot, full_html=False, include_plotlyjs=False)
            plotly_included = True
        else:
            frag_top = pio.to_html(fig_top, full_html=False, include_plotlyjs=False)
            frag_bot = pio.to_html(fig_bot, full_html=False, include_plotlyjs=False)

        # Add the two plots for this DUT
        fragments.append(f'<div class="plotwrap">{frag_top}</div>')
        fragments.append(f'<div class="plotwrap">{frag_bot}</div>')

        # Add horizontal separator after each DUT except the last
        if idx < len(all_duts) - 1:
            fragments.append('<hr style="border:1px solid #ccc; margin:40px 0;">')

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_path = os.path.join(out_dir, "temperature_traces_interactive.html")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{page_title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; margin: 16px; }}
  h1 {{ font-size: 22px; margin-bottom: 6px; }}
  .meta {{ color:#555; margin-bottom: 18px; }}
  .plotwrap {{ margin-bottom: 28px; }}
</style>
</head>
<body>
  <h1>{page_title}</h1>
  <div class="meta">Generated: {ts}</div>
  {''.join(fragments)}
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


# ── Run both ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Matplotlib: 4 DUTs per figure (onscreen)
    # plot_temps_grouped(df, title_prefix="Temperature Trace")

    # 2) Plotly HTML: TWO stacked charts per DUT (self-contained) + open
    html_file = write_plotly_html_per_dut_inline(
        df, OUT_DIR,
        page_title="Temperature Traces + % Difference (PT3 vs DUT) — One DUT per section"
    )
    webbrowser.open(f"file:///{html_file}")
