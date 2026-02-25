from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
One-plot figure with 12 bars from three CSVs (3 sources × 2 genders × 2 indices).

Updates requested:
- Put numeric labels on EVERY bar
- Reorder x-axis groups: all Male groups first, then all Female groups
  (Within each gender, keep source order: Wilson, CORA 1989–2006, CORA 1873–2025)

Reads from folder: Full_Data_Validation/
- wilson_1989_2006(.csv)
- cora_1989_2006(.csv)
- cora_1873_2025(.csv)

Each CSV must contain:
speaker_gender, masculine_index, feminine_index
speaker_gender should be M, F

Output:
- all_12_bars_reordered.png
- all_12_bars_reordered.pdf
"""

YLIMS = (-0.20, 0.45)
DATA_DIR = Path("Full_Data_Validation")

FILES = [
    ("Windsor et al. (1989–2006)", "wilson_1989_2006"),
    ("CORA (1989–2006)",          "cora_1989_2006"),
    ("CORA (1873–2025)",          "cora_1873_2025"),
]

GENDER_ORDER = ["M", "F"]
GENDER_LABELS = {"M": "Male senators", "F": "Female senators"}

def read_csv_flexible(path_no_ext: Path) -> pd.DataFrame:
    if path_no_ext.exists():
        return pd.read_csv(path_no_ext, encoding="utf-8")
    if path_no_ext.with_suffix(".csv").exists():
        return pd.read_csv(path_no_ext.with_suffix(".csv"), encoding="utf-8")
    raise FileNotFoundError(f"Could not find {path_no_ext} or {path_no_ext.with_suffix('.csv')}")

def set_nature_friendly_rcparams():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )

def format_value(v: float) -> str:

    if abs(v) < 0.01:
        return f"{v:.4f}"
    return f"{v:.3f}"

def label_all_bars(ax: plt.Axes, bars, ylims=YLIMS):
    """
    Label every bar with its value.
    Places labels just above positive bars and just below negative bars.
    """
    y0, y1 = ylims
    span = y1 - y0
    pad = 0.015 * span

    for b in bars:
        v = float(b.get_height())
        x = b.get_x() + b.get_width() / 2

        if v >= 0:
            y = v + pad
            va = "bottom"
        else:
            y = v - pad
            va = "top"

        ax.text(
            x,
            y,
            format_value(v),
            ha="center",
            va=va,
            fontsize=9,
        )

def main():
    set_nature_friendly_rcparams()

    rows = []
    for source_label, fname in FILES:
        df = read_csv_flexible(DATA_DIR / fname)

        required = {"speaker_gender", "masculine_index", "feminine_index"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{fname} is missing columns: {sorted(missing)}")

        df = df.copy()
        df["speaker_gender"] = df["speaker_gender"].astype(str).str.strip()
        df = df[df["speaker_gender"].isin(GENDER_ORDER)]
        df = df.set_index("speaker_gender").loc[GENDER_ORDER].reset_index()

        for g in GENDER_ORDER:
            r = df[df["speaker_gender"] == g].iloc[0]
            rows.append(
                {
                    "source": source_label,
                    "speaker_gender": g,
                    "gender_label": GENDER_LABELS[g],
                    "masculine_index": float(r["masculine_index"]),
                    "feminine_index": float(r["feminine_index"]),
                }
            )

    plot_df = pd.DataFrame(rows)

    group_order = []

    for source_label, _ in FILES:
        group_order.append(f"{source_label}\n{GENDER_LABELS['M']}")

    for source_label, _ in FILES:
        group_order.append(f"{source_label}\n{GENDER_LABELS['F']}")

    plot_df["group_label"] = plot_df.apply(
        lambda r: f"{r['source']}\n{r['gender_label']}", axis=1
    )
    plot_df["group_label"] = pd.Categorical(plot_df["group_label"], categories=group_order, ordered=True)
    plot_df = plot_df.sort_values("group_label").reset_index(drop=True)

    x = np.arange(len(group_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 4.8))

    masc_vals = plot_df["masculine_index"].to_numpy()
    fem_vals = plot_df["feminine_index"].to_numpy()

    bars_m = ax.bar(
        x - width / 2,
        masc_vals,
        width,
        label="Masculine language",
        color="tab:blue",
        edgecolor="black",
        linewidth=0.6,
    )
    bars_f = ax.bar(
        x + width / 2,
        fem_vals,
        width,
        label="Feminine language",
        color="tab:orange",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(group_order)
    ax.set_ylabel("Composite index (z sum)")
    ax.set_title("Feminine and masculine language by senator gender for technical validation")

    ax.set_ylim(*YLIMS)
    ax.axhline(0, linewidth=1.0, alpha=0.75)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax.legend(loc="upper left", frameon=False)

    label_all_bars(ax, bars_m, ylims=YLIMS)
    label_all_bars(ax, bars_f, ylims=YLIMS)

    plt.tight_layout()
    fig.savefig("Wilson_all_12_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig("Wilson_all_12_bars.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()

