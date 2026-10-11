"""Compare per-class results of several detection runs (e.g. A vs B vs C).

The first run is the reference; for every other run the difference
(run - reference) is reported, so you can see which abnormalities improve
or degrade after bone suppression.

Example
    python -m src.detection.compare --metric AP50 --out results/detection/compare ^
        A=results/detection/A_original.csv B=results/detection/B_suppressed.csv C=results/detection/C_fused.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="NAME=path/to/results.csv (first = reference)")
    ap.add_argument("--metric", default="AP50", choices=["AP50", "AP50_95", "precision", "recall"])
    ap.add_argument("--out", required=True, help="output prefix (writes .csv, .md, .png)")
    args = ap.parse_args()

    tables = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        tables[name] = pd.read_csv(path).set_index("class")[args.metric]
    df = pd.DataFrame(tables)
    ref = list(tables)[0]
    for name in list(tables)[1:]:
        df[f"{name}-{ref}"] = df[name] - df[ref]
    order = [c for c in df.index if c != "ALL"] + (["ALL"] if "ALL" in df.index else [])
    df = df.loc[order]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"))
    with open(out.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(f"| class | " + " | ".join(df.columns) + " |\n")
        f.write("|---|" + "---|" * len(df.columns) + "\n")
        for cls, row in df.iterrows():
            cells = [f"{v:+.3f}" if "-" in col else f"{v:.3f}" for col, v in row.items()]
            f.write(f"| {cls} | " + " | ".join(cells) + " |\n")
    print(df.round(3).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot = df[list(tables)].drop(index="ALL", errors="ignore")
        ax = plot.plot.barh(figsize=(9, 0.45 * len(plot) + 1.5), width=0.8)
        ax.set_xlabel(args.metric)
        ax.invert_yaxis()
        ax.set_title(f"Per-class {args.metric}")
        plt.tight_layout()
        plt.savefig(out.with_suffix(".png"), dpi=150)
    except Exception as e:  # plotting is optional
        print(f"[warn] plot skipped: {e}")
    print(f"Saved {out}.csv / .md / .png")


if __name__ == "__main__":
    main()
