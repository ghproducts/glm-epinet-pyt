#!/usr/bin/env python3
"""2x3 grid: U_total (top row) and max_confidence (bottom row) vs `up`,
one column per method (base / mc_dropout / epinet).

Each point is one input FASTQ file (up x sample x read_type x mate), so each
panel carries 61 points. Per-panel y-limits: the three methods sit at different
absolute levels, and forcing a shared axis would flatten the within-method
trend that the figure exists to show. The dashed line is a least-squares fit
over the plotted points; the annotation gives the fit over
the plotted points. Slopes and p-values live in results/darkmatter_gc_analysis/
rather than on the panels -- up=0.5 is plotted but excluded from those statistics
(single file, no replicates).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

METHODS = ["base", "mc_dropout", "epinet"]
COLORS = {"base": "tab:orange", "epinet": "tab:purple", "mc_dropout": "tab:green"}
MARKERS = {"base": "s", "epinet": "o", "mc_dropout": "^"}
ROWS = ["U_total", "max_confidence"]


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent.parent
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "darkmatter_nt_phylum")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="Phylum-rank")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10.3, 6.9))
    for j, m in enumerate(METHODS):
        d = pd.read_csv(args.results_dir / m / "inference_uncertainty.csv")
        for i, metric in enumerate(ROWS):
            ax = axes[i, j]
            pf = d.groupby(["up", "sample", "read_type", "mate"])[metric].mean().reset_index()
            jit = (pf.groupby("up").cumcount() - 9.5) * 0.55
            ax.scatter(pf.up * 100 + jit, pf[metric], s=18, alpha=.7,
                       color=COLORS[m], marker=MARKERS[m], zorder=3,
                       linewidths=0.8, edgecolors="none")
            coef = np.polyfit(pf.up * 100, pf[metric], 1)
            xs = np.linspace(15, 110, 30)
            ax.plot(xs, np.polyval(coef, xs), color=COLORS[m], ls="--", lw=2.2, zorder=2)

            ax.set_box_aspect(1)          # square panels
            ax.set_xticks([25, 50, 75, 100]); ax.set_xlim(15, 110)
            ax.grid(alpha=.3, ls=":", lw=1.0); ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_linewidth(1.3)
            ax.tick_params(width=1.3)
            if i == 0:
                ax.set_title(m, fontsize=11, color=COLORS[m], fontweight="bold")
            if i == 1:
                ax.set_xlabel("Unknown percentage (`up`)")
            else:
                # both rows share the same x meaning; square panels leave no
                # room for two sets of tick labels without them colliding
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(metric, fontsize=10)

    fig.suptitle(f"Darkmatter {args.label}. U_total and max_confidence vs unknown fraction.",
                 fontsize=11.5, y=.975)
    fig.tight_layout(rect=(0, 0, 1, .955))
    fig.subplots_adjust(hspace=.07, wspace=.34)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140 if args.out.suffix == ".png" else None)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
