#!/usr/bin/env python3
"""Figure for the 10/13 weekly report: dark matter flagged fraction vs UP for the
early-stopped models (seed 1), kNN and Mahalanobis at 5% FPR.
usage: plot_report_1013.py [family|phylum]   (default family)"""
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R, OUT, FPR = "results/clean1500_es", ".docs/latex/figs", 0.05
RANK = sys.argv[1] if len(sys.argv) > 1 else "family"
BB = [("NT_transformer", "NT", "#2a78d6"), ("DNABERT2", "DNABERT2", "#eb6834"),
      ("hyenaDNA", "hyenaDNA", "#1baf7a"), ("CARMANIA", "CARMANIA", "#8e5bd0")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
for ax, (sc, title) in zip(axes, [("knn", "$k$-NN ($k{=}1$)"), ("maha", "Mahalanobis")]):
    ends = []
    for i, (b, name, col) in enumerate(BB):
        d = f"{R}/{b}_{RANK}_s1"
        dm = pd.read_parquet(f"{d}/scores_darkmatter.parquet"); idt = pd.read_parquet(f"{d}/scores_id_test.parquet")
        ps = dm.assign(fl=dm[sc] > np.quantile(idt[sc], 1 - FPR)).groupby(["up", "sample"]).fl.mean().reset_index()
        x = ps.up + (i - 1.5) * 0.018
        ax.scatter(x, ps.fl, s=14, color=col, alpha=0.45, lw=0)
        m = ps.groupby("up").fl.mean()
        ax.plot(m.index + (i - 1.5) * 0.018, m.values, color=col, lw=2, marker="o", ms=5, mec="white", mew=1)
        ends.append([m.values[-1], name, col])
    ends.sort(); y = [e[0] for e in ends]
    for k in range(1, len(y)):                      # keep direct labels from overlapping
        y[k] = max(y[k], y[k - 1] + 0.014)
    for (v, name, col), yy in zip(ends, y):
        ax.text(1.05, yy, name, color=INK, fontsize=8.5, va="center")
        ax.plot([1.03, 1.045], [v, yy], color=col, lw=1)
    ax.axhline(FPR, color=MUTED, lw=1, ls="--"); ax.text(0.5, FPR - 0.009, "5% FPR (id_test)", color=MUTED, fontsize=8, ha="center")
    ax.set_title(title, fontsize=10, color=INK); ax.set_xlabel("Unknown percentage (UP)")
    ax.set_xticks([0.25, 0.75, 1.0]); ax.set_xlim(0.18, 1.2)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("Fraction of windows flagged")
fig.tight_layout(); fig.savefig(f"{OUT}/darkmatter_es_{RANK}.pdf"); fig.savefig(f"{OUT}/darkmatter_es_{RANK}.png", dpi=150)
print("wrote figure")
