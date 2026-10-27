#!/usr/bin/env python3
"""Figures for the 10/07 weekly report: length sweep, and taxa-subset slope vs reference GC."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = ".docs/latex/figs"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a"]           # validated categorical slots 1-3 (light)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

# --- length sweep
d = pd.read_csv("results/length_sweep_phylum.csv")
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.axhline(0.5, color=MUTED, lw=1, ls=":")
ax.text(d.length.min(), 0.505, "chance", color=MUTED, fontsize=8, va="bottom")
for col, c, m in zip(["novel genus", "novel family", "non-bacterial"], C, ["o", "s", "^"]):
    ax.plot(d.length, d[col], color=c, lw=2, marker=m, ms=6, mec="white", mew=1)
    ax.text(d.length.iloc[-1] * 1.08, d[col].iloc[-1], col, color=INK, fontsize=9, va="center")
ax.set_xscale("log"); ax.set_xticks(d.length); ax.set_xticklabels(d.length)
ax.set_xlabel("Window length (bp)"); ax.set_ylabel("AUROC vs ID")
ax.set_xlim(d.length.min() * 0.85, d.length.max() * 2.6)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/length_sweep.pdf"); fig.savefig(f"{OUT}/length_sweep.png", dpi=150)

# --- taxa subsets: slope vs reference GC (NT only)
t = pd.read_csv("results/taxa_subset_sweep_nt.csv")
s = t[t.subset != "ALL"]; a = t[t.subset == "ALL"].iloc[0]
r, p = stats.pearsonr(s.ref_gc, s.slope)
fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.axhline(0, color=MUTED, lw=1)
sig = s.p < 0.05
ax.scatter(s.ref_gc[sig], s.slope[sig], s=46, color=C[0], edgecolor="white", lw=1, zorder=3, label="5-phylum subset, p < 0.05")
ax.scatter(s.ref_gc[~sig], s.slope[~sig], s=46, facecolor="white", edgecolor=C[0], lw=1.5, zorder=3, label="5-phylum subset, p ≥ 0.05")
ax.scatter([a.ref_gc], [a.slope], s=70, marker="D", color=C[1], edgecolor="white", lw=1, zorder=4, label="all 28 eligible phyla")
b = np.polyfit(s.ref_gc, s.slope, 1); xs = np.linspace(s.ref_gc.min(), s.ref_gc.max(), 50)
ax.plot(xs, np.polyval(b, xs), color=MUTED, lw=1, ls="--", zorder=2)
ax.set_xlabel("Reference GC content"); ax.set_ylabel("Slope of flagged fraction vs up")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/taxa_subset_gc.pdf"); fig.savefig(f"{OUT}/taxa_subset_gc.png", dpi=150)
print(f"r = {r:.3f}, p = {p:.4f}; ALL phyla: refGC {a.ref_gc:.4f} slope {a.slope:+.4f} n_phyla {int(a.n_phyla)}")

# --- taxa subsets: the 15 slopes themselves (the result the PI asked about)
t = pd.read_csv("results/taxa_subset_sweep_nt.csv")
s = t[t.subset != "ALL"].sort_values("slope").reset_index(drop=True); a = t[t.subset == "ALL"].iloc[0]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.axhline(0, color=MUTED, lw=1)
x = np.arange(1, len(s) + 1); sig = (s.p < 0.05).values
ax.scatter(x[sig], s.slope[sig], s=46, color=C[0], edgecolor="white", lw=1, zorder=3, label="p < 0.05")
ax.scatter(x[~sig], s.slope[~sig], s=46, facecolor="white", edgecolor=C[0], lw=1.5, zorder=3, label="p ≥ 0.05")
ax.axhline(a.slope, color=C[1], lw=1.5, ls="--", zorder=2)
ax.text(len(s) + 0.4, a.slope, "all 28\nphyla", color=INK, fontsize=8, va="center")
ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x], fontsize=8)
ax.set_xlabel("Random 5-phylum reference (sorted by slope)"); ax.set_ylabel("Slope of flagged fraction vs up")
ax.set_xlim(0.3, len(s) + 2.2); ax.set_ylim(-0.001, s.slope.max() * 1.12)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/taxa_subset_slopes.pdf"); fig.savefig(f"{OUT}/taxa_subset_slopes.png", dpi=150)
