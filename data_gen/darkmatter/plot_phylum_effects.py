#!/usr/bin/env python3
"""Per-phylum influence on the dark matter slope, from 500 random 5-phylum references.

Effect = OLS coefficient of slope on phylum presence (each reference holds 5 of
28 phyla), centred to mean zero, so it reads as "including this phylum shifts
the slope by this much relative to an average phylum". 95% bootstrap CIs over
references. A phylum that flattens the slope makes high-UP samples look more
familiar, i.e. the unknown species resemble it more than the known ones do.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
d = pd.read_csv("results/taxa_subset_phylum_effects.csv").sort_values("effect").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.4, 7.4))
ax.axvline(0, color=MUTED, lw=1)
y = np.arange(len(d)); sig = (d.lo > 0) | (d.hi < 0)
ax.hlines(y, d.lo, d.hi, color=BLUE, lw=2)
ax.scatter(d.effect[sig], y[sig], s=46, color=BLUE, edgecolor="white", lw=1, zorder=3, label="95% interval excludes 0")
ax.scatter(d.effect[~sig], y[~sig], s=46, facecolor="white", edgecolor=BLUE, lw=1.5, zorder=3, label="95% interval includes 0")
ax.set_yticks(y); ax.set_yticklabels(d.name, fontsize=8.5)
ax.set_xlabel("Change in slope when the phylum is in the reference")
lim = max(abs(d.lo.min()), abs(d.hi.max())) * 1.1; ax.set_xlim(-lim, lim)
kw = dict(transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
ax.text(0.02, 1.01, "← unknown species\n    resemble it more", ha="left", **kw)
ax.text(0.98, 1.01, "known species →\nresemble it more", ha="right", **kw)
ax.legend(loc="lower right", fontsize=8, frameon=False)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(".docs/latex/figs/taxa_subset_phylum_effects.pdf"); fig.savefig(".docs/latex/figs/taxa_subset_phylum_effects.png", dpi=150)
print("wrote figure")
