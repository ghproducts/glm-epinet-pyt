#!/usr/bin/env python3
"""Cross-backbone summary of the clean1500 grid (after aggregate_seeds.py for each backbone).
Writes results/clean1500/summary_{calibration,detection,darkmatter}.csv and prints compact tables."""
import pandas as pd
BB = ["NT_transformer", "DNABERT2", "hyenaDNA", "CARMANIA"]
suf = lambda b: "" if b == "NT_transformer" else f"_{b}"
cal = pd.concat([pd.read_csv(f"results/clean1500/seeds_calibration{suf(b)}.csv").assign(backbone=b) for b in BB])
det = pd.concat([pd.read_csv(f"results/clean1500/seeds_detection{suf(b)}.csv").assign(backbone=b) for b in BB])
sw = pd.concat([pd.read_csv(f"results/clean1500/seeds_darkmatter_fpr_sweep{suf(b)}.csv").assign(backbone=b) for b in BB])
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
c = cal.groupby(["backbone", "rank", "set", "method"])[["accuracy", "ece"]].mean().reset_index()
c.to_csv("results/clean1500/summary_calibration.csv", index=False)
print("=== id_test accuracy / ECE (mean of 5 seeds)")
x = c[c.set == "id_test"].pivot_table(index=["rank", "backbone"], columns="method", values=["accuracy", "ece"]).round(3)
print(x.to_string())
print("\n=== ECE on in-vocab novel tiers (phylum: novel genus/family; family: novel genus)")
y = c[c.set.isin(["novel_genus", "novel_family"])].pivot_table(index=["rank", "set", "backbone"], columns="method", values="ece").round(3)
print(y.to_string())
d = det.copy(); d["error"] = d.error.astype(str).replace({"0": "0%", "_err0.5": "0.5%", "_err5": "5%"})
dm = d.groupby(["backbone", "rank", "error", "tier", "score"]).auroc.agg(["mean", "std"]).reset_index()
dm.to_csv("results/clean1500/summary_detection.csv", index=False)
SC = ["msp", "energy", "entropy", "mc_epistemic", "epi_epistemic", "maha", "knn"]
for r in ["family", "phylum"]:
    print(f"\n=== detection AUROC, {r} models, no injected error (mean of 5 seeds; sd <= shown max)")
    z = dm[(dm["rank"] == r) & (dm.error == "0%") & dm.score.isin(SC)]
    print(z.pivot_table(index=["tier", "backbone"], columns="score", values="mean")[SC].round(3).to_string())
    print(f"  max sd: {z['std'].max():.3f}")
print("\n=== error sweep, family models: AUROC at 0 / 0.5 / 5% error (kNN and best UQ score)")
z = dm[(dm["rank"] == "family") & dm.score.isin(["knn", "maha", "mc_epistemic", "epi_epistemic", "msp"])]
print(z.pivot_table(index=["tier", "backbone", "score"], columns="error", values="mean")[["0%", "0.5%", "5%"]].round(3).to_string())
g = sw.groupby(["backbone", "rank", "score", "fpr"]).agg(f25=("f25", "mean"), f75=("f75", "mean"), f100=("f100", "mean"),
      slope=("slope", "mean"), n_sig=("p", lambda v: (v < 0.05).sum()), n_pos=("slope", lambda v: (v > 0).sum())).reset_index()
g["mono"] = sw.assign(m=(sw.f25 < sw.f75) & (sw.f75 < sw.f100)).groupby(["backbone", "rank", "score", "fpr"]).m.sum().values
g.to_csv("results/clean1500/summary_darkmatter.csv", index=False)
print("\n=== dark matter: seeds (of 5) with significant positive slope / strictly increasing, at 1% | 5% | 10% FPR")
h = g[g.fpr.isin([0.01, 0.05, 0.10])].copy()
h["cell"] = h.apply(lambda r: f"{int(r.n_sig) if r.slope > 0 else -int(r.n_sig)}/{int(r.mono)}", axis=1)
print(h.pivot_table(index=["rank", "backbone", "score"], columns="fpr", values="cell", aggfunc="first").to_string())
print("\n=== dark matter flagged fraction at 5% FPR, UP 0.25 / 0.75 / 1.0 (mean of seeds)")
k = g[g.fpr == 0.05].copy(); k["f"] = k.apply(lambda r: f"{r.f25:.3f}/{r.f75:.3f}/{r.f100:.3f}", axis=1)
print(k.pivot_table(index=["rank", "backbone"], columns="score", values="f", aggfunc="first").to_string())
