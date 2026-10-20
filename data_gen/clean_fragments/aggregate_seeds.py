#!/usr/bin/env python3
"""Aggregate one backbone's clean1500 runs (--backbone, default NT_transformer) across seeds: dark matter FPR sweep (kNN, Mahalanobis),
OOD detection AUROC, and ID calibration. Writes results/clean1500/seeds_*.csv."""
import argparse, glob, re
import numpy as np, pandas as pd
from scipy import stats
ap = argparse.ArgumentParser(); ap.add_argument("--base-only", action="store_true",
    help="read <run>/base_only (kNN/Mahalanobis/softmax only); detection computed here"); ap.add_argument("--backbone", default="NT_transformer"); A = ap.parse_args()
SUB = "base_only/" if A.base_only else ""
SCORES = ["knn", "maha", "msp", "energy"] + ([] if A.base_only else ["mc_epistemic", "epi_epistemic"])
TAG = ("_baseonly" if A.base_only else "") + ("" if A.backbone == "NT_transformer" else f"_{A.backbone}")
FPRS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
sweep, det, cal = [], [], []
for d in sorted(glob.glob(f"results/clean1500/{A.backbone}_*_s*")):
    r, s = re.search(r"_(phylum|family)_s(\d+)$", d).groups(); s = int(s)
    try:
        dm = pd.read_parquet(f"{d}/{SUB}scores_darkmatter.parquet"); idt = pd.read_parquet(f"{d}/{SUB}scores_id_test.parquet")
    except FileNotFoundError:
        print("skip (incomplete)", d); continue
    for sc in SCORES:
        for f in FPRS:
            T = np.quantile(idt[sc], 1 - f)
            ps = dm.assign(fl=dm[sc] > T).groupby(["up", "sample"]).fl.mean().reset_index()
            sl, _, _, p, _ = stats.linregress(ps.up, ps.fl); m = ps.groupby("up").fl.mean()
            sweep.append(dict(rank=r, seed=s, score=sc, fpr=f, f25=m[0.25], f75=m[0.75], f100=m[1.0], slope=sl, p=p))
    if A.base_only:
        from sklearn.metrics import roc_auc_score
        for tier in ["novel_genus", "novel_family", "nonbacterial"]:
            o = pd.read_parquet(f"{d}/{SUB}scores_{tier}.parquet")
            for sc in SCORES:
                det.append(pd.DataFrame([dict(error="0", tier=tier, score=sc, rank=r, seed=s,
                    auroc=roc_auc_score(np.r_[np.zeros(len(idt)), np.ones(len(o))], np.r_[idt[sc], o[sc]]))]))
    else:
        det.append(pd.read_csv(f"{d}/detection.csv").assign(rank=r, seed=s))
        cal.append(pd.read_csv(f"{d}/calibration.csv").assign(rank=r, seed=s))
sw = pd.DataFrame(sweep); det = pd.concat(det); cal = pd.concat(cal) if cal else None
sw.to_csv(f"results/clean1500/seeds_darkmatter_fpr_sweep{TAG}.csv", index=False)
det.to_csv(f"results/clean1500/seeds_detection{TAG}.csv", index=False)
if cal is not None: cal.to_csv(f"results/clean1500/seeds_calibration{TAG}.csv", index=False)
pd.set_option("display.width", 220)
print("seeds per rank:", sw.groupby("rank").seed.unique().to_dict())
for sc in [x for x in SCORES if x != "energy"]:
    g = sw[sw.score == sc].groupby(["rank", "fpr"])
    t = g.agg(f25=("f25", "mean"), f75=("f75", "mean"), f100=("f100", "mean"), slope=("slope", "mean"), slope_sd=("slope", "std"),
              n_pos=("slope", lambda x: (x > 0).sum()), n_sig=("p", lambda x: (x < 0.05).sum()), n=("p", "size"),
              mono=("f75", "size"))
    t["mono"] = g[["f25", "f75", "f100"]].apply(lambda x: ((x.f25 < x.f75) & (x.f75 < x.f100)).sum())
    print(f"\n=== dark matter, {sc} (mean over seeds; n_sig = seeds with p<0.05; mono = seeds with 0.25<0.75<1.0)"); print(t.round(4).to_string())
d0 = det[det.error.astype(str) == "0"]
print("\n=== detection AUROC (no injected error), mean ± sd over seeds")
a = d0.groupby(["rank", "tier", "score"]).auroc.agg(["mean", "std", "size"]).round(3)
print(a.unstack("tier").to_string())
if cal is not None:
    print("\n=== calibration on id_test")
    print(cal[cal.set == "id_test"].groupby(["rank", "method"])[["accuracy", "ece"]].agg(["mean", "std"]).round(3).to_string())
