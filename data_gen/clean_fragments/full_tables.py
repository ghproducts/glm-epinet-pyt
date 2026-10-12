#!/usr/bin/env python3
"""Full clean1500 result tables, every backbone x rank x method x set, mean and sd over seeds.

Writes results/clean1500/full_{calibration,detection,darkmatter}.csv (+ _per_seed versions)
and results/clean1500/full_tables.md (markdown for reading).
  calibration: accuracy and ECE on the in-vocabulary part of every set where at least half the
               fragments have a label the model knows (id_test; novel_genus for both ranks;
               novel_family for phylum models), at 0 / 0.5 / 5% injected error.
  detection:   AUROC and FPR@95%TPR, id_test vs each tier at the SAME error level, all 9 scores.
  darkmatter:  flagged fraction per UP at each FPR (threshold = id_test quantile), per-sample
               slope vs UP and its p (15 samples), AUROC id_test vs each UP level, all 9 scores.
"""
import argparse, glob, re
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results/clean1500"); A = ap.parse_args()
SC = ["msp", "energy", "entropy", "mc_entropy", "mc_epistemic", "epi_entropy", "epi_epistemic", "maha", "knn"]
FPRS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
cal, det, dmr = [], [], []
for d in sorted(glob.glob(f"{A.root}/*_s[0-9]*")):
    m = re.search(r"/(\w+?)_(phylum|family)_s(\d+)$", d)
    if not m or not glob.glob(f"{d}/detection.csv"): continue
    b, r, s = m.group(1), m.group(2), int(m.group(3)); key = dict(backbone=b, rank=r, seed=s)
    c = pd.read_csv(f"{d}/calibration.csv")
    c["error"] = c.set.str.extract(r"_err([\d.]+)$")[0].fillna("0"); c["set"] = c.set.str.replace(r"_err[\d.]+$", "", regex=True)
    cal.append(c.assign(**key))
    det.append(pd.read_csv(f"{d}/detection.csv").assign(**key))
    dm = pd.read_parquet(f"{d}/scores_darkmatter.parquet"); idt = pd.read_parquet(f"{d}/scores_id_test.parquet")
    for sc in SC:
        auc = {f"auroc_up{u}": roc_auc_score(np.r_[np.zeros(len(idt)), np.ones((dm.up == u).sum())], np.r_[idt[sc], dm[sc][dm.up == u]])
               for u in (0.25, 0.75, 1.0)}
        for f in FPRS:
            ps = dm.assign(fl=dm[sc] > np.quantile(idt[sc], 1 - f)).groupby(["up", "sample"]).fl.mean().reset_index()
            sl, _, _, p, _ = stats.linregress(ps.up, ps.fl); mu = ps.groupby("up").fl.mean()
            dmr.append(dict(**key, score=sc, fpr=f, flag_up0_25=mu[0.25], flag_up0_75=mu[0.75], flag_up1_0=mu[1.0],
                            slope=sl, p=p, increasing=bool(mu[0.25] < mu[0.75] < mu[1.0]), **auc))
cal, det, dmr = pd.concat(cal), pd.concat(det), pd.DataFrame(dmr)
det["error"] = det.error.astype(str).str.replace("_err", "")
for name, x in [("calibration", cal), ("detection", det), ("darkmatter", dmr)]:
    x.to_csv(f"{A.root}/full_{name}_per_seed.csv", index=False)
g = lambda x, keys, vals: x.groupby(keys)[vals].agg(["mean", "std"]).pipe(
    lambda t: t.set_axis([f"{v}_{s}" for v, s in t.columns], axis=1)).reset_index()
C = g(cal, ["backbone", "rank", "set", "error", "method"], ["accuracy", "ece"]).merge(
    cal.groupby(["backbone", "rank", "set", "error", "method"])[["n", "oov"]].mean().reset_index())
D = g(det, ["backbone", "rank", "error", "tier", "score"], ["auroc", "fpr95"])
M = g(dmr, ["backbone", "rank", "score", "fpr"], ["flag_up0_25", "flag_up0_75", "flag_up1_0", "slope", "auroc_up0.25", "auroc_up0.75", "auroc_up1.0"])
M = M.merge(dmr.groupby(["backbone", "rank", "score", "fpr"]).agg(
    p_median=("p", "median"), n_sig_pos=("p", lambda v: 0), n_seeds=("p", "size")).reset_index())
M["n_sig_pos"] = dmr.assign(z=(dmr.p < 0.05) & (dmr.slope > 0)).groupby(["backbone", "rank", "score", "fpr"]).z.sum().values
M["n_increasing"] = dmr.groupby(["backbone", "rank", "score", "fpr"]).increasing.sum().values
C.to_csv(f"{A.root}/full_calibration.csv", index=False); D.to_csv(f"{A.root}/full_detection.csv", index=False)
M.to_csv(f"{A.root}/full_darkmatter.csv", index=False)
print("seeds:", cal.groupby(["backbone", "rank"]).seed.nunique().to_dict())
