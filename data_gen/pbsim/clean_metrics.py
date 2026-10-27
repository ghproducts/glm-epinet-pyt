#!/usr/bin/env python3
"""Calibration + OOD metrics from the clean-checkpoint evaluation outputs.

ECE uses 50 equal-mass bins, matching the paper (.docs/latex/UQ_plos_paper).
AUROC treats an uncertainty score as a detection statistic for ID vs novel.

A correctness check runs first, but NOT the one an earlier version of this file
assumed. `inference.py` passes encode_labels=False, and the taxa_rank remap in
load_local_dataset writes the rank's raw NCBI taxid into `labels`; `pred` is
mapped back through the checkpoint's id2label, so both sides live in TAXID
space, not 0..N-1. Counting distinct labels per tier and demanding it equal
n_classes therefore rejects perfectly valid tiers -- it wrongly excluded family/
id_novel_genus, which carries 167 of 429 families simply because a novel-genus
sample cannot span every family.

The correct test is whether a tier's labels fall inside the checkpoint's own
label set, and it is a matter of degree rather than pass/fail. A read whose true
taxid is absent from the head can never be predicted correctly, so it drags
accuracy down by construction. Measured for phylum:

  ood_novel_family    2.1% out-of-vocabulary   acc 0.778 all reads / 0.794 in-vocab
  ood_nonbacterial   97.7% out-of-vocabulary   acc 0.000 -- meaningless, AUROC only

So tiers are scored when most reads are in-vocabulary, and both figures are
reported alongside the OOV fraction; a tier that is essentially all-OOV is
skipped for accuracy/ECE.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

CKPT = {"phylum": "checkpoints/pbsim_clean/phylum/checkpoint-57734",
        "family": "checkpoints/pbsim_clean/family/checkpoint-57734"}

TIERS = ["id_test", "id_novel_genus", "ood_novel_family", "ood_nonbacterial"]
# Above this fraction of out-of-vocabulary labels a tier's accuracy is dominated
# by reads the head cannot possibly get right, so it is reported for AUROC only.
OOV_MAX = 0.5
METHODS = ["base", "base_scaled", "mc_dropout", "epinet"]


def ece_equal_mass(conf, correct, n_bins=50):
    n = len(conf)
    if n < n_bins:
        return np.nan
    o = np.argsort(conf); c, a = conf[o], correct[o]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    return sum((hi - lo) / n * abs(a[lo:hi].mean() - c[lo:hi].mean())
               for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", default="phylum")
    ap.add_argument("--n-classes", type=int, default=36)
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args()
    root = args.root or Path("results/clean_eval") / args.rank

    data = {}
    for t in TIERS:
        for m in METHODS:
            f = root / t / m / "inference_uncertainty.csv"
            if f.is_file():
                data[(t, m)] = pd.read_csv(f)

    cfg = json.load(open(Path(CKPT[args.rank]) / "config.json"))
    known = {int(v) for v in cfg["id2label"].values()}
    print(f"=== label check ({args.rank}, checkpoint knows {len(known)} taxids) ===")
    usable = {}
    for t in TIERS:
        d = data.get((t, "base"))
        if d is None:
            print(f"  {t:18s} MISSING"); continue
        present = set(d.labels.unique())
        frac_oov = float((~d.labels.isin(known)).mean())
        ok = frac_oov < OOV_MAX
        usable[t] = ok
        print(f"  {t:18s} n={len(d):6d}  {len(present):3d} of {len(known)} taxids present  "
              f"OOV {frac_oov:6.1%}  "
              f"{'accuracy/ECE valid' if ok else 'mostly OOV -> AUROC only'}")

    print(f"\n=== calibration (ECE, 50 equal-mass bins) ===")
    rows = []
    for t in TIERS:
        if not usable.get(t):
            continue
        for m in METHODS:
            d = data.get((t, m))
            if d is None:
                continue
            conf = d.max_confidence.values
            corr = (d.pred == d.labels).values.astype(float)
            iv = d.labels.isin(known).values          # in-vocabulary reads
            rows.append(dict(tier=t, method=m, n=len(d), oov=float((~iv).mean()),
                             accuracy=corr.mean(), acc_invocab=corr[iv].mean() if iv.any() else np.nan,
                             mean_conf=conf.mean(), overconf=conf.mean() - corr.mean(),
                             ece=ece_equal_mass(conf, corr),
                             ece_invocab=ece_equal_mass(conf[iv], corr[iv]) if iv.sum() > 50 else np.nan))
    cal = pd.DataFrame(rows)
    if len(cal):
        print(cal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        cal.to_csv(f"results/clean_calibration_{args.rank}.csv", index=False)

    print(f"\n=== OOD detection (AUROC vs id_test) ===")
    rows = []
    for m in METHODS:
        idt = data.get(("id_test", m))
        if idt is None:
            continue
        for t in TIERS[1:]:
            d = data.get((t, m))
            if d is None:
                continue
            r = {"method": m, "tier": t}
            for col in ["max_confidence", "U_total", "U_epistemic"]:
                if col not in d or (idt[col].std() == 0 and d[col].std() == 0):
                    r[col] = np.nan; continue
                a, b = idt[col].values, d[col].values
                s = np.r_[a, b]
                if col == "max_confidence":
                    s = -s            # low confidence => novel
                r[col] = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], s)
            rows.append(r)
    ood = pd.DataFrame(rows)
    if len(ood):
        print(ood.rename(columns={"max_confidence": "AUROC_msp", "U_total": "AUROC_entropy",
                                  "U_epistemic": "AUROC_epistemic"})
                 .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        ood.to_csv(f"results/clean_ood_{args.rank}.csv", index=False)


if __name__ == "__main__":
    main()
