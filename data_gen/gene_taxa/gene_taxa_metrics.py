#!/usr/bin/env python3
"""Calibration + uncertainty-based OOD metrics for the gene_taxa evaluation.

Same protocol as data_gen/pbsim/clean_metrics.py: ECE over 50 equal-mass bins
(matching .docs/latex/UQ_plos_paper), AUROC treating an uncertainty score as a
detection statistic against the ID tier.

Which tiers can carry which metric, and why:
  test      ID, 437 labels, all valid          accuracy + ECE + AUROC baseline
  taxa_out  novel taxon, 436 valid labels      accuracy + ECE + AUROC
  gene_out  novel gene, every row label=1      AUROC ONLY -- that label is a
            placeholder, not a class assignment (label 1 is 0.25% of test), so
            any accuracy or ECE computed on it is meaningless by construction.

Labels are NOT re-encoded: inference.py calls load_local_dataset with
encode_labels=False and no taxa_rank, so the raw integers reach the model's own
label space. (pbsim needs the encoding check in clean_metrics.py because it
remaps taxids through a lineage table; gene_taxa does not.) The check is kept
here anyway, as a guard rather than a diagnosis.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("results/gene_taxa_eval")
TIERS = ["test", "taxa_out", "gene_out"]
LABELLED = {"test": True, "taxa_out": True, "gene_out": False}
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
    data = {}
    for t in TIERS:
        for m in METHODS:
            f = ROOT / t / m / "inference_uncertainty.csv"
            if f.is_file():
                data[(t, m)] = pd.read_csv(f)

    print("=== tiers present ===")
    for t in TIERS:
        d = data.get((t, "base"))
        if d is None:
            print(f"  {t:9s} MISSING"); continue
        note = "labels valid" if LABELLED[t] else "DUMMY labels -> AUROC only"
        print(f"  {t:9s} n={len(d):6d}  distinct labels={d.labels.nunique():3d}  {note}")

    print("\n=== calibration (ECE, 50 equal-mass bins) ===")
    rows = []
    for t in TIERS:
        if not LABELLED[t]:
            continue
        for m in METHODS:
            d = data.get((t, m))
            if d is None:
                continue
            conf = d.max_confidence.values
            corr = (d.pred == d.labels).values.astype(float)
            rows.append(dict(tier=t, method=m, n=len(d), accuracy=corr.mean(),
                             mean_conf=conf.mean(), overconf=conf.mean() - corr.mean(),
                             ece=ece_equal_mass(conf, corr)))
    cal = pd.DataFrame(rows)
    if len(cal):
        print(cal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        cal.to_csv("results/gene_taxa_calibration.csv", index=False)

    print("\n=== OOD detection (AUROC vs test; 0.5 = no signal) ===")
    rows = []
    for m in METHODS:
        idt = data.get(("test", m))
        if idt is None:
            continue
        for t in ["taxa_out", "gene_out"]:
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
        ood.to_csv("results/gene_taxa_ood.csv", index=False)


if __name__ == "__main__":
    main()
