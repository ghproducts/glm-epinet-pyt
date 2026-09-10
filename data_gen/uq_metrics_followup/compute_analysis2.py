#!/usr/bin/env python3
"""Analysis 2: ECE done properly (bin-count sensitivity, ACE, NLL, Brier).

Per-example predicted-class probability + true label is required for any of
this. A scan of every axis's saved CSVs (see README.md "Data availability
audit") found real per-example probability survives in exactly two places:

  1. data_gen/aleatoric_boundary/csv_data/margin_scores.csv
     -- base DNABERT2 checkpoint (seed 1), margin-boundary axis, n=1584,
        column `prob_class1`. Covers only the `base` method (the margin
        score itself was computed from `base`'s own forward pass); the
        other 8 methods on this axis only persisted derived U_total/
        U_epistemic/U_aleatoric/vote_pct, not the underlying probability.
  2. data_gen/label_noise/evidential_ood_retest.csv
     -- evidential-head DNABERT2 checkpoint (seed 1), OOD-severity axis
        (real/shuffled/random_dna), n=1584 per condition, column
        `max_confidence` (= max(p0,p1), the Dirichlet mean probability of
        the predicted class). Covers only the `evidential` method.

No other axis/method combination in this project's saved CSVs has a
persisted per-example probability -- every other script only kept the
already-decomposed U_total/U_epistemic/U_aleatoric summary, not the raw
softmax/Dirichlet probability vector it was computed from. This is reported
plainly in README.md as a real data-availability gap, not fabricated.

Neither surviving dataset has more than one seed's worth of per-example
probabilities, so the "mean +/- std across seeds" requirement cannot be
met for ECE specifically anywhere in this project's existing data (seed
replication for U_total/epistemic/aleatoric exists but never carried the
probability alongside it) -- also reported as a gap, not silently dropped.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "inputs")
AB = os.path.join(IN, "aleatoric_boundary")
LN = os.path.join(IN, "label_noise")

BIN_COUNTS = [10, 15, 20, 25]
ADAPTIVE_BINS = 15
EPS = 1e-12


def equal_width_ece(p_true_class, correct, n_bins):
    """Standard ECE: bin by *confidence* (= P(predicted class)), equal-width
    bins over [0,1]. p_true_class here is actually the confidence in the
    PREDICTED class (conf), correct is 0/1 whether that prediction was right.
    """
    conf = p_true_class
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins, right=True) - 1, 0, n_bins - 1)
    N = len(conf)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece += (mask.sum() / N) * abs(acc_b - conf_b)
    return ece


def adaptive_ece(conf, correct, n_bins):
    """ACE: equal-MASS bins (each bin gets ~N/n_bins examples), via rank qcut
    on confidence.
    """
    N = len(conf)
    order = np.argsort(conf)
    conf_sorted = conf[order]
    correct_sorted = correct[order]
    edges = np.linspace(0, N, n_bins + 1).astype(int)
    ace = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        acc_b = correct_sorted[lo:hi].mean()
        conf_b = conf_sorted[lo:hi].mean()
        ace += ((hi - lo) / N) * abs(acc_b - conf_b)
    return ace


def nll_binary(p1, label):
    p1c = np.clip(p1, EPS, 1 - EPS)
    p_true = np.where(label == 1, p1c, 1 - p1c)
    return float(-np.log(p_true).mean())


def brier_binary(p1, label):
    return float(np.mean((p1 - label) ** 2))


rows = []  # dataset,method,condition,n,accuracy,bin_count,ece,ace,nll,brier


def summarize(name, method, condition, p1, pred, label):
    p1 = np.asarray(p1, dtype=float)
    label = np.asarray(label, dtype=int)
    pred = np.asarray(pred, dtype=int)
    conf = np.where(pred == 1, p1, 1 - p1)  # confidence in the PREDICTED class
    correct = (pred == label).astype(int)
    acc = correct.mean()
    nll = nll_binary(p1, label)
    brier = brier_binary(p1, label)
    ace15 = adaptive_ece(conf, correct, ADAPTIVE_BINS)
    for nb in BIN_COUNTS:
        ece = equal_width_ece(conf, correct, nb)
        rows.append(dict(dataset=name, method=method, condition=condition, n=len(p1),
                          accuracy=acc, bin_count=nb, ece_equal_width=ece,
                          ace_equal_mass_15bin=ace15, nll=nll, brier=brier))


# ---------------------------------------------------------------------------
# 1. margin_scores.csv -- base method, margin-boundary axis
#    Overall + per-quartile (the direct R2-3-style check: does per-quartile
#    ECE look "fine" even though Q1 accuracy is far worse than Q4's?).
# ---------------------------------------------------------------------------
ms = pd.read_csv(os.path.join(AB, "csv_data", "margin_scores.csv"))
summarize("margin_boundary", "base", "overall", ms["prob_class1"], ms["pred_base"], ms["label"])
for q, sub in ms.groupby("quartile"):
    summarize("margin_boundary", "base", q, sub["prob_class1"], sub["pred_base"], sub["label"])

# ---------------------------------------------------------------------------
# 2. evidential_ood_retest.csv -- evidential method, OOD-severity axis
#    max_confidence = confidence in the PREDICTED class already; recover p1.
# ---------------------------------------------------------------------------
ev = pd.read_csv(os.path.join(LN, "evidential_ood_retest.csv"))
ev["p1"] = np.where(ev["pred"] == 1, ev["max_confidence"], 1 - ev["max_confidence"])
for kind, sub in ev.groupby("kind"):
    summarize("ood_severity", "evidential", kind, sub["p1"], sub["pred"], sub["labels"])

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "analysis2_ece_ace_nll_brier.csv"), index=False)
print(out.to_string(index=False))
print()
print(f"Wrote {len(out)} rows to analysis2_ece_ace_nll_brier.csv")

# Bin-count sensitivity: max ECE range across the 4 bin counts, per row group
piv = out.pivot_table(index=["dataset", "method", "condition"], columns="bin_count", values="ece_equal_width")
piv["range"] = piv.max(axis=1) - piv.min(axis=1)
piv.to_csv(os.path.join(HERE, "analysis2_bin_sensitivity.csv"))
print()
print("Bin-count sensitivity (max-min ECE across bin counts 10/15/20/25):")
print(piv.to_string())
