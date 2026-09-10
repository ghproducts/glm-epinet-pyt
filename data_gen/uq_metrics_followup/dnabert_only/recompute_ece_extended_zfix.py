#!/usr/bin/env python3
"""Recomputes pooled + stratified ECE/NLL/Brier for all 11
method-configurations (base, base_scaled, mc_dropout, conv_epinet, laplace,
ensemble_k5, ensemble_k3, evidential, cnn_mc_dropout, cnn_ensemble, rf) on
BOTH datasets, using the UPDATED data:

  - promoter_alisim: the full 19-point grid (16-point dense grid + 3-point
    extension {5.0, 8.0, 12.0}), `conv_epinet` from the retrained
    `epinet_zfix` checkpoint, `base`/`base_scaled` freshly run on all 19
    points.
  - aleatoric_boundary: same 4 margin quartiles as before (no grid to
    extend), `conv_epinet` from the retrained `epinet_zfix` checkpoint, all
    8 other methods unchanged.

Direct adaptation of `all_methods_stratified_ece.py` (same
invert_binary_entropy / ece_binned / nll_brier / metrics_row logic) --
only the input paths differ (extended-grid / zfix files instead of the
original ones).

Usage:
    /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \\
        data_gen/uq_metrics_followup/dnabert_only/recompute_ece_extended_zfix.py
"""
import os
import numpy as np
import pandas as pd

THIS_REPO = "/scratch/home/glh52/glm-epinet-pyt/.claude/worktrees/agent-ad064a1f05d218c98"
ALISIM_DIR = os.path.join(THIS_REPO, "data_gen/promoter_alisim/uncertainty_eval/dense_grid_extended_zfix")
BOUNDARY_DIR = os.path.join(THIS_REPO, "data_gen/aleatoric_boundary")
OUT_DIR = os.path.join(THIS_REPO, "data_gen/uq_metrics_followup/dnabert_only")

# Methods whose U_total = U_aleatoric + U_epistemic exactly.
SUM_METHODS = ["mc_dropout", "conv_epinet", "laplace", "ensemble_k5", "ensemble_k3",
               "cnn_mc_dropout", "cnn_ensemble"]
ALEATORIC_ONLY_METHODS = ["evidential"]
RF_NAME = {"aleatoric_boundary": "rf", "promoter_alisim_dense": "rf_kmer"}
SUM_METHODS_NO_RF = [m for m in SUM_METHODS if m not in ("rf", "rf_kmer")]


def invert_binary_entropy(u_total):
    u_total = np.clip(u_total, 1e-12, 1.0)
    lo = np.full_like(u_total, 0.5)
    hi = np.full_like(u_total, 1.0 - 1e-12)
    for _ in range(60):
        mid = (lo + hi) / 2
        h = -(mid * np.log(mid) + (1 - mid) * np.log(1 - mid)) / np.log(2)
        lo = np.where(h > u_total, mid, lo)
        hi = np.where(h > u_total, hi, mid)
    return (lo + hi) / 2


def ece_binned(conf, correct, n_bins=15):
    n = len(conf)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, edges[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        if len(idx) == 0:
            continue
        ece += (len(idx) / n) * abs(conf[idx].mean() - correct[idx].mean())
    return ece


def nll_brier(p1, label):
    eps = 1e-12
    p1c = np.clip(p1, eps, 1 - eps)
    p_true = np.where(label == 1, p1c, 1 - p1c)
    return -np.log(p_true).mean(), ((p1 - label) ** 2).mean()


def metrics_row(p1, pred, label, **extra):
    correct = (pred == label).astype(int)
    conf = np.where(pred == 1, p1, 1 - p1)
    nll, brier = nll_brier(p1, label)
    row = {"n": len(p1), "accuracy": correct.mean(), "nll": nll, "brier": brier,
           "ece_15bin": ece_binned(conf, correct, 15)}
    row.update(extra)
    return row


def main():
    pooled_rows, strat_rows = [], []

    # ================= aleatoric_boundary =================
    mscore = pd.read_csv(f"{BOUNDARY_DIR}/csv_data/margin_scores.csv")
    p1 = mscore["prob_class1"].values.astype(float)
    pred = mscore["pred_base"].values.astype(int)
    label = mscore["label"].values.astype(int)
    quartile = mscore["quartile"].values
    with open(f"{THIS_REPO}/data_gen/promoter_alisim/uncertainty_eval/dense_grid_extended_zfix/fitted_temperature.txt") as f:
        T_line = [l for l in f if l.startswith("T=")][0]
        T_FITTED = float(T_line.strip().split("=")[1])
    z = np.log(np.clip(p1, 1e-12, 1 - 1e-12) / np.clip(1 - p1, 1e-12, 1 - 1e-12))
    p1_scaled = 1 / (1 + np.exp(-z / T_FITTED))

    for name, probs in [("base", p1), ("base_scaled", p1_scaled)]:
        pooled_rows.append(metrics_row(probs, pred, label, dataset="aleatoric_boundary", method=name, stratum="pooled"))
        for q in sorted(set(quartile)):
            m = quartile == q
            strat_rows.append(metrics_row(probs[m], pred[m], label[m], dataset="aleatoric_boundary", method=name, stratum=q))

    uqb = pd.read_csv(f"{BOUNDARY_DIR}/csv_data/uncertainty_by_method_zfix.csv")  # conv_epinet updated to zfix
    boundary_methods = SUM_METHODS_NO_RF + [RF_NAME["aleatoric_boundary"]] + ALEATORIC_ONLY_METHODS
    for m in boundary_methods:
        sub = uqb[uqb["method"] == m]
        if m in ALEATORIC_ONLY_METHODS:
            u = sub["U_aleatoric"].values.astype(float)
        else:
            u = (sub["U_aleatoric"] + sub["U_epistemic"]).values.astype(float)
        conf = invert_binary_entropy(u)
        pr = sub["pred"].values.astype(int)
        lb = sub["label"].values.astype(int)
        qt = sub["quartile"].values
        p1_m = np.where(pr == 1, conf, 1 - conf)
        method_label = "rf" if m == RF_NAME["aleatoric_boundary"] else m
        pooled_rows.append(metrics_row(p1_m, pr, lb, dataset="aleatoric_boundary", method=method_label, stratum="pooled"))
        for q in sorted(set(qt)):
            mm = qt == q
            strat_rows.append(metrics_row(p1_m[mm], pr[mm], lb[mm], dataset="aleatoric_boundary", method=method_label, stratum=q))

    # ================= promoter_alisim (19-point extended grid) =================
    base19 = pd.read_csv(f"{ALISIM_DIR}/per_example_base_basescaled_19pt.csv")
    base19["branch_length"] = base19["branch_length"].round(6)
    for name, col in [("base", "p1"), ("base_scaled", "p1_scaled")]:
        pred_col = "pred" if col == "p1" else "pred_scaled"
        probs = base19[col].values.astype(float)
        pr = base19[pred_col].values.astype(int)
        lb = base19["labels"].values.astype(int)
        bl = base19["branch_length"].values
        pooled_rows.append(metrics_row(probs, pr, lb, dataset="promoter_alisim_extended19", method=name, stratum="pooled"))
        for t in sorted(set(bl)):
            m = bl == t
            strat_rows.append(metrics_row(probs[m], pr[m], lb[m], dataset="promoter_alisim_extended19", method=name, stratum=t))

    uqa = pd.read_csv(f"{ALISIM_DIR}/per_example_uncertainty_19pt.csv")  # conv_epinet zfix + 19pt others
    uqa["branch_length"] = uqa["branch_length"].round(6)
    alisim_methods = SUM_METHODS_NO_RF + [RF_NAME["promoter_alisim_dense"]] + ALEATORIC_ONLY_METHODS
    for m in alisim_methods:
        sub = uqa[uqa["method"] == m]
        if m in ALEATORIC_ONLY_METHODS:
            u = sub["U_aleatoric"].values.astype(float)
        else:
            u = (sub["U_aleatoric"] + sub["U_epistemic"]).values.astype(float)
        conf = invert_binary_entropy(u)
        pr = sub["pred"].values.astype(int)
        lb = sub["labels"].values.astype(int)
        blm = sub["branch_length"].values
        p1_m = np.where(pr == 1, conf, 1 - conf)
        method_label = "rf" if m == RF_NAME["promoter_alisim_dense"] else m
        pooled_rows.append(metrics_row(p1_m, pr, lb, dataset="promoter_alisim_extended19", method=method_label, stratum="pooled"))
        for t in sorted(set(blm)):
            mm = blm == t
            strat_rows.append(metrics_row(p1_m[mm], pr[mm], lb[mm], dataset="promoter_alisim_extended19", method=method_label, stratum=t))

    pooled = pd.DataFrame(pooled_rows)
    strat = pd.DataFrame(strat_rows)
    order = ["base", "base_scaled", "mc_dropout", "conv_epinet", "laplace", "ensemble_k5", "ensemble_k3",
             "evidential", "cnn_mc_dropout", "cnn_ensemble", "rf"]
    pooled["method"] = pd.Categorical(pooled["method"], categories=order, ordered=True)
    strat["method"] = pd.Categorical(strat["method"], categories=order, ordered=True)
    pooled = pooled.sort_values(["dataset", "method"])
    strat = strat.sort_values(["dataset", "method", "stratum"])

    pooled.to_csv(os.path.join(OUT_DIR, "pooled_ece_allmethods_extended_zfix.csv"), index=False)
    strat.to_csv(os.path.join(OUT_DIR, "stratified_ece_allmethods_extended_zfix.csv"), index=False)
    pd.set_option("display.width", 220)
    print("=== POOLED (11 methods x 2 datasets, extended grid + zfix conv_epinet) ===")
    print(pooled.round(4).to_string(index=False))
    print("\n=== STRATIFIED (first/last few rows) ===")
    print(strat.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
