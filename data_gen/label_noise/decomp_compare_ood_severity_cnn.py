#!/usr/bin/env python3
"""Third leg of cell #2: same non-pretrained CNN, trained once on clean
(r00) data, evaluated on the real / shuffled / random_dna severity ladder
under both mc_dropout and ensemble sampling. See
decomp_compare_label_noise_cnn.py for why this isolates pretraining from
sampling mechanism as the explanation for the label-noise-cell divergence.
"""
from __future__ import annotations

import os
import random
import sys

import pandas as pd
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_scratch import one_hot_encode, train_cnn, batched_logits
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.common.variance_decomp import compute_uncertainty_variance

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEEDS = [1, 2, 3, 4, 42]
K_DROPOUT = 16
BASES = ["A", "C", "G", "T"]


def build_variant(seqs, kind: str, seed: int = 42):
    if kind == "real":
        return list(seqs)
    if kind == "shuffled":
        return ["".join(random.Random(seed + i).sample(s, len(s))) for i, s in enumerate(seqs)]
    if kind == "random_dna":
        return ["".join(random.Random(seed + i).choices(BASES, k=len(s))) for i, s in enumerate(seqs)]
    raise ValueError(kind)


def collect(logits_all, labels):
    unc = compute_uncertainty(logits_all)
    unc_var = compute_uncertainty_variance(logits_all)
    return pd.DataFrame({
        "labels": labels.numpy(), "pred": unc["predicted_class"].numpy(),
        "bald_U_epistemic": unc["normalized_epistemic_uncertainty"].numpy(),
        "bald_U_aleatoric": unc["normalized_aleatoric_uncertainty"].numpy(),
        "var_U_epistemic": unc_var["normalized_epistemic_uncertainty"].numpy(),
        "var_U_aleatoric": unc_var["normalized_aleatoric_uncertainty"].numpy(),
    })


def mwu(a, b, label):
    u, p = mannwhitneyu(a, b, alternative="greater")
    return {"comparison": label, "U": u, "p": p}


def main():
    train_df = pd.read_csv("data_gen/label_noise/csv_data_r00/train.csv")
    test_df = pd.read_csv("data_gen/label_noise/csv_data_r00/test.csv")
    X_train = one_hot_encode(train_df["sequence"].tolist())
    y_train = torch.tensor(train_df["label"].values, dtype=torch.long)
    labels = torch.tensor(test_df["original_label"].values, dtype=torch.long)

    variants = {
        kind: one_hot_encode(build_variant(test_df["sequence"], kind))
        for kind in ["real", "shuffled", "random_dna"]
    }

    results_table, mwu_table = [], []

    # ---- mc_dropout ----
    model = train_cnn(X_train, y_train, seed=SEEDS[0], device=DEVICE)
    enable_mc_dropout(model, p=0.3)
    dropout_out = {}
    with torch.no_grad():
        for kind, X in variants.items():
            logits_all = torch.stack(
                [batched_logits(model, X, DEVICE) for _ in range(K_DROPOUT)], dim=0
            )
            dropout_out[kind] = collect(logits_all, labels)

    # ---- ensemble ----
    members = [train_cnn(X_train, y_train, seed=s, device=DEVICE) for s in SEEDS]
    ens_out = {}
    with torch.no_grad():
        for kind, X in variants.items():
            logits_all = torch.stack([batched_logits(m, X, DEVICE) for m in members], dim=0)
            ens_out[kind] = collect(logits_all, labels)

    for name, out in [("cnn_mc_dropout", dropout_out), ("cnn_ensemble", ens_out)]:
        for score_col in ["bald_U_epistemic", "bald_U_aleatoric", "var_U_epistemic", "var_U_aleatoric"]:
            real = out["real"][score_col]
            for kind in ["real", "shuffled", "random_dna"]:
                d = out[kind]
                results_table.append({
                    "method": name, "score": score_col, "variant": kind,
                    "n": len(d), "mean": d[score_col].mean(), "std": d[score_col].std(),
                })
            for kind in ["shuffled", "random_dna"]:
                r = mwu(out[kind][score_col], real, f"{name}/{score_col}: {kind} > real")
                r["method"], r["score"] = name, score_col
                mwu_table.append(r)

    results_df = pd.DataFrame(results_table)
    mwu_df = pd.DataFrame(mwu_table)
    results_df.to_csv("data_gen/label_noise/decomp_compare_ood_results_cnn.csv", index=False)
    mwu_df.to_csv("data_gen/label_noise/decomp_compare_ood_mwu_cnn.csv", index=False)
    print("\n=== RESULTS ===")
    print(results_df.to_string(index=False))
    print("\n=== MANN-WHITNEY (proxy > real) ===")
    print(mwu_df.to_string(index=False))


if __name__ == "__main__":
    main()
