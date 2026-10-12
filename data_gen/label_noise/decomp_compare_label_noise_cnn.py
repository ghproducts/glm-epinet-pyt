#!/usr/bin/env python3
"""Third leg of cell #1: a non-pretrained *neural* baseline (small 1D-CNN,
trained from scratch, no GLM), tested under both UQ-sampling mechanisms
used elsewhere in this cross-check:

  - mc_dropout : one CNN per rate, dropout active at eval, K=16 passes.
  - ensemble   : K=5 independently random-initialized-and-trained CNNs per
                 rate (seeds 1,2,3,4,42 -- this project's standard seed set),
                 sharing no weights at all (unlike the DNABERT2 deep
                 ensembles, which all start from the same pretrained init).

This isolates whether the earlier finding -- ensembling shows label noise as
rising aleatoric with flat/falling epistemic, while dropout/epinet sampling
leaks it into epistemic -- is a property of *pretraining* or a property of
the *sampling mechanism itself*. If the CNN's mc_dropout leaks the same way
DNABERT2's did, and the CNN's ensemble doesn't, the mechanism is the cause,
not the GLM backbone.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_scratch import one_hot_encode, train_cnn, batched_logits
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.common.variance_decomp import compute_uncertainty_variance

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RATES = ["r00", "r05", "r10", "r20", "r40"]
SEEDS = [1, 2, 3, 4, 42]
K_DROPOUT = 16


def decompose(logits_all, labels):
    unc = compute_uncertainty(logits_all)
    unc_var = compute_uncertainty_variance(logits_all)
    acc = (unc["predicted_class"] == labels).float().mean().item()
    return {
        "accuracy": acc,
        "bald_U_epistemic_mean": unc["normalized_epistemic_uncertainty"].mean().item(),
        "bald_U_aleatoric_mean": unc["normalized_aleatoric_uncertainty"].mean().item(),
        "var_U_epistemic_mean": unc_var["normalized_epistemic_uncertainty"].mean().item(),
        "var_U_aleatoric_mean": unc_var["normalized_aleatoric_uncertainty"].mean().item(),
    }


def main():
    rows = []
    for rate in RATES:
        train_df = pd.read_csv(f"data_gen/label_noise/csv_data_{rate}/train.csv")
        test_df = pd.read_csv(f"data_gen/label_noise/csv_data_{rate}/test.csv")
        X_train = one_hot_encode(train_df["sequence"].tolist())
        y_train = torch.tensor(train_df["label"].values, dtype=torch.long)
        X_test = one_hot_encode(test_df["sequence"].tolist())
        labels = torch.tensor(test_df["original_label"].values, dtype=torch.long)

        # ---- mc_dropout: one model, K stochastic passes ----
        model = train_cnn(X_train, y_train, seed=SEEDS[0], device=DEVICE)
        enable_mc_dropout(model, p=0.3)
        with torch.no_grad():
            logits_all = torch.stack(
                [batched_logits(model, X_test, DEVICE) for _ in range(K_DROPOUT)], dim=0
            )
        row = {"rate": rate, "sampling": "mc_dropout", **decompose(logits_all, labels)}
        rows.append(row)
        print(row)

        # ---- ensemble: K independently-trained models ----
        members = [train_cnn(X_train, y_train, seed=s, device=DEVICE) for s in SEEDS]
        with torch.no_grad():
            logits_all = torch.stack([batched_logits(m, X_test, DEVICE) for m in members], dim=0)
        row = {"rate": rate, "sampling": "ensemble", **decompose(logits_all, labels)}
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv("data_gen/label_noise/decomp_compare_label_noise_cnn.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
