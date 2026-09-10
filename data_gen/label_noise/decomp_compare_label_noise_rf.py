#!/usr/bin/env python3
"""Non-pretrained baseline: k-mer frequency + Random Forest, trained from
scratch at every label-noise rate. Each tree's `predict_proba` is treated
as one of K samples (the classical analogue of K independently-trained
checkpoints), same `[K, B, C]` shape into `compute_uncertainty`.

Data: data_gen/label_noise/csv_data_r{rate}/{train,test}.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from nn_proj.common.utils import compute_uncertainty
from nn_proj.common.variance_decomp import compute_uncertainty_variance

RATES = ["r00", "r05", "r10", "r20", "r40"]
K = 6            # 6-mers, matching NT_transformer's tokenization convention
N_TREES = 200    # ensemble size -- comparable order of magnitude to k_samples=16 elsewhere,
                 # but a Random Forest sees no benefit from few trees, so use a standard value
MIN_SAMPLES_LEAF = 20  # sklearn's default (1) grows trees to pure leaves, so a single tree's
                        # own predict_proba is almost always exactly {0,1} -- a degenerate
                        # artifact that collapses the per-tree entropy term to ~0 regardless of
                        # true label noise. 20 is a standard leaf-smoothing value (used whenever
                        # RF probability estimates need to be non-degenerate, e.g. for calibration)
                        # that lets individual trees actually express uncertainty.
SEED = 0
EPS = 1e-6


def kmer_vectorizer():
    return CountVectorizer(analyzer="char", ngram_range=(K, K), lowercase=False)


def per_tree_probs(rf: RandomForestClassifier, X) -> torch.Tensor:
    """[n_estimators, n_samples, n_classes] stack of each tree's own
    predict_proba, the RF analogue of K independent forward passes."""
    probs = np.stack([tree.predict_proba(X) for tree in rf.estimators_], axis=0)
    probs = np.clip(probs, EPS, 1.0 - EPS)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    # compute_uncertainty/compute_uncertainty_variance both softmax their input,
    # and softmax(log p) == p exactly (p already sums to 1) -- so log-probabilities
    # round-trip through the existing logit-shaped decomposition code unchanged.
    return torch.from_numpy(np.log(probs)).float()


def main():
    rows = []
    for rate in RATES:
        train_df = pd.read_csv(f"data_gen/label_noise/csv_data_{rate}/train.csv")
        test_df = pd.read_csv(f"data_gen/label_noise/csv_data_{rate}/test.csv")

        vec = kmer_vectorizer()
        X_train = vec.fit_transform(train_df["sequence"])
        X_test = vec.transform(test_df["sequence"])

        rf = RandomForestClassifier(
            n_estimators=N_TREES, random_state=SEED, n_jobs=-1, min_samples_leaf=MIN_SAMPLES_LEAF,
        )
        rf.fit(X_train, train_df["label"].values)

        logits_all = per_tree_probs(rf, X_test)  # [K, B, C]
        unc = compute_uncertainty(logits_all)
        unc_var = compute_uncertainty_variance(logits_all)

        labels = torch.from_numpy(test_df["original_label"].values)
        acc = (unc["predicted_class"] == labels).float().mean().item()

        row = {
            "rate": rate, "accuracy": acc,
            "bald_U_epistemic_mean": unc["normalized_epistemic_uncertainty"].mean().item(),
            "bald_U_epistemic_std": unc["normalized_epistemic_uncertainty"].std().item(),
            "bald_U_aleatoric_mean": unc["normalized_aleatoric_uncertainty"].mean().item(),
            "bald_U_aleatoric_std": unc["normalized_aleatoric_uncertainty"].std().item(),
            "var_U_epistemic_mean": unc_var["normalized_epistemic_uncertainty"].mean().item(),
            "var_U_epistemic_std": unc_var["normalized_epistemic_uncertainty"].std().item(),
            "var_U_aleatoric_mean": unc_var["normalized_aleatoric_uncertainty"].mean().item(),
            "var_U_aleatoric_std": unc_var["normalized_aleatoric_uncertainty"].std().item(),
        }
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv("data_gen/label_noise/decomp_compare_label_noise_rf.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
