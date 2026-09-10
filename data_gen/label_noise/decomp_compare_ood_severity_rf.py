#!/usr/bin/env python3
"""Non-pretrained baseline for cell #2: k-mer frequency + Random Forest,
trained once (clean r00 training data) and evaluated on the same
real / shuffled / random_dna severity ladder used for mc_dropout,
conv_epinet, and the deep ensembles.

See decomp_compare_label_noise_rf.py for why per-tree class probabilities
are a legitimate classical analogue of K sampled forward passes.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import mannwhitneyu

from nn_proj.common.utils import compute_uncertainty
from nn_proj.common.variance_decomp import compute_uncertainty_variance

K = 6
N_TREES = 200
MIN_SAMPLES_LEAF = 20  # see decomp_compare_label_noise_rf.py: sklearn's default (1) grows
                        # pure leaves, collapsing per-tree aleatoric to ~0 as an artifact.
SEED = 0
EPS = 1e-6
BASES = ["A", "C", "G", "T"]


def kmer_vectorizer():
    return CountVectorizer(analyzer="char", ngram_range=(K, K), lowercase=False)


def per_tree_probs(rf: RandomForestClassifier, X) -> torch.Tensor:
    probs = np.stack([tree.predict_proba(X) for tree in rf.estimators_], axis=0)
    probs = np.clip(probs, EPS, 1.0 - EPS)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return torch.from_numpy(np.log(probs)).float()


def build_variant(seqs, kind: str, seed: int = 42):
    if kind == "real":
        return list(seqs)
    if kind == "shuffled":
        return ["".join(random.Random(seed + i).sample(s, len(s))) for i, s in enumerate(seqs)]
    if kind == "random_dna":
        return ["".join(random.Random(seed + i).choices(BASES, k=len(s))) for i, s in enumerate(seqs)]
    raise ValueError(kind)


def mwu(a, b, label):
    u, p = mannwhitneyu(a, b, alternative="greater")
    return {"comparison": label, "U": u, "p": p}


def main():
    train_df = pd.read_csv("data_gen/label_noise/csv_data_r00/train.csv")
    test_df = pd.read_csv("data_gen/label_noise/csv_data_r00/test.csv")

    vec = kmer_vectorizer()
    X_train = vec.fit_transform(train_df["sequence"])
    rf = RandomForestClassifier(
        n_estimators=N_TREES, random_state=SEED, n_jobs=-1, min_samples_leaf=MIN_SAMPLES_LEAF,
    )
    rf.fit(X_train, train_df["label"].values)

    variants = {kind: build_variant(test_df["sequence"], kind) for kind in ["real", "shuffled", "random_dna"]}
    labels = test_df["original_label"].values

    out = {}
    for kind, seqs in variants.items():
        X = vec.transform(seqs)  # unseen k-mers (e.g. from random_dna) are dropped, matching training vocab
        logits_all = per_tree_probs(rf, X)
        unc = compute_uncertainty(logits_all)
        unc_var = compute_uncertainty_variance(logits_all)
        out[kind] = pd.DataFrame({
            "labels": labels, "pred": unc["predicted_class"].numpy(),
            "bald_U_epistemic": unc["normalized_epistemic_uncertainty"].numpy(),
            "bald_U_aleatoric": unc["normalized_aleatoric_uncertainty"].numpy(),
            "var_U_epistemic": unc_var["normalized_epistemic_uncertainty"].numpy(),
            "var_U_aleatoric": unc_var["normalized_aleatoric_uncertainty"].numpy(),
        })

    results_table, mwu_table = [], []
    for score_col in ["bald_U_epistemic", "bald_U_aleatoric", "var_U_epistemic", "var_U_aleatoric"]:
        real = out["real"][score_col]
        for kind in ["real", "shuffled", "random_dna"]:
            d = out[kind]
            results_table.append({
                "method": "rf_kmer", "score": score_col, "variant": kind,
                "n": len(d), "mean": d[score_col].mean(), "std": d[score_col].std(),
            })
        for kind in ["shuffled", "random_dna"]:
            r = mwu(out[kind][score_col], real, f"rf_kmer/{score_col}: {kind} > real")
            r["score"] = score_col
            mwu_table.append(r)

    results_df = pd.DataFrame(results_table)
    mwu_df = pd.DataFrame(mwu_table)
    results_df.to_csv("data_gen/label_noise/decomp_compare_ood_results_rf.csv", index=False)
    mwu_df.to_csv("data_gen/label_noise/decomp_compare_ood_mwu_rf.csv", index=False)
    print("\n=== RESULTS ===")
    print(results_df.to_string(index=False))
    print("\n=== MANN-WHITNEY (proxy > real) ===")
    print(mwu_df.to_string(index=False))


if __name__ == "__main__":
    main()
