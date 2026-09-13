#!/usr/bin/env python3
"""Non-pretrained baselines (k-mer Random Forest, from-scratch 1D-CNN) in the
same per-example schema as run_sweep.py, so summarize.py treats them as just
more methods.

These answer R2-7's "why only GLMs" directly: no pretraining, no transfer, no
GPU needed for RF. Filed under backbone name `nonpretrained` because they have
no backbone -- keeping them out of the four GLM columns rather than pretending
they belong to one.

    python run_baselines.py --datasets boundary,alisim --seeds 1,2,3

Methods:
  rf              200-tree RF over 6-mer counts; each tree's predict_proba is
                  one of K samples (the classical analogue of an ensemble)
  cnn_mc_dropout  one CNN per seed, dropout live at eval, K=16 passes
  cnn_ensemble    K=5 independently initialized+trained CNNs (filed as seed 'ens')
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep_lib as S  # noqa: E402

sys.path.insert(0, S.REPO)
sys.path.insert(0, os.path.join(S.REPO, "data_gen", "label_noise"))
from nn_proj.common.utils import compute_uncertainty  # noqa: E402

RF_K = 6
RF_N_TREES = 200
RF_MIN_SAMPLES_LEAF = 20  # sklearn's default of 1 grows pure leaves, making every
                          # per-tree predict_proba exactly 0/1 and collapsing the
                          # per-tree entropy term regardless of real uncertainty
CNN_K_DROPOUT = 16
CNN_DROPOUT_P = 0.3
CNN_ENSEMBLE_SEEDS = [1, 2, 3, 4, 42]
BACKBONE_LABEL = "nonpretrained"


def load_train():
    df = pd.read_csv(S.LABEL_NOISE_TRAIN)
    return df["sequence"].tolist(), df["label"].astype(int).to_numpy()


def load_eval(dataset: str):
    """Same rows, same order, as the GLM cells -- so per-example comparisons
    line up. Rebuilt from the raw CSVs rather than tokenized datasets, since
    these models take sequences not token ids."""
    if dataset == "boundary":
        from nn_proj.common.datasets import load_NT_tasks
        ds = load_NT_tasks(task="promoter_all", split="test")
        margins = pd.read_csv(S.BOUNDARY_MARGINS)
        meta = pd.DataFrame({"margin": margins["margin"], "quartile": margins["quartile"]})
        return list(ds["sequence"]), np.asarray(ds["labels"], dtype=int), meta
    path = S.ALISIM_EXTENDED_CSV if dataset == "alisim_extended" else S.ALISIM_CSV
    df = pd.read_csv(path)
    label_col = "label" if "label" in df.columns else "labels"
    meta_cols = [c for c in ("anchor_id", "branch_length",
                             "realized_identity_to_anchor", "original_label")
                 if c in df.columns]
    return df["sequence"].tolist(), df[label_col].astype(int).to_numpy(), df[meta_cols]


def emit(probs_k: torch.Tensor, labels: np.ndarray, meta: pd.DataFrame,
         method: str) -> list[dict]:
    """probs_k: [K, N, C] per-sample probabilities."""
    unc = compute_uncertainty(torch.log(probs_k.clamp_min(1e-12)))
    rows = []
    for i in range(len(labels)):
        pred = int(unc["predicted_class"][i])
        row = {**{c: meta[c].iloc[i] for c in meta.columns},
               "method": method, "row_idx": i,
               "U_total": float(unc["normalized_total_uncertainty"][i]),
               "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
               "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
               "pred": pred, "labels": int(labels[i]),
               "correct": int(pred == int(labels[i]))}
        row.update(S.probs_to_row_fields(unc["mean_probs"][i]))
        rows.append(row)
    return rows


def run_rf(train_seqs, train_y, eval_seqs, labels, meta, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(analyzer="char", ngram_range=(RF_K, RF_K), lowercase=False)
    Xtr = vec.fit_transform(train_seqs)
    Xte = vec.transform(eval_seqs)
    rf = RandomForestClassifier(n_estimators=RF_N_TREES, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                                random_state=int(seed), n_jobs=-1)
    rf.fit(Xtr, train_y)
    # each tree is one posterior-like sample -> [K, N, C]. Densify once, not
    # once per tree (200 trees x a 6400 x 4096 matrix otherwise).
    Xte_dense = Xte.toarray()
    per_tree = np.stack([t.predict_proba(Xte_dense) for t in rf.estimators_], axis=0)
    return emit(torch.from_numpy(per_tree).float(), labels, meta, "rf")


def run_cnn(train_seqs, train_y, eval_seqs, labels, meta, seed, mode):
    from cnn_scratch import batched_logits, one_hot_encode, train_cnn
    dev = S.DEVICE
    Xtr = one_hot_encode(train_seqs)
    ytr = torch.from_numpy(train_y).long()
    Xte = one_hot_encode(eval_seqs)

    if mode == "cnn_mc_dropout":
        model = train_cnn(Xtr, ytr, seed=int(seed), device=dev, dropout_p=CNN_DROPOUT_P)
        model.train()  # keep dropout live at eval -- that is the sampling mechanism
        with torch.no_grad():
            stack = torch.stack([batched_logits(model, Xte, dev).softmax(-1)
                                 for _ in range(CNN_K_DROPOUT)], dim=0)
        return emit(stack, labels, meta, "cnn_mc_dropout")

    probs = []
    for s in CNN_ENSEMBLE_SEEDS:
        m = train_cnn(Xtr, ytr, seed=int(s), device=dev, dropout_p=CNN_DROPOUT_P)
        m.eval()
        with torch.no_grad():
            probs.append(batched_logits(m, Xte, dev).softmax(-1))
        del m
    return emit(torch.stack(probs, dim=0), labels, meta, "cnn_ensemble")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="boundary,alisim")
    ap.add_argument("--methods", default="rf,cnn_mc_dropout,cnn_ensemble")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    train_seqs, train_y = load_train()
    print(f"train: {len(train_seqs)} sequences")
    failures = []
    for dataset in [d.strip() for d in args.datasets.split(",")]:
        eval_seqs, labels, meta = load_eval(dataset)
        print(f"{dataset}: {len(eval_seqs)} eval sequences")
        for method in [m.strip() for m in args.methods.split(",")]:
            seeds = ["ens"] if method == "cnn_ensemble" else [s.strip() for s in args.seeds.split(",")]
            for seed in seeds:
                out = S.cell_path(dataset, BACKBONE_LABEL, method, seed)
                if os.path.exists(out) and not args.force:
                    print(f"[skip] {dataset}/{method}/{seed}: already done")
                    continue
                try:
                    print(f"=== {dataset} / {method} / seed {seed} ===", flush=True)
                    if method == "rf":
                        rows = run_rf(train_seqs, train_y, eval_seqs, labels, meta, seed)
                    else:
                        rows = run_cnn(train_seqs, train_y, eval_seqs, labels, meta,
                                       seed if seed != "ens" else 1, method)
                    path = S.write_cell(rows, dataset, BACKBONE_LABEL, method, seed)
                    acc = sum(r["correct"] for r in rows) / len(rows)
                    print(f"    wrote {path}  n={len(rows)}  acc={acc:.4f}", flush=True)
                except Exception as exc:
                    import traceback
                    print(f"[FAILED] {dataset}/{method}/{seed}: {type(exc).__name__}: {exc}",
                          flush=True)
                    traceback.print_exc()
                    failures.append((dataset, method, seed))
    print(f"\ndone. {len(failures)} failures")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
