#!/usr/bin/env python3
"""Extends `mc_dropout`, `evidential`, `laplace`, `ensemble_k5`,
`ensemble_k3`, `cnn_mc_dropout`, `cnn_ensemble`, `rf_kmer` to the 3 NEW
branch-length points (5.0, 8.0, 12.0) added past the dense grid's top end
(3.0). None of these 8 methods touch `nn_proj.models.epinet`, so they are
UNAFFECTED by the epinet per-example-z fix / retrain -- only the grid
extension applies to them, not a checkpoint change. Existing 16-point rows
for these methods (`uncertainty_eval/dense_grid/per_example_uncertainty.csv`)
are reused unchanged; this script computes ONLY the 3 new points, using
`csv_data/promoter_alisim_extended.csv` (400 anchors x 3 branch lengths =
1200 rows) as input.

Direct adaptation of `uncertainty_eval/dense_grid/run_new_methods_dense.py`
(same checkpoints, same K, same hyperparameters) -- only the input CSV
differs.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=1 /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \\
        data_gen/promoter_alisim/uncertainty_eval/dense_grid_extended_zfix/run_other_methods_extended.py
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT_FOR_IMPORT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _REPO_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_IMPORT)
_LABEL_NOISE_DIR = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen", "label_noise")
if _LABEL_NOISE_DIR not in sys.path:
    sys.path.insert(0, _LABEL_NOISE_DIR)
_DENSE_GRID_DIR = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen", "promoter_alisim", "uncertainty_eval", "dense_grid")
if _DENSE_GRID_DIR not in sys.path:
    sys.path.insert(0, _DENSE_GRID_DIR)

import numpy as np
import pandas as pd
import torch
import transformers
from safetensors.torch import load_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from nn_proj.common.datasets import load_local_dataset, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.models.laplace import LaplaceConfig, LaplaceSeqClassifier, fit_diagonal_laplace
from nn_proj.models.evidential import EvidentialConfig, EvidentialWrapper, HFEvidentialSeqClassifier
from cnn_scratch import one_hot_encode, train_cnn, batched_logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
DATA_PATH = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/csv_data/promoter_alisim_extended.csv")
OUT_DIR = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/uncertainty_eval/dense_grid_extended_zfix")
METADATA_COLS = ["anchor_id", "original_label", "branch_length", "realized_identity_to_anchor"]

REPO_ROOT = "/scratch/home/glh52/glm-epinet-pyt"
LABEL_NOISE_R00_TRAIN = os.path.join(REPO_ROOT, "data_gen/label_noise/csv_data_r00/train.csv")
EVID_CKPT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/evidential")

ENSEMBLE_SEEDS_K5 = [1, 2, 3, 4, 42]
ENSEMBLE_SEEDS_K3 = [1, 2, 3]
CNN_ENSEMBLE_SEEDS = [1, 2, 3, 4, 42]
CNN_MCD_SEED = 1

MODEL_MAX_LENGTH = 75
BATCH_SIZE = 32
K_MC = 10
K_LAPLACE = 16
LAPLACE_FIT_EXAMPLES = 2000
LAPLACE_PRIOR_PRECISION = 1.0
K_CNN_DROPOUT = 16
CNN_DROPOUT_P = 0.3
RF_K = 6
RF_N_TREES = 200
RF_MIN_SAMPLES_LEAF = 20
RF_SEED = 0

OUT_COLS = ["anchor_id", "branch_length", "realized_identity_to_anchor", "original_label",
            "method", "U_aleatoric", "U_epistemic", "pred", "labels", "correct"]


def load_base_arch(num_labels: int = 2):
    config = transformers.AutoConfig.from_pretrained(HUB_ID, num_labels=num_labels, trust_remote_code=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )
    return model, config


def load_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=MODEL_MAX_LENGTH, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    tokenizer.eos_token = tokenizer.pad_token
    return tokenizer


def load_alisim_dataset_for_dnabert(tokenizer):
    ds = load_local_dataset(DATA_PATH, encode_labels=True)
    metadata_cols = [c for c in METADATA_COLS if c in ds.column_names]
    tokenized, collator = prep_for_trainer(ds, tokenizer, max_length=MODEL_MAX_LENGTH, metadata_cols=metadata_cols)
    collate_ds = tokenized.remove_columns(["sequence"] + metadata_cols)
    return tokenized, collate_ds, collator, metadata_cols


def _extract_metadata(tokenized_ds, metadata_cols, idx_start, idx_end):
    rows = []
    for i in range(idx_start, idx_end):
        row = {}
        for c in metadata_cols:
            v = tokenized_ds[c][i]
            row[c] = v.item() if isinstance(v, torch.Tensor) else v
        rows.append(row)
    return rows


def load_dnabert_base(ckpt_dir, config):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )
    model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
    return model.to(DEVICE).eval()


@torch.no_grad()
def run_mc_dropout(tokenized_ds, collate_ds, collator, metadata_cols):
    print("=== mc_dropout ===")
    _, config = load_base_arch()
    model = load_dnabert_base(os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/base"), config)
    enable_mc_dropout(model, p=0.1)

    loader = torch.utils.data.DataLoader(collate_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    rows = []
    idx = 0
    for batch in loader:
        labels_key = "labels" if "labels" in batch else "label"
        labels = batch.pop(labels_key).to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        logits_all = torch.stack([model(**inputs).logits for _ in range(K_MC)], dim=0).float().cpu()
        unc = compute_uncertainty(logits_all)
        metas = _extract_metadata(tokenized_ds, metadata_cols, idx, idx + labels.shape[0])
        for i in range(labels.shape[0]):
            rows.append({
                **metas[i], "method": "mc_dropout",
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                "pred": int(unc["predicted_class"][i]), "labels": int(labels[i].cpu()),
            })
        idx += labels.shape[0]
    del model
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def run_evidential(tokenized_ds, collate_ds, collator, metadata_cols):
    import json
    print("=== evidential ===")
    base, _ = load_base_arch()
    with open(os.path.join(EVID_CKPT, "evidential_config.json")) as f:
        saved_cfg = json.load(f)
    evid_cfg = EvidentialConfig(
        num_classes=2,
        evidence_activation=saved_cfg.get("evidence_activation", "softplus"),
        loss_type=saved_cfg.get("loss_type", "mse"),
        annealing_step=saved_cfg.get("annealing_step", 10),
    )
    wrapper = EvidentialWrapper(base, evid_cfg)
    model = HFEvidentialSeqClassifier(wrapper).to(DEVICE)
    model.load_state_dict(load_file(os.path.join(EVID_CKPT, "model.safetensors")), strict=True)
    model.eval()

    loader = torch.utils.data.DataLoader(collate_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    rows = []
    idx = 0
    num_classes = evid_cfg.num_classes
    for batch in loader:
        labels_key = "labels" if "labels" in batch else "label"
        labels = batch.pop(labels_key).to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**inputs)
        evidence = out["logits"].detach().cpu().float()
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S
        unc = compute_uncertainty(torch.log(probs.clamp_min(1e-12)).unsqueeze(0))
        vacuity = num_classes / S.squeeze(-1)
        metas = _extract_metadata(tokenized_ds, metadata_cols, idx, idx + labels.shape[0])
        for i in range(labels.shape[0]):
            rows.append({
                **metas[i], "method": "evidential",
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "U_epistemic": float(vacuity[i]),
                "pred": int(unc["predicted_class"][i]), "labels": int(labels[i].cpu()),
            })
        idx += labels.shape[0]
    del model, base
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def run_laplace(tokenized_ds, collate_ds, collator, metadata_cols):
    print("=== laplace ===")
    _, config = load_base_arch()
    tokenizer = load_tokenizer()
    ckpt = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/base")
    model = load_dnabert_base(ckpt, config)

    fit_ds_raw = load_local_dataset(LABEL_NOISE_R00_TRAIN)
    fit_tok, fit_collator = prep_for_trainer(fit_ds_raw, tokenizer, max_length=MODEL_MAX_LENGTH, metadata_cols=())
    fit_tok = fit_tok.remove_columns(["sequence"])

    lap_cfg = LaplaceConfig(classifier_attr="classifier", prior_precision=LAPLACE_PRIOR_PRECISION,
                             max_examples=LAPLACE_FIT_EXAMPLES)
    weight_var, bias_var, n_used = fit_diagonal_laplace(model, fit_tok, fit_collator, lap_cfg, batch_size=64)
    print(f"[laplace] fit on {n_used} examples")
    lap_model = LaplaceSeqClassifier(model, weight_var.to(DEVICE), bias_var.to(DEVICE),
                                      classifier_attr="classifier").to(DEVICE)
    lap_model.eval()

    loader = torch.utils.data.DataLoader(collate_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    rows = []
    idx = 0
    for batch in loader:
        labels_key = "labels" if "labels" in batch else "label"
        labels = batch.pop(labels_key).to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        lap_model(**inputs)
        logits_all = lap_model.sample_logits(n_samples=K_LAPLACE).cpu()
        unc = compute_uncertainty(logits_all)
        metas = _extract_metadata(tokenized_ds, metadata_cols, idx, idx + labels.shape[0])
        for i in range(labels.shape[0]):
            rows.append({
                **metas[i], "method": "laplace",
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                "pred": int(unc["predicted_class"][i]), "labels": int(labels[i].cpu()),
            })
        idx += labels.shape[0]
    del model, lap_model
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def run_dnabert_ensembles(tokenized_ds, collate_ds, collator, metadata_cols):
    print("=== ensemble_k5 / ensemble_k3 ===")
    _, config = load_base_arch()
    members = {
        s: load_dnabert_base(os.path.join(REPO_ROOT, f"checkpoints/seed_{s}/DNABERT2/label_noise_r00/base"), config)
        for s in ENSEMBLE_SEEDS_K5
    }

    loader = torch.utils.data.DataLoader(collate_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    rows_k5, rows_k3 = [], []
    idx = 0
    for batch in loader:
        labels_key = "labels" if "labels" in batch else "label"
        labels = batch.pop(labels_key).to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        logits_by_seed = {s: members[s](**inputs).logits.cpu() for s in ENSEMBLE_SEEDS_K5}
        logits_k5 = torch.stack([logits_by_seed[s] for s in ENSEMBLE_SEEDS_K5], dim=0)
        logits_k3 = torch.stack([logits_by_seed[s] for s in ENSEMBLE_SEEDS_K3], dim=0)
        unc5 = compute_uncertainty(logits_k5)
        unc3 = compute_uncertainty(logits_k3)
        metas = _extract_metadata(tokenized_ds, metadata_cols, idx, idx + labels.shape[0])
        for i in range(labels.shape[0]):
            rows_k5.append({
                **metas[i], "method": "ensemble_k5",
                "U_aleatoric": float(unc5["normalized_aleatoric_uncertainty"][i]),
                "U_epistemic": float(unc5["normalized_epistemic_uncertainty"][i]),
                "pred": int(unc5["predicted_class"][i]), "labels": int(labels[i].cpu()),
            })
            rows_k3.append({
                **metas[i], "method": "ensemble_k3",
                "U_aleatoric": float(unc3["normalized_aleatoric_uncertainty"][i]),
                "U_epistemic": float(unc3["normalized_epistemic_uncertainty"][i]),
                "pred": int(unc3["predicted_class"][i]), "labels": int(labels[i].cpu()),
            })
        idx += labels.shape[0]
    members.clear()
    torch.cuda.empty_cache()
    return rows_k5, rows_k3


def run_cnn_methods(alisim_df: pd.DataFrame):
    print("=== cnn_mc_dropout / cnn_ensemble ===")
    train_df = pd.read_csv(LABEL_NOISE_R00_TRAIN)
    X_train = one_hot_encode(train_df["sequence"].tolist())
    y_train = torch.tensor(train_df["label"].values, dtype=torch.long)
    X_eval = one_hot_encode(alisim_df["sequence"].tolist())
    labels = torch.tensor(alisim_df["labels"].values, dtype=torch.long)

    cnn_device = DEVICE

    model = train_cnn(X_train, y_train, seed=CNN_MCD_SEED, device=cnn_device, dropout_p=CNN_DROPOUT_P)
    enable_mc_dropout(model, p=CNN_DROPOUT_P)
    with torch.no_grad():
        logits_all_mcd = torch.stack(
            [batched_logits(model, X_eval, cnn_device) for _ in range(K_CNN_DROPOUT)], dim=0
        )
    unc_mcd = compute_uncertainty(logits_all_mcd)
    del model
    if cnn_device.startswith("cuda"):
        torch.cuda.empty_cache()

    members = [train_cnn(X_train, y_train, seed=s, device=cnn_device, dropout_p=CNN_DROPOUT_P)
               for s in CNN_ENSEMBLE_SEEDS]
    with torch.no_grad():
        logits_all_ens = torch.stack([batched_logits(m, X_eval, cnn_device) for m in members], dim=0)
    unc_ens = compute_uncertainty(logits_all_ens)
    del members
    if cnn_device.startswith("cuda"):
        torch.cuda.empty_cache()

    rows_mcd, rows_ens = [], []
    for i in range(len(alisim_df)):
        base_meta = {
            "anchor_id": alisim_df["anchor_id"].iloc[i],
            "branch_length": float(alisim_df["branch_length"].iloc[i]),
            "realized_identity_to_anchor": float(alisim_df["realized_identity_to_anchor"].iloc[i]),
            "original_label": int(alisim_df["original_label"].iloc[i]),
        }
        rows_mcd.append({
            **base_meta, "method": "cnn_mc_dropout",
            "U_aleatoric": float(unc_mcd["normalized_aleatoric_uncertainty"][i]),
            "U_epistemic": float(unc_mcd["normalized_epistemic_uncertainty"][i]),
            "pred": int(unc_mcd["predicted_class"][i]), "labels": int(labels[i]),
        })
        rows_ens.append({
            **base_meta, "method": "cnn_ensemble",
            "U_aleatoric": float(unc_ens["normalized_aleatoric_uncertainty"][i]),
            "U_epistemic": float(unc_ens["normalized_epistemic_uncertainty"][i]),
            "pred": int(unc_ens["predicted_class"][i]), "labels": int(labels[i]),
        })
    return rows_mcd, rows_ens


def run_rf(alisim_df: pd.DataFrame):
    print("=== rf_kmer ===")
    train_df = pd.read_csv(LABEL_NOISE_R00_TRAIN)
    vec = CountVectorizer(analyzer="char", ngram_range=(RF_K, RF_K), lowercase=False)
    X_train = vec.fit_transform(train_df["sequence"])
    rf = RandomForestClassifier(
        n_estimators=RF_N_TREES, random_state=RF_SEED, n_jobs=-1, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    )
    rf.fit(X_train, train_df["label"].values)

    X_eval = vec.transform(alisim_df["sequence"])
    eps = 1e-6
    probs = np.stack([tree.predict_proba(X_eval) for tree in rf.estimators_], axis=0)
    probs = np.clip(probs, eps, 1.0 - eps)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    logits_all = torch.from_numpy(np.log(probs)).float()
    unc = compute_uncertainty(logits_all)

    rows = []
    for i in range(len(alisim_df)):
        rows.append({
            "anchor_id": alisim_df["anchor_id"].iloc[i],
            "branch_length": float(alisim_df["branch_length"].iloc[i]),
            "realized_identity_to_anchor": float(alisim_df["realized_identity_to_anchor"].iloc[i]),
            "original_label": int(alisim_df["original_label"].iloc[i]),
            "method": "rf_kmer",
            "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
            "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
            "pred": int(unc["predicted_class"][i]), "labels": int(alisim_df["labels"].iloc[i]),
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = load_tokenizer()
    tokenized_ds, collate_ds, collator, metadata_cols = load_alisim_dataset_for_dnabert(tokenizer)
    print(f"Loaded {len(tokenized_ds)} rows from {DATA_PATH}; metadata cols: {metadata_cols}")
    alisim_df = pd.read_csv(DATA_PATH)

    new_rows = []
    new_rows += run_mc_dropout(tokenized_ds, collate_ds, collator, metadata_cols)
    new_rows += run_evidential(tokenized_ds, collate_ds, collator, metadata_cols)
    new_rows += run_laplace(tokenized_ds, collate_ds, collator, metadata_cols)
    rows_k5, rows_k3 = run_dnabert_ensembles(tokenized_ds, collate_ds, collator, metadata_cols)
    new_rows += rows_k5
    new_rows += rows_k3
    rows_mcd, rows_ens = run_cnn_methods(alisim_df)
    new_rows += rows_mcd
    new_rows += rows_ens
    new_rows += run_rf(alisim_df)

    new_df = pd.DataFrame(new_rows)
    new_df["correct"] = (new_df["pred"] == new_df["labels"]).astype(int)
    new_df = new_df[OUT_COLS]

    out_path = os.path.join(OUT_DIR, "per_example_other_methods_extended_only.csv")
    new_df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(new_df)} rows, {new_df['method'].nunique()} methods)")


if __name__ == "__main__":
    main()
