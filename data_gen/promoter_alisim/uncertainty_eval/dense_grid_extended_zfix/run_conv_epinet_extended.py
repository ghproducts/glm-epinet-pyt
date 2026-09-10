#!/usr/bin/env python3
"""Re-run `conv_epinet` on the FULL 19-point promoter_alisim branch-length
grid (16-point dense grid + 3-point extension {5.0, 8.0, 12.0}) under the
NEWLY RETRAINED `epinet_zfix` checkpoint
(`checkpoints/seed_1/DNABERT2/label_noise_r00/epinet_zfix`), which was
trained under the per-example-z fix (fb38241) from the start -- unlike the
original `epinet` checkpoint, which predates that fix and only ever had it
applied as an eval-time monkeypatch (see
data_gen/promoter_alisim/README.md's "Epinet fix used" section and
`.docs`/critique_independent.md Finding #1.1).

This is a FULL re-run (all 19 branch lengths), not just the 3 new points,
because the checkpoint itself changed -- every existing conv_epinet number
in `uncertainty_eval/dense_grid/` was computed with the stale checkpoint and
needs replacing, not just extending.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=0 /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \\
        data_gen/promoter_alisim/uncertainty_eval/dense_grid_extended_zfix/run_conv_epinet_extended.py
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT_FOR_IMPORT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _REPO_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_IMPORT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file

from nn_proj.common.datasets import load_local_dataset, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithConvPrior
from nn_proj.models.epinet.feature_fns import NT_feature_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"

REPO_ROOT = "/scratch/home/glh52/glm-epinet-pyt"
DENSE_CSV = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/csv_data/promoter_alisim_dense.csv")
EXTENDED_CSV = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/csv_data/promoter_alisim_extended.csv")
OUT_DIR = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/uncertainty_eval/dense_grid_extended_zfix")

BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/base")
# The NEW checkpoint -- trained under the fixed per-example-z code, not the
# stale pre-fix checkpoint every prior conv_epinet number used.
EPI_CKPT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/epinet_zfix")

MODEL_MAX_LENGTH = 75  # DNABERT2 tokens_per_base=0.25 x 300bp, per CLAUDE.md
BATCH_SIZE = 32
K_EPI = 10  # matches original K for conv_epinet/mc_dropout in this project
METADATA_COLS = ["anchor_id", "original_label", "branch_length", "realized_identity_to_anchor"]
OUT_COLS = ["anchor_id", "branch_length", "realized_identity_to_anchor", "original_label",
            "method", "U_aleatoric", "U_epistemic", "pred", "labels", "correct"]


def load_combined_dataset():
    dense = pd.read_csv(DENSE_CSV)
    ext = pd.read_csv(EXTENDED_CSV)
    combined = pd.concat([dense, ext], ignore_index=True)
    print(f"[info] combined grid: {len(dense)} (dense) + {len(ext)} (extended) = {len(combined)} rows, "
          f"{combined['branch_length'].nunique()} distinct branch lengths")
    return combined


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


def _extract_metadata(tokenized_ds, metadata_cols, idx_start, idx_end):
    rows = []
    for i in range(idx_start, idx_end):
        row = {}
        for c in metadata_cols:
            v = tokenized_ds[c][i]
            row[c] = v.item() if isinstance(v, torch.Tensor) else v
        rows.append(row)
    return rows


@torch.no_grad()
def run_conv_epinet(tokenized_ds, collate_ds, collator, metadata_cols):
    print("=== conv_epinet (epinet_zfix checkpoint) ===")
    base, config = load_base_arch()
    epi_cfg = EpinetConfig(num_classes=2, include_inputs=True, vocab_size=config.vocab_size)
    wrapper = EpinetWrapper(base, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=K_EPI).to(DEVICE)

    dummy = collator([collate_ds[i] for i in range(2)])
    dummy = {k: v.to(DEVICE) for k, v in dummy.items() if k not in ("labels", "label")}
    _ = model(**dummy)

    model.load_state_dict(load_file(os.path.join(EPI_CKPT, "model.safetensors")), strict=True)
    model.eval()

    loader = torch.utils.data.DataLoader(collate_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    rows = []
    idx = 0
    for batch in loader:
        labels_key = "labels" if "labels" in batch else "label"
        labels = batch.pop(labels_key).to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        logits_all = model.wrapper(inputs, n_index_samples=K_EPI, return_all=True).cpu()  # [K,B,C]
        unc = compute_uncertainty(logits_all)
        metas = _extract_metadata(tokenized_ds, metadata_cols, idx, idx + labels.shape[0])
        for i in range(labels.shape[0]):
            rows.append({
                **metas[i],
                "method": "conv_epinet",
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                "pred": int(unc["predicted_class"][i]),
                "labels": int(labels[i].cpu()),
            })
        idx += labels.shape[0]
    del model, base
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def run_base_and_scaled(sequences, labels_arr, metas, T):
    print("=== base / base_scaled (full 19-point grid) ===")
    model, config = load_base_arch()
    model.load_state_dict(load_file(os.path.join(BASE_CKPT, "model.safetensors")), strict=True)
    model = model.to(DEVICE).eval()
    tokenizer = load_tokenizer()

    logits_out = []
    for i in range(0, len(sequences), 64):
        batch = sequences[i:i + 64]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        logits_out.append(model(**enc).logits.float().cpu())
    logits = torch.cat(logits_out, dim=0)
    del model
    torch.cuda.empty_cache()

    probs = F.softmax(logits, dim=-1).numpy()
    p1 = probs[:, 1]
    probs_scaled = F.softmax(logits / T, dim=-1).numpy()
    p1_scaled = probs_scaled[:, 1]
    pred = (p1 >= 0.5).astype(int)
    pred_scaled = (p1_scaled >= 0.5).astype(int)

    rows = []
    for i in range(len(sequences)):
        rows.append({
            **metas[i],
            "labels": int(labels_arr[i]),
            "p1": float(p1[i]), "pred": int(pred[i]),
            "p1_scaled": float(p1_scaled[i]), "pred_scaled": int(pred_scaled[i]),
        })
    return rows


def fit_temperature_on_train():
    """Same procedure as data_gen/uq_metrics_followup/dnabert_only/dnabert_base_and_scaled.py:
    LBFGS/NLL on a held-out 10% stratified split of promoter_all's training data."""
    import torch.nn as nn
    from datasets import load_dataset, ClassLabel

    tokenizer = load_tokenizer()
    model, config = load_base_arch()
    model.load_state_dict(load_file(os.path.join(BASE_CKPT, "model.safetensors")), strict=True)
    model = model.to(DEVICE).eval()

    ds_all = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", split="train")
    ds = ds_all.filter(lambda ex: ex["task"] == "promoter_all")
    if "labels" not in ds.column_names and "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")
    if not isinstance(ds.features["labels"], ClassLabel):
        ds = ds.class_encode_column("labels")
    split = ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="labels")
    val = split["test"]

    logits_out = []
    seqs = list(val["sequence"])
    for i in range(0, len(seqs), 64):
        enc = tokenizer(seqs[i:i + 64], truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits_out.append(model(**enc).logits.float().cpu())
    val_logits = torch.cat(logits_out, dim=0)
    val_labels = torch.tensor(val["labels"])
    del model
    torch.cuda.empty_cache()

    log_T = torch.zeros((), requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = loss_fn(val_logits / T, val_labels)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_T).detach().cpu().item())
    print(f"[fit T] T = {T:.6f} (n_val={len(val)})")
    return T


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    combined = load_combined_dataset()

    tokenizer = load_tokenizer()
    # load_local_dataset expects a path; write the combined frame to a temp csv
    tmp_csv = os.path.join(OUT_DIR, "_combined_19pt_grid.csv")
    combined.to_csv(tmp_csv, index=False)
    ds = load_local_dataset(tmp_csv, encode_labels=True)
    metadata_cols = [c for c in METADATA_COLS if c in ds.column_names]
    tokenized, collator = prep_for_trainer(ds, tokenizer, max_length=MODEL_MAX_LENGTH, metadata_cols=metadata_cols)
    collate_ds = tokenized.remove_columns(["sequence"] + metadata_cols)
    print(f"Loaded {len(tokenized)} rows; metadata cols: {metadata_cols}")

    epi_rows = run_conv_epinet(tokenized, collate_ds, collator, metadata_cols)
    epi_df = pd.DataFrame(epi_rows)
    epi_df["correct"] = (epi_df["pred"] == epi_df["labels"]).astype(int)
    epi_df = epi_df[OUT_COLS]
    epi_out_path = os.path.join(OUT_DIR, "per_example_conv_epinet_zfix_19pt.csv")
    epi_df.to_csv(epi_out_path, index=False)
    print(f"wrote {epi_out_path} ({len(epi_df)} rows)")

    T = fit_temperature_on_train()
    metas_full = _extract_metadata(tokenized, metadata_cols, 0, len(tokenized))
    labels_full = [tokenized["labels"][i].item() if hasattr(tokenized["labels"][i], "item") else tokenized["labels"][i]
                   for i in range(len(tokenized))]
    base_rows = run_base_and_scaled(list(combined["sequence"]), labels_full, metas_full, T)
    base_df = pd.DataFrame(base_rows)
    base_out_path = os.path.join(OUT_DIR, "per_example_base_basescaled_19pt.csv")
    base_df.to_csv(base_out_path, index=False)
    with open(os.path.join(OUT_DIR, "fitted_temperature.txt"), "w") as f:
        f.write(f"T={T}\ndata_seed=42\ncheckpoint={BASE_CKPT}\n")
    print(f"wrote {base_out_path} ({len(base_df)} rows), T={T}")

    os.remove(tmp_csv)


if __name__ == "__main__":
    main()
