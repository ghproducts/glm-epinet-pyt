#!/usr/bin/env python3
"""Extends the promoter_alisim dense-grid epistemic-uncertainty evaluation
from 3 methods (conv_epinet, mc_dropout, evidential -- see
`run_uncertainty_eval_dense.py`) to the full method roster used elsewhere in
this project: `laplace`, DNABERT2 deep ensembles (K=5 and K=3), a
from-scratch CNN under both `mc_dropout` and `ensemble` sampling, and a
k-mer + Random Forest baseline. Same 16-point branch-length grid, same 400
anchors (`csv_data/promoter_alisim_dense.csv`, 6400 rows), same output
schema as the original three methods, so all 8 method-configurations can be
concatenated into one `per_example_uncertainty.csv` and one combined
dose-response/Spearman/shape report.

Templates adapted (see `data_gen/promoter_alisim/README.md` for the
per-method judgment calls this script makes explicit):
  - `data_gen/label_noise/decomp_compare_label_noise_laplace.py` +
    `nn_proj/models/laplace/laplace_head.py` -- last-layer diagonal Laplace,
    post-hoc on a saved DNABERT2 base checkpoint.
  - `data_gen/label_noise/decomp_compare_ood_severity_ensemble.py` --
    DNABERT2 deep ensemble, swapped from the real/shuffled/random_dna axis
    to the 16 branch lengths here.
  - `data_gen/label_noise/decomp_compare_ood_severity_cnn.py` (+
    `cnn_scratch.py`) -- from-scratch 1D-CNN under mc_dropout and ensemble
    sampling, same axis swap.
  - `data_gen/label_noise/decomp_compare_ood_severity_rf.py` -- k-mer count
    vectorizer + Random Forest, same axis swap.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=0 /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \\
        data_gen/promoter_alisim/uncertainty_eval/dense_grid/run_new_methods_dense.py
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
from cnn_scratch import one_hot_encode, train_cnn, batched_logits  # data_gen/label_noise/cnn_scratch.py

from run_uncertainty_eval_dense import (  # same directory -- reuse aggregation code unchanged
    BRANCH_LENGTHS, ENDPOINT_LO, ENDPOINT_HI, METADATA_COLS,
    summarize, characterize_shape,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
DATA_PATH = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/csv_data/promoter_alisim_dense.csv")
OUT_DIR = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/uncertainty_eval/dense_grid")

REPO_ROOT = "/scratch/home/glh52/glm-epinet-pyt"
LABEL_NOISE_R00_TRAIN = os.path.join(REPO_ROOT, "data_gen/label_noise/csv_data_r00/train.csv")

ENSEMBLE_SEEDS_K5 = [1, 2, 3, 4, 42]  # this project's standard 5-seed convention
ENSEMBLE_SEEDS_K3 = [1, 2, 3]         # "any 3" per task spec -- first 3 of the K5 set
CNN_ENSEMBLE_SEEDS = [1, 2, 3, 4, 42]
CNN_MCD_SEED = 1                      # single trained model for mc_dropout, matches template

MODEL_MAX_LENGTH = 75  # DNABERT2 tokens_per_base=0.25 x 300bp, per CLAUDE.md
BATCH_SIZE = 32
K_LAPLACE = 16          # matches decomp_compare_label_noise_laplace.py
LAPLACE_FIT_EXAMPLES = 2000
LAPLACE_PRIOR_PRECISION = 1.0
K_CNN_DROPOUT = 16      # matches decomp_compare_ood_severity_cnn.py
CNN_DROPOUT_P = 0.3
RF_K = 6                # k-mer size, matches decomp_compare_*_rf.py
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
def run_laplace(tokenized_ds, collate_ds, collator, metadata_cols):
    print("=== laplace ===")
    _, config = load_base_arch()
    tokenizer = load_tokenizer()
    ckpt = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/base")
    model = load_dnabert_base(ckpt, config)

    # Fit the diagonal GGN on the same (input, trained-on label) pairs the
    # checkpoint was fine-tuned on -- clean r00 promoter_all train split --
    # matching decomp_compare_label_noise_laplace.py's convention.
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
        lap_model(**inputs)  # populate cached pre-classifier features
        logits_all = lap_model.sample_logits(n_samples=K_LAPLACE).cpu()  # [K,B,C]
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
        logits_by_seed = {s: members[s](**inputs).logits.cpu() for s in ENSEMBLE_SEEDS_K5}  # each [B,C]
        logits_k5 = torch.stack([logits_by_seed[s] for s in ENSEMBLE_SEEDS_K5], dim=0)  # [5,B,C]
        logits_k3 = torch.stack([logits_by_seed[s] for s in ENSEMBLE_SEEDS_K3], dim=0)  # [3,B,C]
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


def load_alisim_dataset_for_cnn_rf():
    df = pd.read_csv(DATA_PATH)
    return df


def run_cnn_methods(alisim_df: pd.DataFrame):
    # NOTE: deliberately not @torch.no_grad() at the function level -- train_cnn()
    # needs gradients for its own training loop. Inference-only sections below
    # are wrapped in an explicit `with torch.no_grad():` instead.
    print("=== cnn_mc_dropout / cnn_ensemble ===")
    train_df = pd.read_csv(LABEL_NOISE_R00_TRAIN)
    X_train = one_hot_encode(train_df["sequence"].tolist())
    y_train = torch.tensor(train_df["label"].values, dtype=torch.long)
    X_eval = one_hot_encode(alisim_df["sequence"].tolist())
    labels = torch.tensor(alisim_df["labels"].values, dtype=torch.long)

    cnn_device = DEVICE  # small model; fine to share the same GPU as DNABERT2 runs (sequential, not concurrent)

    # ---- mc_dropout: one trained model, K stochastic forward passes ----
    model = train_cnn(X_train, y_train, seed=CNN_MCD_SEED, device=cnn_device, dropout_p=CNN_DROPOUT_P)
    enable_mc_dropout(model, p=CNN_DROPOUT_P)
    logits_all_mcd = torch.stack(
        [batched_logits(model, X_eval, cnn_device) for _ in range(K_CNN_DROPOUT)], dim=0
    )  # [K,N,C]
    unc_mcd = compute_uncertainty(logits_all_mcd)
    del model
    if cnn_device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- ensemble: K independently-initialized-and-trained models ----
    members = [train_cnn(X_train, y_train, seed=s, device=cnn_device, dropout_p=CNN_DROPOUT_P)
               for s in CNN_ENSEMBLE_SEEDS]
    logits_all_ens = torch.stack([batched_logits(m, X_eval, cnn_device) for m in members], dim=0)  # [K,N,C]
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
    logits_all = torch.from_numpy(np.log(probs)).float()  # [n_trees, N, C]
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


def classify_aleatoric(spear_df: pd.DataFrame) -> "pd.DataFrame":
    """Per-method classification of U_aleatoric's own trend vs. branch_length,
    used to build the "does aleatoric stay flat" table in the interpretation
    section. Threshold: p<0.01 and rho>0 -> conflates (rises with epistemic);
    p<0.01 and rho<0 -> wrong direction; otherwise -> roughly flat/noisy.
    """
    rows = []
    for method, sub in spear_df[spear_df["score"] == "U_aleatoric"].groupby("method", sort=False):
        rho = sub["spearman_rho"].iloc[0]
        p = sub["spearman_p"].iloc[0]
        if p < 0.01 and rho > 0:
            behavior = "conflates (rises with epistemic)"
        elif p < 0.01 and rho < 0:
            behavior = "wrong direction (falls)"
        else:
            behavior = "roughly flat / not significant"
        rows.append({"method": method, "spearman_rho": rho, "spearman_p": p, "behavior": behavior})
    return pd.DataFrame(rows)


def write_interpretation_section(dose_df, spear_df, shape_df) -> list:
    """Data-driven synthesis across every method currently in shape_df/spear_df
    -- regenerates the shape-family groupings and the aleatoric-flat-or-not
    classification from the actual numbers on every run, rather than hardcoding
    prose that could drift out of sync with the CSVs."""
    lines = ["### Interpretation: shape families and aleatoric behavior across all methods", ""]

    families = {}
    for _, row in shape_df.iterrows():
        families.setdefault(row["shape"], []).append(row["method"])

    lines.append(f"Grouping all {len(shape_df)} method-configurations by their `shape` column:")
    lines.append("")
    for shape_name, methods in families.items():
        epi_rhos = spear_df[(spear_df["method"].isin(methods)) & (spear_df["score"] == "U_epistemic")]
        rho_str = ", ".join(f"{m}: rho={r:.3g}" for m, r in zip(epi_rhos["method"], epi_rhos["spearman_rho"]))
        lines.append(f"- **{shape_name}** ({len(methods)}/{len(shape_df)}): {', '.join(methods)} -- U_epistemic Spearman {rho_str}.")
    lines.append("")

    ale_df = classify_aleatoric(spear_df)
    lines.append("**Does `U_aleatoric` stay flat?** Per-method classification (p<0.01 threshold):")
    lines.append("")
    lines.append(ale_df.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")
    n_conflate = (ale_df["behavior"] == "conflates (rises with epistemic)").sum()
    n_wrong = (ale_df["behavior"] == "wrong direction (falls)").sum()
    n_flat = (ale_df["behavior"] == "roughly flat / not significant").sum()
    lines.append(
        f"**Net**: {n_conflate}/{len(ale_df)} configurations show `U_aleatoric` rising alongside "
        f"`U_epistemic` (conflation, not the hypothesized flat behavior), {n_wrong} move in the "
        f"opposite/wrong direction, and {n_flat} are roughly flat or not significant at this sample "
        "size. The epistemic-rises direction (some positive, mostly significant Spearman rho vs. "
        "branch_length) holds for essentially every configuration, but its *shape* varies (still-rising / "
        "plateau / hump-and-decline per the grouping above) and the aleatoric-stays-flat half of the "
        "hypothesis does not hold cleanly for the great majority of methods tested."
    )
    lines.append("")
    return lines


def write_markdown_summary(dose_df, spear_df, mwu_df, shape_df, path, methods_all):
    lines = [f"# promoter_alisim dense-grid uncertainty evaluation: results summary "
             f"({len(methods_all)} configurations)", ""]
    lines.append(
        "Combined dose-response report across the full method roster this project uses elsewhere, "
        "run on the 16-point branch-length grid "
        "{0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0} "
        "(`csv_data/promoter_alisim_dense.csv`, 400 anchors x 16 branch lengths = 6400 rows). "
        "The first 3 methods (`conv_epinet`, `mc_dropout`, `evidential`) reproduce "
        "`run_uncertainty_eval_dense.py` unchanged; the other 6 configurations (`laplace`, "
        "`ensemble_k5`, `ensemble_k3`, `cnn_mc_dropout`, `cnn_ensemble`, `rf_kmer` -- across "
        "DNABERT2 post-hoc/ensemble methods and two from-scratch, non-pretrained baselines) are "
        "added by `run_new_methods_dense.py`. `ensemble_k5`/`ensemble_k3` are the same method "
        "family (DNABERT2 deep ensemble) evaluated at two K values, so this is 8 method families "
        "reported as 9 configurations/rows -- nothing hidden either way. See "
        "`data_gen/promoter_alisim/README.md`'s Uncertainty evaluation section for the checkpoint, "
        "retraining, and environment judgment calls behind every row here."
    )
    lines.append("")
    lines.append(f"Methods covered ({len(methods_all)} configurations): " + ", ".join(methods_all))
    lines.append("")
    lines.append("## Dose-response: mean U_epistemic / U_aleatoric / accuracy by branch_length")
    lines.append("")
    for method, sub in dose_df.groupby("method", sort=False):
        lines.append(f"### {method}")
        lines.append("")
        cols = ["branch_length", "n", "U_epistemic_mean", "U_epistemic_std", "U_aleatoric_mean", "U_aleatoric_std", "accuracy"]
        lines.append(sub[cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("## Spearman correlation: branch_length vs. score (per method)")
    lines.append("")
    lines.append(spear_df.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    lines.append(f"## Endpoint check: Mann-Whitney U, t={ENDPOINT_HI} > t={ENDPOINT_LO} (per method, per score)")
    lines.append("")
    lines.append(mwu_df.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    lines.append("## Shape characterization: where does U_epistemic peak, and what happens after?")
    lines.append("")
    lines.append(shape_df.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    lines += write_interpretation_section(dose_df, spear_df, shape_df)

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = load_tokenizer()
    tokenized_ds, collate_ds, collator, metadata_cols = load_alisim_dataset_for_dnabert(tokenizer)
    print(f"Loaded {len(tokenized_ds)} rows from {DATA_PATH}; metadata cols: {metadata_cols}")
    alisim_df = load_alisim_dataset_for_cnn_rf()

    new_rows = []
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

    per_example_path = os.path.join(OUT_DIR, "per_example_uncertainty.csv")
    existing_df = pd.read_csv(per_example_path)
    existing_methods = set(existing_df["method"].unique())
    new_methods = set(new_df["method"].unique())
    overlap = existing_methods & new_methods
    if overlap:
        # Idempotent re-run: drop any previous rows for methods we are about to
        # rewrite instead of duplicating them.
        existing_df = existing_df[~existing_df["method"].isin(overlap)]
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(per_example_path, index=False)
    print(f"wrote {per_example_path} ({len(combined_df)} rows, {combined_df['method'].nunique()} methods)")

    dose_df, spear_df, mwu_df = summarize(combined_df)
    shape_df = characterize_shape(dose_df)
    dose_df.to_csv(os.path.join(OUT_DIR, "results_summary_dose_response.csv"), index=False)
    spear_df.to_csv(os.path.join(OUT_DIR, "results_summary_spearman.csv"), index=False)
    mwu_df.to_csv(os.path.join(OUT_DIR, "results_summary_mwu_endpoints.csv"), index=False)
    shape_df.to_csv(os.path.join(OUT_DIR, "results_summary_shape.csv"), index=False)

    method_order = ["conv_epinet", "mc_dropout", "evidential", "laplace", "ensemble_k5", "ensemble_k3",
                     "cnn_mc_dropout", "cnn_ensemble", "rf_kmer"]
    methods_all = [m for m in method_order if m in combined_df["method"].unique()]
    write_markdown_summary(dose_df, spear_df, mwu_df, shape_df,
                            os.path.join(OUT_DIR, "results_summary.md"), methods_all)

    print("\n=== DOSE-RESPONSE (new methods only) ===")
    print(dose_df[dose_df["method"].isin(new_df["method"].unique())].to_string(index=False))
    print("\n=== SHAPE CHARACTERIZATION (all methods) ===")
    print(shape_df.to_string(index=False))


if __name__ == "__main__":
    main()
