#!/usr/bin/env python3
"""Denser follow-up to `run_uncertainty_eval.py`: does U_epistemic's
non-monotonic shape (weak rise then decline, seen in the original 8-point
branch-length grid for mc_dropout/evidential) resolve into a clearer pattern
at higher resolution, and does it plateau at large branch length (consistent
with the substitution process saturating at its own stationary distribution)?

Runs the identical three UQ methods (mc_dropout, conv_epinet, evidential),
same checkpoints, same K, same feature functions, same epinet fix, as
`run_uncertainty_eval.py` -- only the input CSV (the 16-point dense grid,
`csv_data/promoter_alisim_dense.csv`, 6400 rows) and output directory
(`uncertainty_eval/dense_grid/`) differ. See
`data_gen/promoter_alisim/README.md` for the dense-grid rationale and
`data_gen/promoter_alisim/uncertainty_eval/run_uncertainty_eval.py`'s own
docstring for method/checkpoint/environment details, all unchanged here.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=0 /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \\
        data_gen/promoter_alisim/uncertainty_eval/dense_grid/run_uncertainty_eval_dense.py
"""
from __future__ import annotations

import json
import os
import sys

# Make the repo root importable regardless of cwd -- this file lives one
# directory deeper (uncertainty_eval/dense_grid/) than the original script.
_REPO_ROOT_FOR_IMPORT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if _REPO_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_IMPORT)

import numpy as np
import pandas as pd
import torch
import transformers
from safetensors.torch import load_file
from scipy.stats import mannwhitneyu, spearmanr

from nn_proj.common.datasets import load_local_dataset, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithConvPrior
from nn_proj.models.epinet.feature_fns import NT_feature_fn
from nn_proj.models.evidential import EvidentialConfig, EvidentialWrapper, HFEvidentialSeqClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
DATA_PATH = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/csv_data/promoter_alisim_dense.csv")
OUT_DIR = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/uncertainty_eval/dense_grid")

REPO_ROOT = "/scratch/home/glh52/glm-epinet-pyt"
BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/base")
EPI_CKPT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/epinet")
EVID_CKPT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/evidential")

MODEL_MAX_LENGTH = 75  # DNABERT2 tokens_per_base=0.25 x 300bp, per CLAUDE.md
BATCH_SIZE = 32
K_MC = 10
K_EPI = 10
BRANCH_LENGTHS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
ENDPOINT_LO, ENDPOINT_HI = 0.0, 3.0
METADATA_COLS = ["anchor_id", "original_label", "branch_length", "realized_identity_to_anchor"]


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


def load_alisim_dataset(tokenizer):
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


@torch.no_grad()
def run_mc_dropout(tokenized_ds, collate_ds, collator, metadata_cols):
    print("=== mc_dropout ===")
    model, _ = load_base_arch()
    model.load_state_dict(load_file(os.path.join(BASE_CKPT, "model.safetensors")), strict=True)
    model = model.to(DEVICE)
    enable_mc_dropout(model, p=0.1)

    loader = torch.utils.data.DataLoader(
        collate_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
    )
    rows = []
    idx = 0
    for batch in loader:
        labels_key = "labels" if "labels" in batch else "label"
        labels = batch.pop(labels_key).to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        logits_all = torch.stack([model(**inputs).logits for _ in range(K_MC)], dim=0).cpu()  # [K,B,C]
        unc = compute_uncertainty(logits_all)
        metas = _extract_metadata(tokenized_ds, metadata_cols, idx, idx + labels.shape[0])
        for i in range(labels.shape[0]):
            rows.append({
                **metas[i],
                "method": "mc_dropout",
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                "pred": int(unc["predicted_class"][i]),
                "labels": int(labels[i].cpu()),
            })
        idx += labels.shape[0]
    del model
    torch.cuda.empty_cache()
    return rows


@torch.no_grad()
def run_conv_epinet(tokenized_ds, collate_ds, collator, metadata_cols):
    print("=== conv_epinet ===")
    base, config = load_base_arch()
    epi_cfg = EpinetConfig(num_classes=2, include_inputs=True, vocab_size=config.vocab_size)
    wrapper = EpinetWrapper(base, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=K_EPI).to(DEVICE)

    dummy = collator([collate_ds[i] for i in range(2)])
    dummy = {k: v.to(DEVICE) for k, v in dummy.items() if k not in ("labels", "label")}
    _ = model(**dummy)

    model.load_state_dict(load_file(os.path.join(EPI_CKPT, "model.safetensors")), strict=True)
    model.eval()

    loader = torch.utils.data.DataLoader(
        collate_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
    )
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
def run_evidential(tokenized_ds, collate_ds, collator, metadata_cols):
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

    loader = torch.utils.data.DataLoader(
        collate_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
    )
    rows = []
    idx = 0
    num_classes = evid_cfg.num_classes
    for batch in loader:
        labels_key = "labels" if "labels" in batch else "label"
        labels = batch.pop(labels_key).to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**inputs)
        evidence = out["logits"].detach().cpu().float()  # [B,C]
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)  # [B,1]
        probs = alpha / S
        unc = compute_uncertainty(torch.log(probs.clamp_min(1e-12)).unsqueeze(0))  # [1,B,C] -> K=1 stack
        vacuity = num_classes / S.squeeze(-1)  # [B]

        metas = _extract_metadata(tokenized_ds, metadata_cols, idx, idx + labels.shape[0])
        for i in range(labels.shape[0]):
            rows.append({
                **metas[i],
                "method": "evidential",
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "U_epistemic": float(vacuity[i]),
                "pred": int(unc["predicted_class"][i]),
                "labels": int(labels[i].cpu()),
            })
        idx += labels.shape[0]
    del model, base
    torch.cuda.empty_cache()
    return rows


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (dose_response_table, spearman_table, mwu_endpoint_table)."""
    dose_rows, spear_rows, mwu_rows = [], [], []

    for method, sub in df.groupby("method"):
        for bl, g in sub.groupby("branch_length"):
            dose_rows.append({
                "method": method, "branch_length": bl, "n": len(g),
                "U_epistemic_mean": g["U_epistemic"].mean(), "U_epistemic_std": g["U_epistemic"].std(),
                "U_aleatoric_mean": g["U_aleatoric"].mean(), "U_aleatoric_std": g["U_aleatoric"].std(),
                "accuracy": (g["pred"] == g["labels"]).mean(),
            })

        for score in ["U_epistemic", "U_aleatoric"]:
            rho, p = spearmanr(sub["branch_length"], sub[score])
            spear_rows.append({"method": method, "score": score, "spearman_rho": rho, "spearman_p": p, "n": len(sub)})

        lo = sub[sub["branch_length"] == ENDPOINT_LO]
        hi = sub[sub["branch_length"] == ENDPOINT_HI]
        for score in ["U_epistemic", "U_aleatoric"]:
            u, p = mannwhitneyu(hi[score], lo[score], alternative="greater")
            mwu_rows.append({
                "method": method, "score": score, "comparison": f"t={ENDPOINT_HI} > t={ENDPOINT_LO}",
                "U": u, "p": p, "n_lo": len(lo), "n_hi": len(hi),
                "mean_lo": lo[score].mean(), "mean_hi": hi[score].mean(),
            })

    dose_df = pd.DataFrame(dose_rows).sort_values(["method", "branch_length"]).reset_index(drop=True)
    spear_df = pd.DataFrame(spear_rows).reset_index(drop=True)
    mwu_df = pd.DataFrame(mwu_rows).reset_index(drop=True)
    return dose_df, spear_df, mwu_df


def characterize_shape(dose_df: pd.DataFrame) -> pd.DataFrame:
    """For each method, find the branch_length at which U_epistemic peaks,
    and describe decline/plateau/still-rising behavior from peak to the
    largest branch length, and rise from t=0 baseline to peak."""
    rows = []
    for method, sub in dose_df.groupby("method"):
        sub = sub.sort_values("branch_length").reset_index(drop=True)
        baseline = sub.iloc[0]["U_epistemic_mean"]
        peak_idx = sub["U_epistemic_mean"].idxmax()
        peak_bl = sub.loc[peak_idx, "branch_length"]
        peak_val = sub.loc[peak_idx, "U_epistemic_mean"]
        last_bl = sub.iloc[-1]["branch_length"]
        last_val = sub.iloc[-1]["U_epistemic_mean"]

        pct_decline_from_peak = 100.0 * (peak_val - last_val) / peak_val if peak_val != 0 else float("nan")
        pct_above_baseline_at_end = 100.0 * (last_val - baseline) / baseline if baseline != 0 else float("nan")
        pct_rise_baseline_to_peak = 100.0 * (peak_val - baseline) / baseline if baseline != 0 else float("nan")

        if peak_idx == len(sub) - 1:
            shape = "still rising at largest branch_length (no interior peak)"
        elif last_val >= peak_val * 0.97:
            shape = "plateaus near peak (within 3%) after peaking"
        else:
            shape = "declines after peaking"

        rows.append({
            "method": method,
            "baseline_bl0_U_epistemic": baseline,
            "peak_branch_length": peak_bl,
            "peak_U_epistemic": peak_val,
            "pct_rise_baseline_to_peak": pct_rise_baseline_to_peak,
            "last_branch_length": last_bl,
            "last_U_epistemic": last_val,
            "pct_decline_from_peak_to_last": pct_decline_from_peak,
            "pct_above_baseline_at_last": pct_above_baseline_at_end,
            "shape": shape,
        })
    return pd.DataFrame(rows)


def write_markdown_summary(dose_df, spear_df, mwu_df, shape_df, path):
    lines = ["# promoter_alisim dense-grid uncertainty evaluation: results summary", ""]
    lines.append(
        "Denser follow-up to `data_gen/promoter_alisim/uncertainty_eval/results_summary.md` "
        "(the original 8-point grid). Same three UQ methods, same checkpoints, same K, "
        "run on the 16-point branch-length grid "
        "{0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0} "
        "(`csv_data/promoter_alisim_dense.csv`, same 400 anchors). Motivation: the coarse grid "
        "showed a weak, non-monotonic epistemic signal (mc_dropout, evidential) peaking around "
        "branch_length 0.2-0.4 and declining toward 1.5 -- this run adds resolution in that "
        "region and extends to 2.0/3.0 to check for a saturation plateau."
    )
    lines.append("")
    lines.append("## Dose-response: mean U_epistemic / U_aleatoric / accuracy by branch_length")
    lines.append("")
    for method, sub in dose_df.groupby("method"):
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

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = load_tokenizer()
    tokenized_ds, collate_ds, collator, metadata_cols = load_alisim_dataset(tokenizer)
    print(f"Loaded {len(tokenized_ds)} rows from {DATA_PATH}; metadata cols: {metadata_cols}")

    all_rows = []
    all_rows += run_mc_dropout(tokenized_ds, collate_ds, collator, metadata_cols)
    all_rows += run_conv_epinet(tokenized_ds, collate_ds, collator, metadata_cols)
    all_rows += run_evidential(tokenized_ds, collate_ds, collator, metadata_cols)

    df = pd.DataFrame(all_rows)
    df["correct"] = (df["pred"] == df["labels"]).astype(int)
    per_example_path = os.path.join(OUT_DIR, "per_example_uncertainty.csv")
    out_cols = ["anchor_id", "branch_length", "realized_identity_to_anchor", "original_label",
                "method", "U_aleatoric", "U_epistemic", "pred", "labels", "correct"]
    df[out_cols].to_csv(per_example_path, index=False)
    print(f"wrote {per_example_path} ({len(df)} rows)")

    dose_df, spear_df, mwu_df = summarize(df)
    shape_df = characterize_shape(dose_df)
    dose_df.to_csv(os.path.join(OUT_DIR, "results_summary_dose_response.csv"), index=False)
    spear_df.to_csv(os.path.join(OUT_DIR, "results_summary_spearman.csv"), index=False)
    mwu_df.to_csv(os.path.join(OUT_DIR, "results_summary_mwu_endpoints.csv"), index=False)
    shape_df.to_csv(os.path.join(OUT_DIR, "results_summary_shape.csv"), index=False)
    write_markdown_summary(dose_df, spear_df, mwu_df, shape_df, os.path.join(OUT_DIR, "results_summary.md"))

    print("\n=== DOSE-RESPONSE ===")
    print(dose_df.to_string(index=False))
    print("\n=== SPEARMAN (branch_length vs. score) ===")
    print(spear_df.to_string(index=False))
    print("\n=== MANN-WHITNEY (t=3.0 > t=0.0) ===")
    print(mwu_df.to_string(index=False))
    print("\n=== SHAPE CHARACTERIZATION ===")
    print(shape_df.to_string(index=False))


if __name__ == "__main__":
    main()
