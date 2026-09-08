#!/usr/bin/env python3
"""Does U_epistemic rise with controlled evolutionary distance from real
`promoter_all` sequences?

Evaluates three UQ methods (mc_dropout, conv_epinet, evidential) on the
AliSim continuous branch-length divergence dataset
(`data_gen/promoter_alisim/csv_data/promoter_alisim.csv` -- see that
directory's README.md for how it was built) and asks whether each method's
epistemic-uncertainty signal tracks the branch_length axis while its
aleatoric signal stays comparatively flat -- the same "does the
decomposition behave correctly under input novelty" question
`data_gen/label_noise/decomp_compare_ood_severity.py` (mc_dropout,
conv_epinet, two discrete OOD variants) and
`decomp_compare_label_noise_evidential.py` (evidential, label-noise rate
axis) ask elsewhere in this project, applied here to a continuous,
principled evolutionary-distance axis instead of discrete bins.

All three methods reuse the DNABERT2 seed-1 `label_noise_r00` checkpoint
family (`checkpoints/seed_1/DNABERT2/label_noise_r00/{base,epinet,evidential}`)
-- the real task, 0% injected label noise, and the very checkpoint the
AliSim anchors' test split was itself drawn from, so it is the natural,
consistent choice (not a new checkpoint trained for this eval).

conv_epinet uses the batch-indexed `GaussianIndexer` fix from branch
`worktree-fix-epinet-batch-z` (one `z` sample per example, not one shared
across the whole batch) -- see `nn_proj/models/epinet/epinet.py`, copied in
wholesale from that branch. Without it, all K "posterior draws" for a given
sample index within one batch would share the same z, understating the
per-example epistemic spread this eval is trying to measure.

Environment: must run under a Python 3.10+ interpreter with `transformers`
in the 4.29-4.30 range (NOT 4.31+, NOT 4.55) -- see the "Environment" section
of `data_gen/promoter_alisim/uncertainty_eval/README` note / task README
for the full 3-way version-conflict diagnosis. Verified working:
`/scratch/home/glh52/venvs/aleatoric_boundary_venv` (Python 3.10.12,
transformers pinned to 4.29.2 for this run; started at 4.33.3, which still
hits the AutoModelForSequenceClassification config_class registration
ValueError below -- see README).

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=5 /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \\
        data_gen/promoter_alisim/uncertainty_eval/run_uncertainty_eval.py
"""
from __future__ import annotations

import json
import os
import sys

# Make the repo root importable regardless of cwd -- `python
# .../run_uncertainty_eval.py` puts this file's own directory on sys.path[0],
# not the repo root, so `import nn_proj...` fails unless this is added
# explicitly (but note this worktree's checkpoints/ live outside the
# worktree too -- see REPO_ROOT below for that separate issue).
_REPO_ROOT_FOR_IMPORT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
DATA_PATH = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/csv_data/promoter_alisim.csv")
OUT_DIR = os.path.join(_REPO_ROOT_FOR_IMPORT, "data_gen/promoter_alisim/uncertainty_eval")

# Absolute paths: `checkpoints/` is gitignored local-run-artifact storage
# that only physically exists under the main repo checkout, not under a
# worktree's own directory (worktrees don't share untracked files) -- see
# the "Environment" note in this eval's README for why this script is not
# run from inside `scripts/` the way the four-stage pipeline normally is.
REPO_ROOT = "/scratch/home/glh52/glm-epinet-pyt"
BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/base")
EPI_CKPT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/epinet")
EVID_CKPT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00/evidential")

MODEL_MAX_LENGTH = 75  # DNABERT2 tokens_per_base=0.25 x 300bp, per CLAUDE.md
BATCH_SIZE = 32
K_MC = 10
K_EPI = 10
BRANCH_LENGTHS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
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
    # InstaDeepAI-specific fixup used throughout this repo (DNABERT2's
    # tokenizer has no eos_token by default).
    tokenizer.eos_token = tokenizer.pad_token
    return tokenizer


def load_alisim_dataset(tokenizer):
    """Loads the AliSim CSV, keeping branch_length/anchor_id/etc. as metadata
    columns carried through tokenization untouched, for later grouping.

    Returns (tokenized_ds, collate_ds, collator, metadata_cols):
    `tokenized_ds` keeps the metadata + raw `sequence` columns (indexed
    directly for per-row metadata lookup); `collate_ds` drops them (a plain
    string column like `anchor_id`, or `sequence` itself, breaks
    DataCollatorWithPadding's tensor conversion, same reason
    decomp_compare_ood_severity.py's `_prep` calls
    `.remove_columns(["sequence"])` before building a DataLoader).
    """
    ds = load_local_dataset(DATA_PATH, encode_labels=True)
    metadata_cols = [c for c in METADATA_COLS if c in ds.column_names]
    tokenized, collator = prep_for_trainer(ds, tokenizer, max_length=MODEL_MAX_LENGTH, metadata_cols=metadata_cols)
    collate_ds = tokenized.remove_columns(["sequence"] + metadata_cols)
    return tokenized, collate_ds, collator, metadata_cols


def _extract_metadata(tokenized_ds, metadata_cols, idx_start, idx_end):
    """Pull metadata columns for rows [idx_start, idx_end) directly from the
    (still torch-formatted) dataset -- set_format(type='torch') wraps numeric
    columns in 0-d tensors, strings are left alone."""
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
    enable_mc_dropout(model, p=0.1)  # sets eval() then re-enables dropout modules' .train()

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

    # Build epinet internals with a dummy forward pass before loading the
    # state dict (lazy-initialized submodules, same as train_epinet.py).
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
                # Per task spec: vacuity is the epistemic-analogue for
                # evidential (its own K=1-stack U_epistemic is identically 0
                # by construction -- see nn_proj/models/evidential/evidential.py
                # predict_evidential's docstring, and
                # data_gen/label_noise/decomp_compare_label_noise_evidential.py's
                # header comment for the same convention). U_aleatoric is
                # evidential's own (expected entropy under the Dirichlet mean).
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
        # --- dose-response table: mean U_epistemic / U_aleatoric per branch_length ---
        for bl, g in sub.groupby("branch_length"):
            dose_rows.append({
                "method": method, "branch_length": bl, "n": len(g),
                "U_epistemic_mean": g["U_epistemic"].mean(), "U_epistemic_std": g["U_epistemic"].std(),
                "U_aleatoric_mean": g["U_aleatoric"].mean(), "U_aleatoric_std": g["U_aleatoric"].std(),
                "accuracy": (g["pred"] == g["labels"]).mean(),
            })

        # --- Spearman correlation: branch_length (continuous) vs each score ---
        for score in ["U_epistemic", "U_aleatoric"]:
            rho, p = spearmanr(sub["branch_length"], sub[score])
            spear_rows.append({"method": method, "score": score, "spearman_rho": rho, "spearman_p": p, "n": len(sub)})

        # --- Endpoint Mann-Whitney: t=0.0 vs t=1.5, alternative "greater" ---
        lo = sub[sub["branch_length"] == 0.0]
        hi = sub[sub["branch_length"] == 1.5]
        for score in ["U_epistemic", "U_aleatoric"]:
            u, p = mannwhitneyu(hi[score], lo[score], alternative="greater")
            mwu_rows.append({
                "method": method, "score": score, "comparison": "t=1.5 > t=0.0",
                "U": u, "p": p, "n_lo": len(lo), "n_hi": len(hi),
                "mean_lo": lo[score].mean(), "mean_hi": hi[score].mean(),
            })

    dose_df = pd.DataFrame(dose_rows).sort_values(["method", "branch_length"]).reset_index(drop=True)
    spear_df = pd.DataFrame(spear_rows).reset_index(drop=True)
    mwu_df = pd.DataFrame(mwu_rows).reset_index(drop=True)
    return dose_df, spear_df, mwu_df


def write_markdown_summary(dose_df, spear_df, mwu_df, path):
    lines = ["# promoter_alisim uncertainty evaluation: results summary", ""]
    lines.append(
        "Dose-response of each UQ method's uncertainty decomposition against "
        "AliSim branch length (substitutions/site) away from real "
        "`promoter_all` test-split anchors. See "
        "`data_gen/promoter_alisim/README.md` for dataset construction and "
        "`data_gen/promoter_alisim/uncertainty_eval/README` section of the "
        "top-level README for method/checkpoint details."
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

    lines.append("## Endpoint check: Mann-Whitney U, t=1.5 > t=0.0 (per method, per score)")
    lines.append("")
    lines.append(mwu_df.to_markdown(index=False, floatfmt=".4g"))
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
    dose_df.to_csv(os.path.join(OUT_DIR, "results_summary_dose_response.csv"), index=False)
    spear_df.to_csv(os.path.join(OUT_DIR, "results_summary_spearman.csv"), index=False)
    mwu_df.to_csv(os.path.join(OUT_DIR, "results_summary_mwu_endpoints.csv"), index=False)
    write_markdown_summary(dose_df, spear_df, mwu_df, os.path.join(OUT_DIR, "results_summary.md"))

    print("\n=== DOSE-RESPONSE ===")
    print(dose_df.to_string(index=False))
    print("\n=== SPEARMAN (branch_length vs. score) ===")
    print(spear_df.to_string(index=False))
    print("\n=== MANN-WHITNEY (t=1.5 > t=0.0) ===")
    print(mwu_df.to_string(index=False))


if __name__ == "__main__":
    main()
