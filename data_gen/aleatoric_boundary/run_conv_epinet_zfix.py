#!/usr/bin/env python3
"""Re-run `conv_epinet` on the aleatoric_boundary margin-quartile test set
(1584 promoter_all test examples) under the NEWLY RETRAINED `epinet_zfix`
checkpoint (`checkpoints/seed_1/DNABERT2/label_noise_r00/epinet_zfix`),
which was trained under the per-example-z fix from the start -- unlike the
original `epinet` checkpoint used by `make_boundary_eval.py`, which predates
the fix (fb38241) and only ever had it applied as an eval-time monkeypatch.

Everything else (tokenizer, DataLoader, K, batch size, row order) is
IDENTICAL to `make_boundary_eval.py`'s `run_conv_epinet` so the resulting
rows join onto the existing `margin_scores.csv` (built once, from the `base`
checkpoint, and unaffected by this fix) by `idx` exactly as before.

Only the 9-method roster's `conv_epinet` rows change here; the other 8
methods (mc_dropout, evidential, laplace, ensemble_k5, ensemble_k3,
cnn_mc_dropout, cnn_ensemble, rf) are untouched and reused unchanged from
`csv_data/uncertainty_by_method.csv` -- there is no branch-length-style axis
to extend on this dataset, so this is purely a checkpoint swap.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=0 /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \\
        data_gen/aleatoric_boundary/run_conv_epinet_zfix.py
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT_FOR_IMPORT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_IMPORT)

import pandas as pd
import torch
import transformers
from safetensors.torch import load_file

from nn_proj.common.datasets import load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithConvPrior
from nn_proj.models.epinet.feature_fns import NT_feature_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
K = 10
BATCH_SIZE = 64
MAX_LENGTH = 75

REPO_ROOT = "/scratch/home/glh52/glm-epinet-pyt"
CKPT_ROOT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00")
# The NEW checkpoint -- trained under the fixed per-example-z code.
EPINET_CKPT = os.path.join(CKPT_ROOT, "epinet_zfix")

OUT_DIR = "data_gen/aleatoric_boundary"
CSV_DIR = os.path.join(OUT_DIR, "csv_data")


def load_hub_config():
    return transformers.AutoConfig.from_pretrained(HUB_ID, num_labels=2, trust_remote_code=True)


def load_tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=MAX_LENGTH, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    tok.eos_token = tok.pad_token
    return tok


def load_test_set(tokenizer):
    test_ds = load_NT_tasks(task="promoter_all", split="test")
    labels = list(test_ds["labels"])
    tokenized, collator = prep_for_trainer(test_ds, tokenizer, max_length=MAX_LENGTH, metadata_cols=())
    tokenized = tokenized.remove_columns(["sequence"])
    return labels, tokenized, collator


def make_loader(tokenized, collator):
    return torch.utils.data.DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)


def run_conv_epinet(tokenized, collator) -> pd.DataFrame:
    config = load_hub_config()
    base = transformers.AutoModelForSequenceClassification.from_pretrained(HUB_ID, config=config, trust_remote_code=True)
    epi_cfg = EpinetConfig(num_classes=2, include_inputs=True, vocab_size=config.vocab_size)
    wrapper = EpinetWrapper(base, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=K).to(DEVICE)

    dummy_batch = collator([tokenized[i] for i in range(2)])
    dummy_batch = {k: v.to(DEVICE) for k, v in dummy_batch.items() if k != "labels"}
    with torch.no_grad():
        model.wrapper(dummy_batch, n_index_samples=1)
    model.load_state_dict(load_file(os.path.join(EPINET_CKPT, "model.safetensors")), strict=True)
    model.eval()

    loader = make_loader(tokenized, collator)
    rows = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            logits_all = model.wrapper(inputs, n_index_samples=K, return_all=True).float().cpu()
            unc = compute_uncertainty(logits_all)
            B = logits_all.shape[1]
            for i in range(B):
                rows.append({
                    "idx": idx,
                    "method": "conv_epinet",
                    "pred": int(unc["predicted_class"][i]),
                    "U_total": float(unc["normalized_total_uncertainty"][i]),
                    "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                    "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                    "vote_pct": float(unc["vote_percentage"][i]),
                })
                idx += 1

    del model, base
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def main():
    tokenizer = load_tokenizer()
    labels, tokenized, collator = load_test_set(tokenizer)
    print(f"Loaded {len(tokenized)} test examples")

    new_epi = run_conv_epinet(tokenized, collator)

    mscore = pd.read_csv(os.path.join(CSV_DIR, "margin_scores.csv"))
    assert len(mscore) == len(new_epi), f"row count mismatch: margin_scores={len(mscore)} vs conv_epinet={len(new_epi)}"
    merged = new_epi.merge(mscore[["idx", "label", "quartile", "margin"]], on="idx", how="left")
    merged = merged.rename(columns={"label": "label"})

    # Match uncertainty_by_method.csv's column set/order (var_U_epistemic /
    # var_U_aleatoric / dirichlet_strength / vacuity are NaN for conv_epinet,
    # exactly as in the original file).
    merged["vacuity"] = float("nan")
    merged["dirichlet_strength"] = float("nan")
    merged["var_U_epistemic"] = float("nan")
    merged["var_U_aleatoric"] = float("nan")
    out_cols = ["idx", "method", "pred", "U_total", "U_epistemic", "U_aleatoric", "vote_pct",
                "vacuity", "dirichlet_strength", "label", "quartile", "margin",
                "var_U_epistemic", "var_U_aleatoric"]
    merged = merged[out_cols]

    out_path = os.path.join(CSV_DIR, "conv_epinet_zfix.csv")
    merged.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(merged)} rows)")

    # Build the combined, zfix-updated 9-method file: original CSV with
    # conv_epinet rows replaced by the new zfix ones. Original file kept
    # untouched for provenance.
    existing = pd.read_csv(os.path.join(CSV_DIR, "uncertainty_by_method.csv"))
    existing_no_epi = existing[existing["method"] != "conv_epinet"]
    combined = pd.concat([existing_no_epi, merged], ignore_index=True)
    combined_path = os.path.join(CSV_DIR, "uncertainty_by_method_zfix.csv")
    combined.to_csv(combined_path, index=False)
    print(f"wrote {combined_path} ({len(combined)} rows, {combined['method'].nunique()} methods)")


if __name__ == "__main__":
    main()
