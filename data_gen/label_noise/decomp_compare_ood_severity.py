#!/usr/bin/env python3
"""BALD vs. variance-based decomposition, cell #2 of the cross-check: does
aleatoric uncertainty stay flat as input severity increases (real ->
shuffled -> random DNA), the way a correctly-behaving decomposition
predicts (input novelty should show up as epistemic, not aleatoric)?

Restricted to the two sampling-based methods already implemented inline in
`all_methods_ood_retest.py` (mc_dropout, conv_epinet) -- the same DNABERT2
seed-1 checkpoints, no retraining.
"""
from __future__ import annotations

import os
import random

import pandas as pd
import torch
import transformers
from safetensors.torch import load_file
from scipy.stats import mannwhitneyu

from nn_proj.common.datasets import load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.common.variance_decomp import compute_uncertainty_variance
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithConvPrior
from nn_proj.models.epinet.feature_fns import NT_feature_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
BASE_CKPT = "checkpoints/seed_1/DNABERT2/label_noise_r00/base"
EPI_CKPT = "checkpoints/seed_1/DNABERT2/label_noise_r00/epinet"
BASES = ["A", "C", "G", "T"]


def load_base_arch():
    config = transformers.AutoConfig.from_pretrained(HUB_ID, num_labels=2, trust_remote_code=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )
    return model, config


def build_variant(test_ds, kind: str, seed: int = 42):
    seqs = list(test_ds["sequence"])
    if kind == "real":
        new_seqs = seqs
    elif kind == "shuffled":
        new_seqs = ["".join(random.Random(seed + i).sample(s, len(s))) for i, s in enumerate(seqs)]
    elif kind == "random_dna":
        new_seqs = ["".join(random.Random(seed + i).choices(BASES, k=len(s))) for i, s in enumerate(seqs)]
    else:
        raise ValueError(kind)
    df = test_ds.to_pandas()
    df["sequence"] = new_seqs
    from datasets import Dataset
    return Dataset.from_pandas(df, preserve_index=False)


def collect(logits_all, labels):
    unc = compute_uncertainty(logits_all)
    unc_var = compute_uncertainty_variance(logits_all)
    rows = []
    for i in range(logits_all.shape[1]):
        rows.append({
            "labels": int(labels[i]), "pred": int(unc["predicted_class"][i]),
            "bald_U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
            "bald_U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
            "var_U_epistemic": float(unc_var["normalized_epistemic_uncertainty"][i]),
            "var_U_aleatoric": float(unc_var["normalized_aleatoric_uncertainty"][i]),
        })
    return rows


def mwu(a, b, label):
    u, p = mannwhitneyu(a, b, alternative="greater")
    return {"comparison": label, "U": u, "p": p}


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=75, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    raw_test = load_NT_tasks(task="promoter_all", split="test")

    def _prep(ds):
        tokenized, collator = prep_for_trainer(ds, tokenizer, metadata_cols=())
        return tokenized.remove_columns(["sequence"]), collator

    variants = {k: _prep(build_variant(raw_test, k)) for k in ["real", "shuffled", "random_dna"]}

    results_table, mwu_table = [], []

    # ---------------- mc_dropout ----------------
    print("=== mc_dropout ===")
    base_model, config = load_base_arch()
    base_model.load_state_dict(load_file(os.path.join(BASE_CKPT, "model.safetensors")), strict=True)
    base_model = base_model.to(DEVICE)
    enable_mc_dropout(base_model, p=0.1)

    mc_out = {}
    for kind, (ds, collator) in variants.items():
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
        rows = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                logits_all = torch.stack([base_model(**inputs).logits for _ in range(16)], dim=0).cpu()
                rows.extend(collect(logits_all, labels.cpu()))
        mc_out[kind] = pd.DataFrame(rows)
    base_model.eval()

    # ---------------- conv_epinet ----------------
    print("=== conv_epinet ===")
    epi_base, _ = load_base_arch()
    epi_cfg = EpinetConfig(num_classes=2, include_inputs=True, vocab_size=config.vocab_size)
    wrapper = EpinetWrapper(epi_base, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    epi_model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=16).to(DEVICE)

    _dummy_ds, _dummy_collator = variants["real"]
    _dummy_batch = _dummy_collator([_dummy_ds[i] for i in range(2)])
    _dummy_batch = {k: v.to(DEVICE) for k, v in _dummy_batch.items() if k != "labels"}
    with torch.no_grad():
        epi_model.wrapper(_dummy_batch, n_index_samples=1)
    epi_model.load_state_dict(load_file(os.path.join(EPI_CKPT, "model.safetensors")), strict=True)
    epi_model.eval()

    epi_out = {}
    for kind, (ds, collator) in variants.items():
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
        rows = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                logits_all = epi_model.wrapper(inputs, n_index_samples=16, return_all=True).cpu()
                rows.extend(collect(logits_all, labels.cpu()))
        epi_out[kind] = pd.DataFrame(rows)

    # ---------------- summarize both methods, both decompositions ----------------
    for name, out in [("mc_dropout", mc_out), ("conv_epinet", epi_out)]:
        for score_col in ["bald_U_epistemic", "bald_U_aleatoric", "var_U_epistemic", "var_U_aleatoric"]:
            real = out["real"][score_col]
            for kind in ["real", "shuffled", "random_dna"]:
                d = out[kind]
                results_table.append({
                    "method": name, "decomp": score_col.split("_")[0], "score": score_col,
                    "variant": kind, "n": len(d), "mean": d[score_col].mean(), "std": d[score_col].std(),
                })
            for kind in ["shuffled", "random_dna"]:
                r = mwu(out[kind][score_col], real, f"{name}/{score_col}: {kind} > real")
                r["method"], r["score"] = name, score_col
                mwu_table.append(r)

    results_df = pd.DataFrame(results_table)
    mwu_df = pd.DataFrame(mwu_table)
    results_df.to_csv("data_gen/label_noise/decomp_compare_ood_results.csv", index=False)
    mwu_df.to_csv("data_gen/label_noise/decomp_compare_ood_mwu.csv", index=False)
    print("\n=== RESULTS ===")
    print(results_df.to_string(index=False))
    print("\n=== MANN-WHITNEY (proxy > real) ===")
    print(mwu_df.to_string(index=False))


if __name__ == "__main__":
    main()
