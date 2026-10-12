#!/usr/bin/env python3
"""Deep-ensemble version of cell #2: does aleatoric uncertainty stay flat
as input severity increases (real -> shuffled -> random DNA), when the K
"samples" come from K independently-trained models instead of dropout/
epinet perturbations of one model?

No new training: reuses checkpoints/seed_{1,2,3}/DNABERT2/label_noise_r00/base
(clean promoter_all, K=3 ensemble members) exactly as in
decomp_compare_label_noise_ensemble.py.
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
from nn_proj.common.utils import compute_uncertainty
from nn_proj.common.variance_decomp import compute_uncertainty_variance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
SEEDS = [1, 2, 3, 4, 42]  # this project's standard 5-seed convention
BASES = ["A", "C", "G", "T"]


def load_base(ckpt_dir, config):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )
    model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
    return model.to(DEVICE).eval()


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


def mwu(a, b, label):
    u, p = mannwhitneyu(a, b, alternative="greater")
    return {"comparison": label, "U": u, "p": p}


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=75, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    config = transformers.AutoConfig.from_pretrained(HUB_ID, num_labels=2, trust_remote_code=True)
    raw_test = load_NT_tasks(task="promoter_all", split="test")

    def _prep(ds):
        tokenized, collator = prep_for_trainer(ds, tokenizer, metadata_cols=())
        return tokenized.remove_columns(["sequence"]), collator

    variants = {k: _prep(build_variant(raw_test, k)) for k in ["real", "shuffled", "random_dna"]}
    members = [load_base(f"checkpoints/seed_{s}/DNABERT2/label_noise_r00/base", config) for s in SEEDS]

    out = {}
    for kind, (ds, collator) in variants.items():
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
        rows = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                logits_all = torch.stack([m(**inputs).logits for m in members], dim=0).cpu()
                unc = compute_uncertainty(logits_all)
                unc_var = compute_uncertainty_variance(logits_all)
                for i in range(logits_all.shape[1]):
                    rows.append({
                        "labels": int(labels[i]), "pred": int(unc["predicted_class"][i]),
                        "bald_U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                        "bald_U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                        "var_U_epistemic": float(unc_var["normalized_epistemic_uncertainty"][i]),
                        "var_U_aleatoric": float(unc_var["normalized_aleatoric_uncertainty"][i]),
                    })
        out[kind] = pd.DataFrame(rows)

    results_table, mwu_table = [], []
    for score_col in ["bald_U_epistemic", "bald_U_aleatoric", "var_U_epistemic", "var_U_aleatoric"]:
        real = out["real"][score_col]
        for kind in ["real", "shuffled", "random_dna"]:
            d = out[kind]
            results_table.append({
                "method": "ensemble", "score": score_col, "variant": kind,
                "n": len(d), "mean": d[score_col].mean(), "std": d[score_col].std(),
            })
        for kind in ["shuffled", "random_dna"]:
            r = mwu(out[kind][score_col], real, f"ensemble/{score_col}: {kind} > real")
            r["score"] = score_col
            mwu_table.append(r)

    results_df = pd.DataFrame(results_table)
    mwu_df = pd.DataFrame(mwu_table)
    results_df.to_csv("data_gen/label_noise/decomp_compare_ood_results_ensemble.csv", index=False)
    mwu_df.to_csv("data_gen/label_noise/decomp_compare_ood_mwu_ensemble.csv", index=False)
    print("\n=== RESULTS ===")
    print(results_df.to_string(index=False))
    print("\n=== MANN-WHITNEY (proxy > real) ===")
    print(mwu_df.to_string(index=False))


if __name__ == "__main__":
    main()
