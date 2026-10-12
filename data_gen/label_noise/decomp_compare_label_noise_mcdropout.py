#!/usr/bin/env python3
"""MC-dropout version of cell #1: does epistemic uncertainty stay flat as
label-noise rate r increases, when the K "samples" come from K stochastic
dropout-active forward passes through *one* fine-tuned model, instead of
epinet index perturbations (decomp_compare_label_noise.py) or K
independently-trained models (decomp_compare_label_noise_ensemble.py)?

This is the missing DNABERT2/mc_dropout row: the original Sep 6 cross-check
(ALEATORIC_EPISTEMIC_CROSSCHECK.md) only ran mc_dropout on the input-novelty
axis (Cell 2), never on label noise (Cell 1) -- only conv_epinet was tested
there.

No new training: reuses checkpoints/seed_{1,2,3}/DNABERT2/label_noise_r*/base
(the plain `base` method's own checkpoints), one seed's model per row,
dropout enabled at eval (rate=0.1, this project's standard, see
enable_mc_dropout), K=16 stochastic passes -- same K as every other script in
this cross-check family. Same clean, untouched test set at every rate.
Reports per (rate, seed) rows, mirroring decomp_compare_label_noise.py's
layout, for direct comparison against conv_epinet's Cell 1 numbers.
"""
from __future__ import annotations

import os

import pandas as pd
import torch
import transformers
from safetensors.torch import load_file

from nn_proj.common.datasets import load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.common.variance_decomp import compute_uncertainty_variance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
RATES = ["r00", "r05", "r10", "r20", "r40"]
SEEDS = [1, 2, 3]  # matches conv_epinet's Cell 1 seed set exactly
K = 16


def load_base(ckpt_dir, config):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )
    model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
    model = model.to(DEVICE)
    enable_mc_dropout(model, p=0.1)
    return model


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=75, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    config = transformers.AutoConfig.from_pretrained(HUB_ID, num_labels=2, trust_remote_code=True)

    test_ds = load_NT_tasks(task="promoter_all", split="test")
    tokenized, collator = prep_for_trainer(test_ds, tokenizer, metadata_cols=())
    tokenized = tokenized.remove_columns(["sequence"])

    rows = []
    for rate in RATES:
        for seed in SEEDS:
            model = load_base(f"checkpoints/seed_{seed}/DNABERT2/label_noise_{rate}/base", config)

            loader = torch.utils.data.DataLoader(tokenized, batch_size=64, shuffle=False, collate_fn=collator)
            all_unc = {"bald_U_epistemic": [], "bald_U_aleatoric": [], "var_U_epistemic": [], "var_U_aleatoric": []}
            acc_n, acc_correct = 0, 0
            with torch.no_grad():
                for batch in loader:
                    labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                    inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                    # K stochastic dropout-active passes through the SAME model/weights.
                    logits_all = torch.stack([model(**inputs).logits for _ in range(K)], dim=0).cpu()
                    unc = compute_uncertainty(logits_all)
                    unc_var = compute_uncertainty_variance(logits_all)
                    all_unc["bald_U_epistemic"].append(unc["normalized_epistemic_uncertainty"])
                    all_unc["bald_U_aleatoric"].append(unc["normalized_aleatoric_uncertainty"])
                    all_unc["var_U_epistemic"].append(unc_var["normalized_epistemic_uncertainty"])
                    all_unc["var_U_aleatoric"].append(unc_var["normalized_aleatoric_uncertainty"])
                    acc_correct += (unc["predicted_class"].cpu() == labels.cpu()).sum().item()
                    acc_n += labels.shape[0]

            row = {"rate": rate, "seed": seed, "accuracy": acc_correct / acc_n}
            for k, v in all_unc.items():
                t = torch.cat(v)
                row[f"{k}_mean"] = t.mean().item()
                row[f"{k}_std"] = t.std().item()
            rows.append(row)
            print(row)

            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = "data_gen/label_noise/decomp_compare_label_noise_mcdropout.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nper-rate mean across seeds:")
    print(df.groupby("rate")[["accuracy", "bald_U_epistemic_mean", "bald_U_aleatoric_mean",
                               "var_U_epistemic_mean", "var_U_aleatoric_mean"]].mean().reindex(RATES).to_string())


if __name__ == "__main__":
    main()
