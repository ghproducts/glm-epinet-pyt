#!/usr/bin/env python3
"""Deep-ensemble version of cell #1: does epistemic uncertainty stay flat
as label-noise rate r increases, when the K "samples" come from K
independently-trained models (seeds 1,2,3's own base checkpoints) instead
of dropout masks or epinet index perturbations around one fixed model?

No new training: reuses checkpoints/seed_{1,2,3}/DNABERT2/label_noise_r*/base
(the plain `base` method's own checkpoints, already produced by the full
label-noise grid) purely as ensemble members -- a genuine deep ensemble,
K=3 (this project's label-noise grid only used 3 seeds, smaller than a
typical 5-10 member ensemble, but the honest number available without new
training). Same clean, untouched test set at every rate.
"""
from __future__ import annotations

import os

import pandas as pd
import torch
import transformers
from safetensors.torch import load_file

from nn_proj.common.datasets import load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty
from nn_proj.common.variance_decomp import compute_uncertainty_variance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
RATES = ["r00", "r05", "r10", "r20", "r40"]
SEEDS = [1, 2, 3, 4, 42]  # ensemble members -- this project's standard 5-seed convention


def load_base(ckpt_dir, config):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )
    model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
    return model.to(DEVICE).eval()


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
        members = [load_base(f"checkpoints/seed_{s}/DNABERT2/label_noise_{rate}/base", config) for s in SEEDS]

        loader = torch.utils.data.DataLoader(tokenized, batch_size=64, shuffle=False, collate_fn=collator)
        all_unc = {"bald_U_epistemic": [], "bald_U_aleatoric": [], "var_U_epistemic": [], "var_U_aleatoric": []}
        acc_n, acc_correct = 0, 0
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                # K = len(members) deterministic forward passes, one per ensemble member.
                logits_all = torch.stack([m(**inputs).logits for m in members], dim=0).cpu()
                unc = compute_uncertainty(logits_all)
                unc_var = compute_uncertainty_variance(logits_all)
                all_unc["bald_U_epistemic"].append(unc["normalized_epistemic_uncertainty"])
                all_unc["bald_U_aleatoric"].append(unc["normalized_aleatoric_uncertainty"])
                all_unc["var_U_epistemic"].append(unc_var["normalized_epistemic_uncertainty"])
                all_unc["var_U_aleatoric"].append(unc_var["normalized_aleatoric_uncertainty"])
                acc_correct += (unc["predicted_class"].cpu() == labels.cpu()).sum().item()
                acc_n += labels.shape[0]

        row = {"rate": rate, "accuracy": acc_correct / acc_n}
        for k, v in all_unc.items():
            t = torch.cat(v)
            row[f"{k}_mean"] = t.mean().item()
            row[f"{k}_std"] = t.std().item()
        rows.append(row)
        print(row)

        for m in members:
            del m
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv("data_gen/label_noise/decomp_compare_label_noise_ensemble.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
