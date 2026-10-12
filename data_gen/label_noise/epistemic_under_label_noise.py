#!/usr/bin/env python3
"""Missing cell #1 of the aleatoric/epistemic cross-check: does epistemic
uncertainty (conv_epinet) stay flat as label-noise rate r increases, the
way a correctly-behaving decomposition predicts (label noise is independent
of x, so nothing about the *input* distribution changes -- only epistemic
should be blind to it, while aleatoric should rise, per the label-noise
pilot's already-confirmed base-method result)?

Evaluates on the untouched, always-clean test set (same 1584 rows at every
r) using each rate's own epinet checkpoint, DNABERT2, seeds {1,2,3}.
"""
from __future__ import annotations

import os

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
RATES = ["r00", "r05", "r10", "r20", "r40"]
SEEDS = [1, 2, 3]


def load_epinet(ckpt_dir, config, dummy_batch):
    base = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )
    epi_cfg = EpinetConfig(num_classes=2, include_inputs=True, vocab_size=config.vocab_size)
    wrapper = EpinetWrapper(base, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=16).to(DEVICE)
    with torch.no_grad():
        model.wrapper(dummy_batch, n_index_samples=1)  # trigger lazy ProjectedMLP.core build
    model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
    model.eval()
    return model


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=75, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    config = transformers.AutoConfig.from_pretrained(HUB_ID, num_labels=2, trust_remote_code=True)

    # Clean, untouched test set -- identical across every noise rate by construction.
    test_ds = load_NT_tasks(task="promoter_all", split="test")
    tokenized, collator = prep_for_trainer(test_ds, tokenizer, metadata_cols=())
    tokenized = tokenized.remove_columns(["sequence"])
    dummy_batch = collator([tokenized[i] for i in range(2)])
    dummy_batch = {k: v.to(DEVICE) for k, v in dummy_batch.items() if k != "labels"}

    rows = []
    for rate in RATES:
        for seed in SEEDS:
            ckpt = f"checkpoints/seed_{seed}/DNABERT2/label_noise_{rate}/epinet"
            model = load_epinet(ckpt, config, dummy_batch)

            loader = torch.utils.data.DataLoader(tokenized, batch_size=64, shuffle=False, collate_fn=collator)
            all_unc = {"U_total": [], "U_epistemic": [], "U_aleatoric": []}
            acc_n, acc_correct = 0, 0
            with torch.no_grad():
                for batch in loader:
                    labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                    inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                    logits_all = model.wrapper(inputs, n_index_samples=16, return_all=True).cpu()
                    unc = compute_uncertainty(logits_all)
                    all_unc["U_total"].append(unc["normalized_total_uncertainty"])
                    all_unc["U_epistemic"].append(unc["normalized_epistemic_uncertainty"])
                    all_unc["U_aleatoric"].append(unc["normalized_aleatoric_uncertainty"])
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
    df.to_csv("data_gen/label_noise/epistemic_under_label_noise.csv", index=False)
    print("\n" + df.to_string(index=False))

    print("\nSeed-averaged:")
    print(df.groupby("rate")[["accuracy", "U_total_mean", "U_epistemic_mean", "U_aleatoric_mean"]]
          .mean().reindex(RATES).to_string())


if __name__ == "__main__":
    main()
