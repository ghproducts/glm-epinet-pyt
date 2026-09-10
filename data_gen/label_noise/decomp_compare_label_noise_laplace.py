#!/usr/bin/env python3
"""Last-layer Laplace (nn_proj.models.laplace) across label-noise rates:
does epistemic uncertainty stay flat as noise rate rises, when K samples
come from a fitted diagonal Gaussian posterior over the classifier head
rather than dropout/epinet/independent checkpoints?

Post-hoc, no retraining: reuses checkpoints/seed_{1,2,3}/DNABERT2/
label_noise_r*/base as the MAP point, fits the diagonal GGN on that rate's
own noisy-labeled train.csv, K=16 posterior samples, evaluated on the
clean test set at every rate."""
from __future__ import annotations

import os

import pandas as pd
import torch
import transformers
from safetensors.torch import load_file

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty
from nn_proj.common.variance_decomp import compute_uncertainty_variance
from nn_proj.models.laplace import LaplaceConfig, LaplaceSeqClassifier, fit_diagonal_laplace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
RATES = ["r00", "r05", "r10", "r20", "r40"]
SEEDS = [1, 2, 3]  # matches conv_epinet's and mc_dropout's Cell 1 seed set
K = 16
FIT_EXAMPLES = 2000  # same cap inference_laplace.py defaults to


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
    test_tok, test_collator = prep_for_trainer(test_ds, tokenizer, metadata_cols=())
    test_tok = test_tok.remove_columns(["sequence"])

    rows = []
    for rate in RATES:
        # The noisy-labeled train split this rate's checkpoint was actually
        # trained on -- fitting the GGN needs (input, *trained-on* label)
        # pairs, not the clean test labels.
        fit_ds_raw = load_local_dataset(f"data_gen/label_noise/csv_data_{rate}/train.csv")
        fit_tok, fit_collator = prep_for_trainer(fit_ds_raw, tokenizer, metadata_cols=())
        fit_tok = fit_tok.remove_columns(["sequence"])

        for seed in SEEDS:
            model = load_base(f"checkpoints/seed_{seed}/DNABERT2/label_noise_{rate}/base", config)

            lap_cfg = LaplaceConfig(classifier_attr="classifier", prior_precision=1.0, max_examples=FIT_EXAMPLES)
            weight_var, bias_var, n_used = fit_diagonal_laplace(model, fit_tok, fit_collator, lap_cfg, batch_size=64)
            lap_model = LaplaceSeqClassifier(model, weight_var.to(DEVICE), bias_var.to(DEVICE),
                                              classifier_attr="classifier").to(DEVICE)

            loader = torch.utils.data.DataLoader(test_tok, batch_size=64, shuffle=False, collate_fn=test_collator)
            all_unc = {"bald_U_epistemic": [], "bald_U_aleatoric": [], "var_U_epistemic": [], "var_U_aleatoric": []}
            acc_n, acc_correct = 0, 0
            with torch.no_grad():
                for batch in loader:
                    labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                    inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                    lap_model(**inputs, labels=None)          # populate cached features
                    logits_all = lap_model.sample_logits(n_samples=K).cpu()  # [K,B,C]
                    unc = compute_uncertainty(logits_all)
                    unc_var = compute_uncertainty_variance(logits_all)
                    all_unc["bald_U_epistemic"].append(unc["normalized_epistemic_uncertainty"])
                    all_unc["bald_U_aleatoric"].append(unc["normalized_aleatoric_uncertainty"])
                    all_unc["var_U_epistemic"].append(unc_var["normalized_epistemic_uncertainty"])
                    all_unc["var_U_aleatoric"].append(unc_var["normalized_aleatoric_uncertainty"])
                    acc_correct += (unc["predicted_class"].cpu() == labels.cpu()).sum().item()
                    acc_n += labels.shape[0]

            row = {"rate": rate, "seed": seed, "accuracy": acc_correct / acc_n, "laplace_fit_n": n_used}
            for k, v in all_unc.items():
                t = torch.cat(v)
                row[f"{k}_mean"] = t.mean().item()
                row[f"{k}_std"] = t.std().item()
            rows.append(row)
            print(row)

            del model, lap_model
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = "data_gen/label_noise/decomp_compare_label_noise_laplace.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nper-rate mean across seeds:")
    print(df.groupby("rate")[["accuracy", "bald_U_epistemic_mean", "bald_U_aleatoric_mean",
                               "var_U_epistemic_mean", "var_U_aleatoric_mean"]].mean().reindex(RATES).to_string())


if __name__ == "__main__":
    main()
