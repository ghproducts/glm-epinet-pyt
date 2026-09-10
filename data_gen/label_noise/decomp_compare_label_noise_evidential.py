#!/usr/bin/env python3
"""Evidential deep learning across label-noise rates: does vacuity
(C / sum(alpha), the closed-form epistemic-like signal -- see
nn_proj/models/evidential/evidential.py) stay flat as noise rate rises?
Evidential parameterizes a Dirichlet in one forward pass rather than
drawing K samples, so its own `U_epistemic` is 0 by construction; vacuity
is the column to read instead, comparable only across rate, not directly
against other methods' bald/var epistemic columns.

Trains from scratch per rate (evidential changes the loss itself, so
there's no frozen base to wrap post-hoc), one seed (seed=1)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import torch
import transformers
from safetensors.torch import load_file

from nn_proj.common.datasets import load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty
from nn_proj.models.evidential import EvidentialConfig, EvidentialWrapper, HFEvidentialSeqClassifier, predict_evidential

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
RATES = ["r00", "r05", "r10", "r20", "r40"]
SEED = 1


def train_one(rate: str, out_dir: str):
    if os.path.isfile(os.path.join(out_dir, "model.safetensors")):
        print(f"[{rate}] checkpoint already exists at {out_dir}, skipping training")
        return
    cmd = [
        sys.executable, "-m", "nn_proj.models.DNABERT2.train_evidential",
        "--data_path", f"data_gen/label_noise/csv_data_{rate}/train.csv",
        "--model_max_length", "75",
        "--per_device_train_batch_size", "64",
        "--per_device_eval_batch_size", "64",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "6",
        "--fp16", "True",
        "--output_dir", out_dir,
        "--eval_strategy", "epoch",
        "--save_strategy", "epoch",
        "--warmup_steps", "50",
        "--logging_steps", "100",
        "--overwrite_output_dir", "True",
        "--annealing_step", "3",
        "--evidential_loss_type", "mse",
        "--evidence_activation", "softplus",
        "--save_model", "True",
        "--seed", str(SEED), "--data_seed", str(SEED),
        "--run_name", f"DNABERT2_evidential_label_noise_{rate}",
    ]
    print(f"[{rate}] training: {' '.join(cmd)}")
    # Single-GPU only: on this multi-GPU node, HF Trainer auto-wraps the
    # model in DataParallel across every visible GPU, which crashes with
    # "CUDA error: peer mapping resources exhausted" (documented in
    # docs/NEW_UQ_METHODS.md -- hit the original evidential smoke test the
    # same way). Restricting to one GPU avoids the DataParallel path.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    subprocess.run(cmd, check=True, cwd="/scratch/home/glh52/glm-epinet-pyt", env=env)


def evaluate_one(rate: str, out_dir: str, tokenizer, test_ds):
    config = transformers.AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    num_labels = config.num_labels
    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )

    cfg_path = os.path.join(out_dir, "evidential_config.json")
    with open(cfg_path) as f:
        saved_cfg = json.load(f)
    evidential_cfg = EvidentialConfig(
        num_classes=num_labels,
        evidence_activation=saved_cfg.get("evidence_activation", "softplus"),
        loss_type=saved_cfg.get("loss_type", "mse"),
        annealing_step=saved_cfg.get("annealing_step", 10),
    )
    wrapper = EvidentialWrapper(base_model, evidential_cfg)
    model = HFEvidentialSeqClassifier(wrapper).to(DEVICE)
    model.load_state_dict(load_file(os.path.join(out_dir, "model.safetensors")), strict=True)
    model.eval()

    metadata_cols = [c for c in test_ds.column_names if c not in ("sequence", "label", "labels")]
    test_tok, test_collator = prep_for_trainer(test_ds, tokenizer, metadata_cols=metadata_cols)

    rows = predict_evidential(model, test_tok, test_collator, batch_size=64)
    df = pd.DataFrame(rows)
    acc = (df["pred"] == df["labels"]).mean()

    del model, base_model
    torch.cuda.empty_cache()

    return {
        "rate": rate,
        "accuracy": float(acc),
        "vacuity_mean": float(df["vacuity"].mean()),
        "vacuity_std": float(df["vacuity"].std()),
        "dirichlet_strength_mean": float(df["dirichlet_strength"].mean()),
        "U_aleatoric_mean": float(df["U_aleatoric"].mean()),
        "U_aleatoric_std": float(df["U_aleatoric"].std()),
    }


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=75, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    test_ds = load_NT_tasks(task="promoter_all", split="test")

    results = []
    for rate in RATES:
        out_dir = f"checkpoints/seed_{SEED}/DNABERT2/label_noise_{rate}/evidential"
        train_one(rate, out_dir)
        res = evaluate_one(rate, out_dir, tokenizer, test_ds)
        print(res)
        results.append(res)

        df = pd.DataFrame(results)
        df.to_csv("data_gen/label_noise/decomp_compare_label_noise_evidential.csv", index=False)

    print("\n" + pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
