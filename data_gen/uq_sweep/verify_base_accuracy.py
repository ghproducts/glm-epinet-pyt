#!/usr/bin/env python3
"""Verify each backbone's `base` checkpoint reproduces the manuscript's
reported in-distribution promoter_all accuracy before any sweep is run.

Manuscript Table tab:reg_id_err_reordered reports ID promoter_all *error*
(%, mean +/- sd over seeds) for the `base` inference strategy:

    NT_transformer  13.4 +/- 1.2   -> acc 86.6
    DNABERT2        13.9 +/- 1.1   -> acc 86.1
    hyenaDNA        15.0 +/- 0.2   -> acc 85.0
    CARMANIA        14.0 +/- 0.2   -> acc 86.0

A checkpoint landing well outside its backbone's band is a training/loading
bug, not noise, and the sweep should not run until it's explained.

Usage:
    python verify_base_accuracy.py [--backbones DNABERT2,NT_transformer,...]
                                   [--seeds 1,2,3,4,42] [--device cpu|cuda]
Writes verify_base_accuracy.csv in this directory.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
import transformers
from safetensors.torch import load_file
from sklearn.metrics import matthews_corrcoef

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from nn_proj.common.datasets import load_NT_tasks, prep_for_trainer  # noqa: E402

# Checkpoints are large and gitignored, so they live in the main checkout even
# when this script runs from a worktree. Override with $GLM_CKPT_ROOT.
CKPT_ROOT = os.environ.get("GLM_CKPT_ROOT") or os.path.join(REPO, "checkpoints")
TASK = "promoter_all"
SEQ_LEN = 300
BATCH_SIZE = 64

# hub id, tokens_per_base (from configs/experiments.yaml), manuscript error mean/sd
BACKBONES = {
    "NT_transformer": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", 0.1667, 13.4, 1.2),
    "DNABERT2":       ("zhihan1996/DNABERT-2-117M",                               0.25,   13.9, 1.1),
    "hyenaDNA":       ("LongSafari/hyenadna-small-32k-seqlen-hf",                 1.0,    15.0, 0.2),
    "CARMANIA":       ("MsAlEhR/carmania-160k-seqlen-human",                      1.0,    14.0, 0.2),
}


def max_length_for(backbone: str) -> int:
    return max(1, int(round(SEQ_LEN * BACKBONES[backbone][1])))


def load_tokenizer(backbone: str):
    hub_id = BACKBONES[backbone][0]
    return transformers.AutoTokenizer.from_pretrained(
        hub_id, model_max_length=max_length_for(backbone), padding_side="right",
        use_fast=True, trust_remote_code=True,
    )


def load_checkpoint_weights(ckpt_dir: str):
    """Backbones don't agree on serialization format: DNABERT2/CARMANIA save
    `model.safetensors`, hyenaDNA saves `pytorch_model.bin`."""
    st = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(st):
        return load_file(st)
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no model.safetensors or pytorch_model.bin in {ckpt_dir}")


def load_model(backbone: str, ckpt_dir: str):
    """Architecture from the hub id, weights from the checkpoint's safetensors.

    Config/architecture always come from the hub id rather than the checkpoint
    dir: DNABERT2's remote code resolves differently for a local path and
    raises `BertConfig has no attribute pad_token_id` on newer transformers.
    """
    hub_id = BACKBONES[backbone][0]
    weights = load_checkpoint_weights(ckpt_dir)

    if backbone == "CARMANIA":
        # Not a native AutoModelForSequenceClassification, and the one backbone
        # whose config must come from the *checkpoint*: its config.json carries
        # `encoder_name`, which the constructor requires.
        from nn_proj.models.CARMANIA.model import (
            CarmaniaForSequenceClassification, CarmaniaSequenceClassificationConfig,
        )
        config = CarmaniaSequenceClassificationConfig.from_pretrained(ckpt_dir)
        model = CarmaniaForSequenceClassification(config)
    else:
        config = transformers.AutoConfig.from_pretrained(hub_id, num_labels=2, trust_remote_code=True)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            hub_id, config=config, trust_remote_code=True
        )
    missing, unexpected = model.load_state_dict(weights, strict=False)
    return model, list(missing), list(unexpected)


@torch.no_grad()
def evaluate(model, loader, device) -> list[int]:
    model.eval().to(device)
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out = model(**batch)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        preds.extend(logits.argmax(-1).cpu().tolist())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--task-dir", default="label_noise_r00",
                    help="checkpoint subdir; label_noise_r00 == clean promoter_all")
    ap.add_argument("--ckpt-root", default=CKPT_ROOT)
    args = ap.parse_args()
    ckpt_root = args.ckpt_root

    rows = []
    for backbone in args.backbones.split(","):
        backbone = backbone.strip()
        if backbone not in BACKBONES:
            raise SystemExit(f"unknown backbone {backbone!r}")
        tokenizer = load_tokenizer(backbone)
        test_ds = load_NT_tasks(task=TASK, split="test")
        labels = list(test_ds["labels"])
        tokenized, collator = prep_for_trainer(
            test_ds, tokenizer, max_length=max_length_for(backbone), metadata_cols=()
        )
        tokenized = tokenized.remove_columns(["sequence"])
        loader = torch.utils.data.DataLoader(
            tokenized, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
        )

        for seed in [s.strip() for s in args.seeds.split(",")]:
            ckpt = os.path.join(ckpt_root, f"seed_{seed}", backbone, args.task_dir, "base")
            if not os.path.isdir(ckpt):
                print(f"[skip] {backbone} seed {seed}: no checkpoint at {ckpt}", flush=True)
                continue
            try:
                model, missing, unexpected = load_model(backbone, ckpt)
                preds = evaluate(model, loader, args.device)
            except Exception as exc:  # keep going; one backbone's failure isn't fatal
                import traceback
                print(f"[FAILED] {backbone} seed {seed}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                rows.append(dict(backbone=backbone, seed=seed, error=f"{type(exc).__name__}: {exc}"))
                continue
            acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)
            _, _, exp_err, exp_sd = BACKBONES[backbone]
            err = 100 * (1 - acc)
            # how many manuscript SDs away from the reported mean
            z = (err - exp_err) / exp_sd if exp_sd else float("nan")
            rows.append(dict(
                backbone=backbone, seed=seed, n=len(labels), accuracy=acc, error_pct=err,
                mcc=matthews_corrcoef(labels, preds),
                manuscript_error_pct=exp_err, manuscript_sd=exp_sd, sd_from_manuscript=z,
                n_missing_keys=len(missing), n_unexpected_keys=len(unexpected),
            ))
            # Two backbones have a very tight manuscript sd (0.2), so a z-score
            # alone flags sub-percentage-point differences as anomalies. Require
            # a materially large absolute gap too, and only treat *worse*-than-
            # reported as a suspected bug -- beating the reported number is not a
            # training defect, though it does need explaining in the writeup.
            delta = err - exp_err
            if abs(z) <= 3 or abs(delta) < 2.0:
                flag = "OK" if delta <= 0 else "OK (slightly worse)"
            elif delta > 0:
                flag = "** WORSE THAN MANUSCRIPT - INVESTIGATE **"
            else:
                flag = "** BETTER THAN MANUSCRIPT - explain in writeup **"
            print(f"{backbone:15s} seed {seed:>2s}  acc {acc:.4f}  err {err:5.2f}%  "
                  f"(manuscript {exp_err}+/-{exp_sd}, {z:+.1f} sd)  "
                  f"missing={len(missing)} unexpected={len(unexpected)}  {flag}", flush=True)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    if "sd_from_manuscript" not in out.columns:
        out["sd_from_manuscript"] = float("nan")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_base_accuracy.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    if not out.empty:
        bad = out[out.sd_from_manuscript.abs() > 3]
        print("all backbones within 3 sd of manuscript" if bad.empty
              else f"OUT OF BAND:\n{bad[['backbone','seed','error_pct','manuscript_error_pct']]}")


if __name__ == "__main__":
    main()
