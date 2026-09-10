#!/usr/bin/env python3
"""Evidential OOD-vacuity check across three severity levels, same size
(n=1584, matching promoter_all's test split):
  1. real       -- promoter_all's real test sequences (ID)
  2. shuffled   -- same sequences, characters shuffled per-sequence (same
                   composition, order destroyed)
  3. random_dna -- synthetic, i.i.d. uniform-random bases, same length,
                   no compositional relationship to any real sequence
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import transformers
from safetensors.torch import load_file
from scipy.stats import mannwhitneyu

from nn_proj.common.datasets import load_NT_tasks, prep_for_trainer
from nn_proj.models.evidential import EvidentialConfig, EvidentialWrapper, HFEvidentialSeqClassifier
from nn_proj.models.evidential.evidential import predict_evidential

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASES = ["A", "C", "G", "T"]


def shuffle_seq(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(BASES) for _ in range(length))


def build_variant(test_ds, kind: str, seed: int):
    rng = random.Random(seed)
    seqs = list(test_ds["sequence"])
    if kind == "real":
        new_seqs = seqs
    elif kind == "shuffled":
        new_seqs = [shuffle_seq(s, rng) for s in seqs]
    elif kind == "random_dna":
        new_seqs = [random_dna(len(s), rng) for s in seqs]
    else:
        raise ValueError(kind)
    df = test_ds.to_pandas()
    df["sequence"] = new_seqs
    from datasets import Dataset, ClassLabel
    ds = Dataset.from_pandas(df, preserve_index=False)
    if "label" in ds.column_names and not isinstance(ds.features.get("labels"), ClassLabel):
        pass
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--evidence_activation", default="softplus")
    ap.add_argument("--model_max_length", type=int, default=75)
    ap.add_argument("--out", default="data_gen/label_noise/evidential_ood_retest.csv")
    args = ap.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M", model_max_length=args.model_max_length,
        padding_side="right", use_fast=True, trust_remote_code=True,
    )
    # DNABERT2 is a custom (trust_remote_code) architecture: a mid-training
    # `checkpoint-N` dir doesn't carry the remote-code registration needed
    # for AutoModel.from_pretrained to resolve it directly. Load the
    # architecture from the original hub id (which has the remote code),
    # then load this checkpoint's weights by hand -- same pattern
    # nn_proj/models/DNABERT2/scaling.py and inference.py already use.
    config = transformers.AutoConfig.from_pretrained(
        "zhihan1996/DNABERT-2-117M", num_labels=2, trust_remote_code=True,
    )
    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "zhihan1996/DNABERT-2-117M", config=config, trust_remote_code=True,
    )
    state_path = os.path.join(args.checkpoint, "model.safetensors")
    sd = load_file(state_path)
    prefix = "wrapper.base."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    base_model.load_state_dict(sd, strict=True)
    base_model = base_model.to(DEVICE)
    cfg = EvidentialConfig(num_classes=2, evidence_activation=args.evidence_activation)
    model = HFEvidentialSeqClassifier(EvidentialWrapper(base_model, cfg)).to(DEVICE)
    model.eval()

    raw_test = load_NT_tasks(task="promoter_all", split="test")

    results = {}
    for kind in ["real", "shuffled", "random_dna"]:
        variant = build_variant(raw_test, kind, seed=42)
        ds, collator = prep_for_trainer(variant, tokenizer, metadata_cols=())
        rows = predict_evidential(model, ds, collator, batch_size=64, outfile=None)
        df = pd.DataFrame(rows)
        acc = (df["pred"] == df["labels"]).mean()
        results[kind] = df
        print(f"{kind}: n={len(df)} acc={acc:.4f} vacuity_mean={df['vacuity'].mean():.4f} "
              f"vacuity_std={df['vacuity'].std():.4f}")

    combined = pd.concat([df.assign(kind=k) for k, df in results.items()], ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    for ood_kind in ["shuffled", "random_dna"]:
        u, p = mannwhitneyu(results[ood_kind]["vacuity"], results["real"]["vacuity"], alternative="greater")
        print(f"\nMann-Whitney (vacuity[{ood_kind}] > vacuity[real]): U={u:.1f}, p={p:.6g}")


if __name__ == "__main__":
    main()
