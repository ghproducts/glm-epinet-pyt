#!/usr/bin/env python3
"""Fast spot-check of the memorization diagnostic on hyenaDNA (character-level
tokenization, architecturally furthest from DNABERT2's BPE), at r=0.05 and
r=0.40, seed 1 only -- see .docs/LABEL_NOISE_PILOT.md recommendation #4.

This is a minimal adaptation of data_gen/label_noise/pilot_diagnostics.py for
a backbone whose checkpoint is saved via `save_pretrained(..., safe_serialization=False)`
(pytorch_model.bin, not model.safetensors) and loaded directly via
`AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, ...)` --
exactly how nn_proj/models/hyenaDNA/inference.py loads it, unlike DNABERT2's
"load base weights then load_state_dict from safetensors" pattern. No other
pipeline code is modified.

Only reproduces diagnostic (a), the memorization check, plus (b)/(c) for
completeness since they come for free from the same forward pass.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import transformers
import yaml

from nn_proj.common.datasets import load_local_dataset, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty

REPO_ROOT = Path(__file__).resolve().parents[2]
RATE_TAG = {0.0: "r00", 0.05: "r05", 0.10: "r10", 0.20: "r20", 0.40: "r40"}
MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _model_max_length() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "experiments.yaml").read_text())
    per_base = cfg["backbones"]["hyenaDNA"]["tokens_per_base"]
    seq_length = cfg["training_tasks"]["label_noise_r00"]["seq_length"]
    return max(1, math.ceil(seq_length * per_base))


@torch.no_grad()
def _run_inference(model, dataset, collator, batch_size=32) -> pd.DataFrame:
    def _plain(v):
        return v.item() if isinstance(v, torch.Tensor) else v

    metadata_cols = [c for c in dataset.column_names
                     if c not in ("input_ids", "attention_mask", "token_type_ids", "labels")]
    metas = {col: [_plain(v) for v in dataset[col]] for col in metadata_cols}
    infer_ds = dataset.remove_columns(metadata_cols)

    loader = torch.utils.data.DataLoader(infer_ds, batch_size=batch_size, shuffle=False,
                                          collate_fn=collator)
    model.eval().to(DEVICE)
    rows = []
    idx = 0
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits.detach().cpu().unsqueeze(0)  # [1, B, C]
        unc = compute_uncertainty(logits)
        B = logits.shape[1]
        for i in range(B):
            row = {
                "labels": int(labels[i]),
                "pred": int(unc["predicted_class"][i]),
                "U_total": float(unc["normalized_total_uncertainty"][i]),
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
            }
            for col in metadata_cols:
                row[col] = metas[col][idx]
            rows.append(row)
            idx += 1
    return pd.DataFrame(rows)


def _load_checkpoint_model(checkpoint_dir: Path, num_labels: int):
    config = transformers.AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    # hyenaDNA's train_base.py saves via save_pretrained(safe_serialization=False)
    # (pytorch_model.bin) and inference.py loads straight from the checkpoint
    # dir -- no separate base-weights-then-load_state_dict step needed here.
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir), config=config, trust_remote_code=True,
    )
    return model


def reconstruct_train_only(rate: float, seed: int, tokenizer, max_length: int):
    tag = RATE_TAG[rate]
    csv_path = REPO_ROOT / "data_gen" / "label_noise" / f"csv_data_{tag}" / "train.csv"
    ds = load_local_dataset(path=str(csv_path), encode_labels=True)
    split = ds.train_test_split(test_size=0.1, seed=seed, stratify_by_column="labels")
    train_ds = split["train"]
    tokenized, collator = prep_for_trainer(
        train_ds, tokenizer, max_length=max_length,
        metadata_cols=("sequence_id", "original_label", "noisy"),
    )
    return tokenized, collator


def load_test_set(rate: float, tokenizer, max_length: int):
    tag = RATE_TAG[rate]
    csv_path = REPO_ROOT / "data_gen" / "label_noise" / f"csv_data_{tag}" / "test.csv"
    ds = load_local_dataset(path=str(csv_path), encode_labels=True)
    tokenized, collator = prep_for_trainer(
        ds, tokenizer, max_length=max_length,
        metadata_cols=("sequence_id", "original_label", "noisy"),
    )
    return tokenized, collator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.05, 0.40])
    ap.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    ap.add_argument("--out", type=Path,
                     default=REPO_ROOT / "data_gen" / "label_noise" / "pilot_metrics_hyenaDNA.csv")
    args = ap.parse_args()

    max_length = _model_max_length()
    print(f"model_max_length = {max_length}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME, model_max_length=max_length, padding_side="right",
        use_fast=True, trust_remote_code=True,
    )

    summary_rows = []
    for rate in args.rates:
        tag = RATE_TAG[rate]
        for seed in args.seeds:
            ckpt_dir = args.checkpoint_root / f"seed_{seed}" / "hyenaDNA" / f"label_noise_{tag}" / "base"
            print(f"\n=== rate={rate} seed={seed}  ckpt={ckpt_dir}")
            model = _load_checkpoint_model(ckpt_dir, num_labels=2)

            train_ds, train_collator = reconstruct_train_only(rate, seed, tokenizer, max_length)
            train_df = _run_inference(model, train_ds, train_collator)

            test_ds, test_collator = load_test_set(rate, tokenizer, max_length)
            test_df = _run_inference(model, test_ds, test_collator)

            noisy = train_df[train_df["noisy"] == True]   # noqa: E712
            clean = train_df[train_df["noisy"] == False]  # noqa: E712

            row = {
                "rate": rate,
                "seed": seed,
                "n_train_only": len(train_df),
                "n_flipped": len(noisy),
                "n_clean": len(clean),
                "acc_vs_corrupted_label_on_flipped": (
                    (noisy["pred"] == noisy["labels"]).mean() if len(noisy) else float("nan")
                ),
                "acc_vs_original_label_on_flipped": (
                    (noisy["pred"] == noisy["original_label"]).mean() if len(noisy) else float("nan")
                ),
                "acc_on_clean": (
                    (clean["pred"] == clean["labels"]).mean() if len(clean) else float("nan")
                ),
                "mean_aleatoric_flipped_train": (
                    noisy["U_aleatoric"].mean() if len(noisy) else float("nan")
                ),
                "mean_aleatoric_clean_train": (
                    clean["U_aleatoric"].mean() if len(clean) else float("nan")
                ),
                "mean_aleatoric_test": test_df["U_aleatoric"].mean(),
                "test_accuracy": (test_df["pred"] == test_df["labels"]).mean(),
            }
            row["gap_flipped_minus_clean_train"] = (
                row["mean_aleatoric_flipped_train"] - row["mean_aleatoric_clean_train"]
            )
            summary_rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

            del model
            torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
