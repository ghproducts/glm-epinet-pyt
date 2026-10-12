#!/usr/bin/env python3
"""Dump base-model logits + labels for one (rank, tier) to an .npz.

Why this exists: nn_proj/models/NT_transformer/scaling.py fits the temperature
on `train_dataset.train_test_split(test_size=0.1)["test"]` of whatever file it
is handed. The driver hands it train.csv, so the "val_dataset" it fits on is a
10% slice of the TRAINING data -- in-sample. That produced T=1.477 for phylum
(which happened to help, since phylum is over-confident on id_test) and T=0.892
for family (which doubled family's ECE, 0.0227 -> 0.0451, by sharpening a model
that was already slightly under-confident).

With logits on disk, temperature can be fitted on a genuinely held-out split and
swept analytically -- no extra forward pass per candidate T, which is what the
current re-run-inference-per-method design forces.

Preprocessing is taken from scaling.py verbatim (same tokenizer, same
model_max_length, same load_local_dataset call with taxa_rank/taxa_df remapping)
EXCEPT the 10% split, so the whole tier is dumped.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np, torch, transformers
from safetensors.torch import load_file

from nn_proj.common.datasets import load_local_dataset, prep_for_trainer
from nn_proj.models.NT_transformer.scaling import collect_logits_labels

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
DATA = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
CKPT = {"phylum": "checkpoints/pbsim_clean/phylum/checkpoint-57734",
        "family": "checkpoints/pbsim_clean/family/checkpoint-57734"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True, choices=["phylum", "family"])
    ap.add_argument("--tier", required=True)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="results/logits")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.rank}_{args.tier}.npz"
    if dest.exists():
        print(f"{dest} exists, nothing to do"); return

    cfg = transformers.AutoConfig.from_pretrained(CKPT[args.rank], trust_remote_code=True)
    tok = transformers.AutoTokenizer.from_pretrained(
        HUB, model_max_length=args.max_len, padding_side="right",
        use_fast=True, trust_remote_code=True)
    tok.eos_token = tok.pad_token

    ds = load_local_dataset(path=str(DATA / f"{args.tier}.csv"), encode_labels=True,
                            rank=args.rank, taxa_df=str(DATA / "full_basic_lineage.csv"))
    ds, coll = prep_for_trainer(ds, tok, max_length=args.max_len,
                                metadata_cols=("taxid", "split"))

    m = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB, config=cfg, trust_remote_code=True)
    m.load_state_dict(load_file(os.path.join(CKPT[args.rank], "model.safetensors")), strict=True)

    logits, labels = collect_logits_labels(model=m, dataset=ds, collate_fn=coll,
                                           batch_size=args.bs)
    np.savez_compressed(dest, logits=logits.numpy().astype(np.float32),
                        labels=labels.numpy().astype(np.int64))
    print(f"wrote {dest}  logits {tuple(logits.shape)}  labels {tuple(labels.shape)}")


if __name__ == "__main__":
    main()
