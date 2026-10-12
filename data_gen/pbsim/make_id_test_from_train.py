#!/usr/bin/env python3
"""Retrofit an `id_test` split onto an already-generated pbsim dataset.

`reads_to_csv.py` now carves an in-distribution holdout out of the training
species at generation time. For datasets produced before that change -- where
the raw FASTQ output is gone and only the CSVs survive -- this reproduces the
same partition directly from train.csv.

The partitioning rule is identical to the pipeline's: per species, shuffle that
species' reads with a seeded RNG and move a fixed number into id_test. A read
therefore appears in exactly one of the two files, so there is no leakage
between the training set and the in-distribution evaluation set.

Why this split matters: every taxonomic split in this dataset holds out *taxa*,
so `id_novel_genus`, `ood_novel_family` and `ood_nonbacterial` all consist of
organisms the model has never seen. Using `id_novel_genus` as the
"in-distribution" reference in an OOD experiment therefore compares novel
organisms against other novel organisms, which understates how well a method
separates known from unknown. `id_test` is the missing reference.

Writes a NEW dataset directory; the source is left untouched.
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import numpy as np, pandas as pd

OTHER_SPLITS = ["id_novel_genus.csv", "ood_novel_family.csv", "ood_nonbacterial.csv",
                "full_basic_lineage.csv"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("/scratch/home/glh52/glm-epinet/DATA/pbsim"))
    ap.add_argument("--dst", type=Path,
                    default=Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test"))
    ap.add_argument("--per-species", type=int, default=10,
                    help="reads held out per training species (pipeline default: 10)")
    ap.add_argument("--seed", type=int, default=123, help="matches reads_to_csv.py's default")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    for name in OTHER_SPLITS:
        src = args.src / name
        if src.is_file():
            shutil.copy2(src, args.dst / name)
            print(f"copied {name}")

    rng = np.random.default_rng(args.seed)
    src_train = args.src / "train.csv"

    # Pass 1: how many reads each species has, so the holdout can be chosen
    # without holding the whole 2.9GB table in memory.
    counts = {}
    for ch in pd.read_csv(src_train, usecols=["taxid"], chunksize=args.chunksize):
        for t, c in ch.taxid.value_counts().items():
            counts[t] = counts.get(t, 0) + c
    print(f"\n{len(counts)} training species, {sum(counts.values()):,} reads")

    # Choose held-out row positions per species up front.
    held = {}
    for t, n in counts.items():
        k = min(args.per_species, max(0, n - 1))     # never empty a species' training set
        held[t] = set(rng.choice(n, size=k, replace=False).tolist()) if k else set()
    print(f"holding out {sum(len(v) for v in held.values()):,} reads "
          f"({args.per_species}/species target)")

    seen = {t: 0 for t in counts}
    n_tr = n_id = 0
    tr_path, id_path = args.dst / "train.csv", args.dst / "id_test.csv"
    first = True
    for ch in pd.read_csv(src_train, chunksize=args.chunksize):
        pos = ch.taxid.map(lambda t: seen[t])          # placeholder, filled below
        idx = []
        for t in ch.taxid.values:
            idx.append(seen[t]); seen[t] += 1
        ch = ch.assign(_pos=idx)
        mask = [p in held[t] for t, p in zip(ch.taxid.values, ch._pos.values)]
        mask = np.array(mask)
        id_part = ch[mask].drop(columns=["_pos"]).assign(split="id_test")
        tr_part = ch[~mask].drop(columns=["_pos"])
        tr_part.to_csv(tr_path, mode="w" if first else "a", header=first, index=False)
        id_part.to_csv(id_path, mode="w" if first else "a", header=first, index=False)
        n_tr += len(tr_part); n_id += len(id_part)
        first = False

    print(f"\nwrote {tr_path}  ({n_tr:,} reads)")
    print(f"wrote {id_path}  ({n_id:,} reads)")
    print(f"total {n_tr + n_id:,} == source {sum(counts.values()):,} "
          f"-> {'OK' if n_tr + n_id == sum(counts.values()) else 'MISMATCH'}")


if __name__ == "__main__":
    main()
