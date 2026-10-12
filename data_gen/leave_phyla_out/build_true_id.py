#!/usr/bin/env python3
"""A genuinely in-distribution evaluation tier.

Every read in eval_pbsim_150bp.csv comes from `id_novel_genus`, i.e. from a
genus held out of training. So what has been labelled "ID" throughout is really
"novel genus, known phylum", and every AUROC so far has been novel-organism vs
novel-organism -- a harder contrast than the standard ID-vs-OOD setup, which
likely depresses every score.

This builds the missing tier: held-out 150bp windows from the SAME species the
models trained on. Windows whose exact sequence appears in any training subset
are dropped, so nothing here was seen during training. Windows may originate
from the same source read at a different offset; that is a different genomic
locus and is the same relationship two overlapping reads have in a real
dataset, but it is noted rather than hidden.

No need to re-run PBsim: train.csv holds 269,808 reads and training consumed at
most 8,000 windows per phylum, so there is ample unused material (and the
reference genomes are not on disk anyway).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

TRAIN_CSV = "/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
LINEAGE = "/scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv"
POOL = ["1224", "201174", "1239", "976", "1117", "200940", "203682", "2818505",
        "203691", "57723"]


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--per-phylum", type=int, default=500)
    ap.add_argument("--seed", type=int, default=99)   # != the build_subsets seed
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    used = set()
    for p in sorted((here / "subsets").glob("subset_*.csv")):
        used |= set(pd.read_csv(p).sequence)
    print(f"training windows to exclude: {len(used):,}")

    lin = pd.read_csv(LINEAGE)
    sp2ph = dict(zip(lin.species.astype(str), lin.phylum.astype(str)))
    buckets = {}
    for ch in pd.read_csv(TRAIN_CSV, usecols=["sequence", "taxid"], chunksize=200_000):
        ch["ph"] = ch.taxid.astype(str).map(sp2ph)
        ch = ch[ch.ph.isin(set(POOL))]
        for p, g in ch.groupby("ph"):
            buckets.setdefault(p, []).append(g[["sequence", "taxid"]])

    rows, n_drop = [], 0
    for p in POOL:
        df = pd.concat(buckets[p], ignore_index=True)
        idx = rng.permutation(len(df))
        got = 0
        for j in idx:
            if got >= args.per_phylum:
                break
            s = df.sequence.iloc[j]
            if len(s) <= args.read_len:
                continue
            st = int(rng.integers(0, len(s) - args.read_len))
            w = s[st:st + args.read_len]
            if w in used:
                n_drop += 1
                continue
            rows.append({"label": 0, "sequence": w, "true_phylum": p,
                         "taxid": df.taxid.iloc[j], "tier": "true_id"})
            got += 1
        print(f"  {p}: {got} windows")
    out = pd.DataFrame(rows)
    out.to_csv(here / f"eval_trueid_{args.read_len}bp.csv", index=False)
    print(f"\ndropped {n_drop} windows that collided with training")
    print(f"wrote {here / f'eval_trueid_{args.read_len}bp.csv'} ({len(out)} reads, "
          f"{out.true_phylum.nunique()} phyla, {out.taxid.nunique()} species)")


if __name__ == "__main__":
    main()
