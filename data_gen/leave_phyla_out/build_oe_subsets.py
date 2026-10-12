#!/usr/bin/env python3
"""Outlier-exposure / background-class variants of the leave-phyla-out subsets.

Plain cross-entropy never shows the model anything outside its label space, so
nothing pressures it to reserve probability mass for unknowns -- one of the
standard explanations for why softmax confidence fails at OOD detection
(Hendrycks, Mazeika & Dietterich 2019). Here each model gets an explicit
background class trained on phyla it will never be tested on, and novelty is
read off as P(background).

Three-way split of the 10-phylum pool per subset:
  5 in-label phyla     -> classes 0..4          (in-distribution at test time)
  2 exposure phyla     -> class 5, "background" (seen in training as "other")
  3 remaining phyla    -> never seen            (truly novel at test time)
Plus the 11 non-pool phyla present in the eval set, also never seen. Scoring
excludes the 2 exposure phyla so the test measures generalisation to novelty
rather than recall of the exposure set.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

TRAIN_CSV = "/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
LINEAGE = "/scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv"
POOL = ["1224", "201174", "1239", "976", "1117", "200940", "203682", "2818505",
        "203691", "57723"]


def windows(seq, n, length, rng):
    if len(seq) <= length:
        return [seq] * n
    return [seq[s:s + length] for s in rng.integers(0, len(seq) - length, size=n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--per-class", type=int, default=8000)
    ap.add_argument("--n-exposure", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    lin = pd.read_csv(LINEAGE)
    sp2ph = dict(zip(lin.species.astype(str), lin.phylum.astype(str)))
    design = pd.read_csv(here / "subsets" / "design.csv")

    buckets = {p: [] for p in POOL}
    for ch in pd.read_csv(TRAIN_CSV, usecols=["sequence", "taxid"], chunksize=200_000):
        ch["ph"] = ch.taxid.astype(str).map(sp2ph)
        ch = ch[ch.ph.isin(set(POOL))]
        for p, g in ch.groupby("ph"):
            buckets[p].append(g.sequence)
    src = {p: pd.concat(v, ignore_index=True) for p, v in buckets.items()}

    wins = {}
    for p in POOL:
        s = src[p]
        reps = int(np.ceil(args.per_class / len(s)))
        w = []
        for seq in s:
            w.extend(windows(seq, reps, args.read_len, rng))
        rng.shuffle(w)
        wins[p] = w[:args.per_class]

    out_dir = here / "subsets_oe"; out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in design.iterrows():
        i = int(r.subset)
        inlab = r.phyla.split("|")
        rest = [p for p in POOL if p not in inlab]
        expo = sorted(rng.choice(rest, size=args.n_exposure, replace=False).tolist())
        novel = [p for p in rest if p not in expo]

        recs = [pd.DataFrame({"labels": lab, "sequence": wins[p], "phylum": p})
                for lab, p in enumerate(inlab)]
        # background class gets the same per-class budget, split across the
        # exposure phyla, so it does not dominate the loss
        per = args.per_class // len(expo)
        for p in expo:
            recs.append(pd.DataFrame({"labels": len(inlab), "sequence": wins[p][:per], "phylum": p}))
        df = pd.concat(recs, ignore_index=True).sample(frac=1, random_state=args.seed + i)
        df.to_csv(out_dir / f"subset_{i}.csv", index=False)
        rows.append({"subset": i, "in_label": "|".join(inlab), "exposure": "|".join(expo),
                     "held_novel": "|".join(novel), "n": len(df)})
        print(f"subset_{i}: in={inlab} expo={expo} novel={novel} rows={len(df)}")
    pd.DataFrame(rows).to_csv(out_dir / "design_oe.csv", index=False)
    print(f"\nwrote {out_dir}/design_oe.csv")


if __name__ == "__main__":
    main()
