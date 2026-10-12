#!/usr/bin/env python3
"""Build leave-phyla-out training sets and a ground-truth evaluation set.

Motivation: the darkmatter `up` trend at phylum rank (confidence RISES with the
unknown fraction) may be explained by the unknown fraction being drawn mostly
from phyla already in the label space. That is testable by making the LABEL
SPACE the independent variable: train several models on disjoint-ish subsets of
phyla and check whether the `up` trend depends on which phyla each model knows.
If the trend is identical regardless of label space, the effect is
compositional; if it varies with membership, the hypothesis is supported.

Two design choices that fix problems in earlier runs:

* Reads are cut to 150bp to MATCH the darkmatter reads. Every previous result
  was confounded by a ~6000bp-train / ~150bp-test shift severe enough to put
  the model below its usable floor (0.83% accuracy vs 0.23% chance at family
  rank). Training at the evaluation length removes that entirely.
* Classes are balanced to a fixed number of windows each, so a model's
  behaviour reflects which phyla it knows rather than their relative abundance
  (abundance is itself part of what is under investigation).

Outputs, under data_gen/leave_phyla_out/:
  subsets/subset_<i>.csv   training data, 5 phyla, `labels` pre-encoded 0..4
  subsets/design.csv       which phyla each subset contains
  eval_pbsim_150bp.csv     held-out pbsim reads w/ `true_phylum` (ground truth
                           positive control: for each model, reads from its 5
                           phyla are ID, the rest are genuinely OOD)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

TRAIN_CSV = "/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
LINEAGE = "/scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv"
# 10 most read-abundant phyla; every one has >=4k source reads, enough to reach
# the per-class window target without excessive repetition of any single read.
POOL = ["1224", "201174", "1239", "976", "1117", "200940", "203682", "2818505",
        "203691", "57723"]


def balanced_design(pool, k, n_sets, rng, tries=20000):
    """n_sets subsets of size k, each phylum used equally often, pairwise
    co-occurrence as uniform as possible (keeps the inclusion matrix
    well-conditioned for regressing trend on membership)."""
    best, best_score = None, None
    per = n_sets * k // len(pool)
    for _ in range(tries):
        slots = np.repeat(np.arange(len(pool)), per)
        rng.shuffle(slots)
        sets, ok = [], True
        pool_slots = list(slots)
        for _ in range(n_sets):
            chosen, used = [], set()
            for idx in list(pool_slots):
                if len(chosen) == k: break
                if idx not in used:
                    chosen.append(idx); used.add(idx); pool_slots.remove(idx)
            if len(chosen) < k: ok = False; break
            sets.append(sorted(chosen))
        if not ok: continue
        M = np.zeros((n_sets, len(pool)), int)
        for i, s in enumerate(sets):
            M[i, s] = 1
        co = M.T @ M
        score = co[~np.eye(len(pool), dtype=bool)].std()
        if best_score is None or score < best_score:
            best, best_score = sets, score
    return [[pool[i] for i in s] for s in best], best_score


def windows(seq, n, length, rng):
    """n random `length`-bp windows from one read (all of it if too short)."""
    if len(seq) <= length:
        return [seq] * n
    return [seq[s:s + length] for s in rng.integers(0, len(seq) - length, size=n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--per-class", type=int, default=8000, help="windows per phylum per subset")
    ap.add_argument("--n-subsets", type=int, default=10)
    ap.add_argument("--k", type=int, default=5, help="phyla per subset")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=Path, default=here)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    lin = pd.read_csv(LINEAGE)
    sp2ph = dict(zip(lin.species.astype(str), lin.phylum.astype(str)))

    sets, score = balanced_design(POOL, args.k, args.n_subsets, rng)
    print(f"design: {args.n_subsets} subsets of {args.k}; co-occurrence sd={score:.2f}")
    counts = {p: sum(p in s for s in sets) for p in POOL}
    print("appearances per phylum:", counts)

    # collect source reads for the pooled phyla only
    need = set(POOL)
    buckets = {p: [] for p in POOL}
    for ch in pd.read_csv(TRAIN_CSV, usecols=["sequence", "taxid"], chunksize=200_000):
        ch["ph"] = ch.taxid.astype(str).map(sp2ph)
        ch = ch[ch.ph.isin(need)]
        for p, g in ch.groupby("ph"):
            buckets[p].append(g.sequence)
    src = {p: pd.concat(v, ignore_index=True) for p, v in buckets.items()}
    print("source reads:", {p: len(v) for p, v in src.items()})

    # one window pool per phylum, reused across subsets so a phylum is identical
    # wherever it appears -- only the label space differs between models
    wins = {}
    for p, s in src.items():
        reps = int(np.ceil(args.per_class / len(s)))
        w = []
        for seq in s:
            w.extend(windows(seq, reps, args.read_len, rng))
        rng.shuffle(w)
        wins[p] = w[:args.per_class]
        print(f"  {p}: {len(src[p])} reads -> {len(wins[p])} windows ({reps}/read)")

    sub_dir = args.outdir / "subsets"; sub_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(sets):
        recs = []
        for lab, p in enumerate(s):
            recs.append(pd.DataFrame({"labels": lab, "sequence": wins[p], "phylum": p}))
        df = pd.concat(recs, ignore_index=True).sample(frac=1, random_state=args.seed + i)
        df.to_csv(sub_dir / f"subset_{i}.csv", index=False)
        rows.append({"subset": i, "phyla": "|".join(s), "n": len(df)})
        print(f"subset_{i}: {s} -> {len(df)} rows")
    pd.DataFrame(rows).to_csv(sub_dir / "design.csv", index=False)

    # ground-truth eval set: held-out pbsim reads, already 150bp, with true
    # phylum carried as metadata (label is a placeholder -- for a model that
    # excludes a read's phylum there is no correct class, same convention as
    # the darkmatter CSV)
    ev = pd.read_csv(args.outdir.parent / "pbsim" / "truncation_control" / f"id_novel_genus_{args.read_len}bp.csv")
    ev["true_phylum"] = ev.taxid.astype(str).map(sp2ph)
    ev = ev.dropna(subset=["true_phylum"])
    out = ev[["sequence", "true_phylum", "taxid"]].copy()
    out.insert(0, "label", 0)
    out.to_csv(args.outdir / f"eval_pbsim_{args.read_len}bp.csv", index=False)
    print(f"\neval set: {len(out)} reads, {out.true_phylum.nunique()} phyla "
          f"-> {args.outdir / f'eval_pbsim_{args.read_len}bp.csv'}")
    print(out.true_phylum.value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
