#!/usr/bin/env python3
"""Leave-phyla-out at the dataset's native read length (6kb), not 150bp.

v1/v2 ran at 150bp to match the darkmatter reads. That constraint does not
apply to this experiment, and it mattered: the truncation control showed the
phylum-rank model reaches 86.6% accuracy at 6000bp against 41.1% at 150bp, so
the earlier null was measured where the model is weak. Here everything runs at
6000bp -- the same length the original pbsim checkpoints were trained on.

Tiers (identical construction to v2, only the length differs):
  id_test  training species, held-out reads   (in-distribution)
  near     id_novel_genus                     (novel genus)
  far      ood_nonbacterial, reads >=6000bp   (novel superkingdom)

The far tier is restricted to reads >=6000bp so that every tier has exactly
6000bp windows: 88.3% of non-bacterial reads qualify, and without the filter
short viral reads would make read LENGTH itself a giveaway.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
POOL = ["1224", "201174", "1239", "976", "1117", "200940", "203682", "2818505",
        "203691", "57723"]


def collect(path, sp2ph, keep=None, min_len=0):
    out = []
    for ch in pd.read_csv(path, usecols=["sequence", "taxid"], chunksize=100_000):
        ch["ph"] = ch.taxid.astype(str).map(sp2ph)
        ch = ch.dropna(subset=["ph"])
        if keep is not None:
            ch = ch[ch.ph.isin(keep)]
        if min_len:
            ch = ch[ch.sequence.str.len() >= min_len]
        out.append(ch)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def win(seq, n, L, rng):
    if len(seq) <= L:
        return [seq] * n
    return [seq[s:s + L] for s in rng.integers(0, len(seq) - L, size=n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=6000)
    ap.add_argument("--per-class", type=int, default=4000)
    ap.add_argument("--max-eval", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=Path, default=here.parent / "leave_phyla_out_v3")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    L = args.read_len

    lin = pd.read_csv(DATA / "full_basic_lineage.csv")
    sp2ph = dict(zip(lin.species.astype(str), lin.phylum.astype(str)))
    design = pd.read_csv(here / "subsets" / "design.csv")
    design.to_csv(args.outdir / "design.csv", index=False)

    src = collect(DATA / "train.csv", sp2ph, set(POOL), min_len=L)
    print("train reads >=%dbp per phylum: %s" % (L, {p: int((src.ph == p).sum()) for p in POOL}))
    wins = {}
    for p in POOL:
        s = src[src.ph == p].sequence
        reps = int(np.ceil(args.per_class / len(s)))
        w = []
        for seq in s:
            w.extend(win(seq, reps, L, rng))
        rng.shuffle(w); wins[p] = w[:args.per_class]

    sub = args.outdir / "subsets"; sub.mkdir(exist_ok=True)
    for _, r in design.iterrows():
        i = int(r.subset); ph = r.phyla.split("|")
        df = pd.concat([pd.DataFrame({"labels": k, "sequence": wins[p], "phylum": p})
                        for k, p in enumerate(ph)], ignore_index=True)
        df.sample(frac=1, random_state=args.seed + i).to_csv(sub / f"subset_{i}.csv", index=False)
        print(f"subset_{i}: {ph} -> {len(df)} rows")

    for tier, fname, keep in [("id_test", "id_test.csv", set(POOL)),
                              ("near", "id_novel_genus.csv", None),
                              ("far", "ood_nonbacterial.csv", None)]:
        d = collect(DATA / fname, sp2ph, keep, min_len=L)
        if len(d) > args.max_eval:
            d = d.sample(args.max_eval, random_state=args.seed)
        ev = pd.DataFrame({"label": 0,
                           "sequence": [win(s, 1, L, rng)[0] for s in d.sequence],
                           "true_phylum": d.ph.values, "taxid": d.taxid.values, "tier": tier})
        ev.to_csv(args.outdir / f"eval_{tier}_{L}bp.csv", index=False)
        print(f"eval_{tier}: {len(ev)} reads, {ev.true_phylum.nunique()} phyla, "
              f"{ev.taxid.nunique()} species, len {ev.sequence.str.len().min()}-{ev.sequence.str.len().max()}")


if __name__ == "__main__":
    main()
