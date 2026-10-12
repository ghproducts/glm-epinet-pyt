#!/usr/bin/env python3
"""Build the leave-phyla-out experiment from the pbsim_id_test dataset.

Differences from v1 (data_gen/leave_phyla_out/build_subsets.py):

* training windows come from pbsim_id_test/train.csv, which has the id_test
  reads removed, so the in-distribution evaluation tier is a genuine holdout
  for these models rather than reads they were trained on;
* four evaluation tiers are produced instead of one, giving a novelty ladder:

      id_test    same species as training, unseen reads   (in-distribution)
      near       id_novel_genus                           (novel genus)
      far        ood_nonbacterial                         (novel superkingdom)

  "near_id" vs "near_ood" is then a split of the `near` tier by whether a
  read's phylum is in a given model's label space.

The phylum pool and the 10x5 design are reused verbatim from v1 so the two
runs are directly comparable.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
LINEAGE = DATA / "full_basic_lineage.csv"
POOL = ["1224", "201174", "1239", "976", "1117", "200940", "203682", "2818505",
        "203691", "57723"]


def windows(seq, n, L, rng):
    if len(seq) <= L:
        return [seq] * n
    return [seq[s:s + L] for s in rng.integers(0, len(seq) - L, size=n)]


def collect(csv_path, sp2ph, keep_phyla=None, usecols=("sequence", "taxid")):
    out = []
    for ch in pd.read_csv(csv_path, usecols=list(usecols), chunksize=200_000):
        ch["ph"] = ch.taxid.astype(str).map(sp2ph)
        if keep_phyla is not None:
            ch = ch[ch.ph.isin(keep_phyla)]
        out.append(ch.dropna(subset=["ph"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def make_eval(df, L, per_read, rng, tier):
    seqs, phs, tx = [], [], []
    for s, p, t in zip(df.sequence, df.ph, df.taxid):
        for w in windows(s, per_read, L, rng):
            seqs.append(w); phs.append(p); tx.append(t)
    out = pd.DataFrame({"label": 0, "sequence": seqs, "true_phylum": phs,
                        "taxid": tx, "tier": tier})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--per-class", type=int, default=8000)
    ap.add_argument("--eval-per-read", type=int, default=1)
    ap.add_argument("--max-eval", type=int, default=6000, help="cap per eval tier")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=Path, default=here.parent / "leave_phyla_out_v2")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    lin = pd.read_csv(LINEAGE)
    sp2ph = dict(zip(lin.species.astype(str), lin.phylum.astype(str)))
    design = pd.read_csv(here / "subsets" / "design.csv")   # reuse v1 design
    design.to_csv(args.outdir / "design.csv", index=False)

    # ---- training windows, from the id_test-free train.csv ----
    src = collect(DATA / "train.csv", sp2ph, set(POOL))
    print("source train reads per phylum:",
          {p: int((src.ph == p).sum()) for p in POOL})
    wins = {}
    for p in POOL:
        s = src[src.ph == p].sequence
        reps = int(np.ceil(args.per_class / len(s)))
        w = []
        for seq in s:
            w.extend(windows(seq, reps, args.read_len, rng))
        rng.shuffle(w)
        wins[p] = w[:args.per_class]

    sub = args.outdir / "subsets"; sub.mkdir(exist_ok=True)
    for _, r in design.iterrows():
        i = int(r.subset); phyla = r.phyla.split("|")
        df = pd.concat([pd.DataFrame({"labels": lab, "sequence": wins[p], "phylum": p})
                        for lab, p in enumerate(phyla)], ignore_index=True)
        df.sample(frac=1, random_state=args.seed + i).to_csv(sub / f"subset_{i}.csv", index=False)
        print(f"subset_{i}: {phyla} -> {len(df)} rows")

    # ---- evaluation tiers ----
    for tier, fname, keep in [("id_test", "id_test.csv", set(POOL)),
                              ("near", "id_novel_genus.csv", None),
                              ("far", "ood_nonbacterial.csv", None)]:
        d = collect(DATA / fname, sp2ph, keep)
        if len(d) > args.max_eval:
            d = d.sample(args.max_eval, random_state=args.seed)
        ev = make_eval(d, args.read_len, args.eval_per_read, rng, tier)
        ev.to_csv(args.outdir / f"eval_{tier}_{args.read_len}bp.csv", index=False)
        print(f"eval_{tier}: {len(ev)} reads, {ev.true_phylum.nunique()} phyla, "
              f"{ev.taxid.nunique()} species")


if __name__ == "__main__":
    main()
