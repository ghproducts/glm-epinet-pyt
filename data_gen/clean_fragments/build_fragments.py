#!/usr/bin/env python3
"""Clean 1.5kb genome-fragment benchmark for the taxonomic-shift study.

Plan: .docs/plans/TAXA_SHIFT_PLAN.md. Fragments are cut directly from the RefSeq
genome FASTAs -- no read simulator -- which is the standard in the GLM
taxonomic-classification literature (Ren 2019, BERTax, ICCTax, DNABERT-S) and
matches what every backbone was pretrained on.

Species tiers come from data_gen/pbsim/splits/species_splits.csv, which already
satisfies clade exclusion (checked: novel-genus genera absent from training and
their families present; novel-family families absent from training).

Training genomes are split into REGIONS so no test sequence can overlap
training sequence (the earlier id_test leakage came from carving test reads
out of the training file):
  per genome, a randomly placed contiguous 20% window is held out, its first
  half -> validation (temperature fitting), second half -> ID test; the rest is
  training. A fragment never crosses a region boundary or a record boundary.

Plasmid records are excluded (they move between lineages, adding label noise).

Sampling:
  train      ~250k fragments, species weighted by genome size (mirrors the old
             depth-1 read set)
  val / id_test / novel tiers   equal fragments per species (capped), so each
             held-out taxon counts the same; viruses would otherwise vanish
             behind eukaryotic genomes in the non-bacterial tier.

Also writes error-injected copies of every test set at 0.5% and 5% per-base
error (substitution / insertion / deletion in equal parts), following ICCTax's
0 / 0.5 / 5% robustness sweep.
"""
from __future__ import annotations
import argparse, gzip
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path("data_gen/pbsim/raw_sequences")
SPLITS = Path("data_gen/pbsim/splits/species_splits.csv")
L = 1500


def read_records(sp: str):
    """Non-plasmid records of one species' genome(s), uppercased."""
    recs = []
    for f in sorted((SRC / sp).glob("*.fna*")):
        with (gzip.open if f.suffix == ".gz" else open)(f, "rt") as h:
            name, cur = None, []
            for line in h:
                if line.startswith(">"):
                    if name is not None and "plasmid" not in name.lower():
                        recs.append("".join(cur))
                    name, cur = line, []
                else:
                    cur.append(line.strip().upper())
            if name is not None and "plasmid" not in name.lower():
                recs.append("".join(cur))
    return [r for r in recs if len(r) >= L]


def regions(recs, rng):
    """Map each record to (train, val, test) coordinate intervals."""
    tot = sum(len(r) for r in recs)
    start = rng.uniform(0, 0.8) * tot            # held-out 20% window, random placement
    cuts = {"val": (start, start + 0.1 * tot), "test": (start + 0.1 * tot, start + 0.2 * tot)}
    out, off = {"train": [], "val": [], "test": []}, 0
    for i, r in enumerate(recs):
        a, b = off, off + len(r)
        held = []
        for k, (lo, hi) in cuts.items():
            lo2, hi2 = max(a, lo), min(b, hi)
            if hi2 - lo2 >= L:
                out[k].append((i, int(lo2 - a), int(hi2 - a)))
            if hi2 > lo2:
                held.append((lo2 - a, hi2 - a))
        # training = the record minus held-out spans, with an L-bp buffer either side
        segs, pos = [], 0
        for lo, hi in sorted(held):
            if lo - L - pos >= L:
                segs.append((pos, int(lo - L)))
            pos = int(hi + L)
        if len(r) - pos >= L:
            segs.append((pos, len(r)))
        out["train"] += [(i, s, e) for s, e in segs]
        off = b
    return out


def cut(recs, intervals, n, rng):
    lens = np.array([e - s - L + 1 for _, s, e in intervals], float)
    if not len(lens) or lens.sum() <= 0:
        return []
    frags, tries = [], 0
    while len(frags) < n and tries < n * 20:
        tries += 1
        j = rng.choice(len(intervals), p=lens / lens.sum())
        i, s, e = intervals[j]
        st = int(rng.integers(s, e - L + 1))
        w = recs[i][st:st + L]
        if w.count("N") == 0 and set(w) <= set("ACGT"):
            frags.append(w)
    return frags


def mutate(seq, rate, rng):
    out, i = [], 0
    for c in seq:
        if rng.random() < rate:
            k = rng.integers(3)
            if k == 0:   out.append(rng.choice([b for b in "ACGT" if b != c]))   # substitution
            elif k == 1: out.append(c); out.append(rng.choice(list("ACGT")))   # insertion
            # k == 2: deletion
        else:
            out.append(c)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/scratch/home/glh52/glm-epinet/DATA/clean1500"))
    ap.add_argument("--n-train", type=int, default=250_000)
    ap.add_argument("--per-species-test", type=int, default=12)    # id_test / val
    ap.add_argument("--per-species-novel", type=int, default=40)   # novel tiers, capped below
    ap.add_argument("--cap-novel", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(SPLITS); sp = sp[sp.has_sequence]
    sp["species"] = sp["species"].astype(int)

    # ---- training species: regions, then fragments
    tr = sp[sp.split == "train"]
    sizes, recs_by, regs_by = {}, {}, {}
    for k, s in enumerate(tr.species):
        recs = read_records(str(s))
        if not recs: continue
        recs_by[s] = recs; regs_by[s] = regions(recs, rng)
        sizes[s] = sum(len(r) for r in recs)
        if k % 200 == 0: print(f"  regions {k}/{len(tr)}", flush=True)
    keys = list(sizes); w = np.array([sizes[s] for s in keys], float)
    alloc = pd.Series(rng.choice(len(keys), a.n_train, p=w / w.sum())).value_counts()
    rows = {"train": [], "val": [], "id_test": []}
    for j, n in alloc.items():
        s = keys[j]
        rows["train"] += [(s, f) for f in cut(recs_by[s], regs_by[s]["train"], n, rng)]
    for s in keys:
        rows["val"] += [(s, f) for f in cut(recs_by[s], regs_by[s]["val"], a.per_species_test, rng)]
        rows["id_test"] += [(s, f) for f in cut(recs_by[s], regs_by[s]["test"], a.per_species_test, rng)]

    # ---- novel tiers: whole genomes, equal fragments per species, capped
    for tier, name in [("id_novel_genus", "novel_genus"), ("ood_novel_family", "novel_family"),
                       ("ood_nonbacterial", "nonbacterial")]:
        ss = sp[sp.split == tier].species.tolist()
        per = max(1, min(a.per_species_novel, a.cap_novel // max(len(ss), 1)))
        rows[name] = []
        for s in ss:
            recs = read_records(str(s))
            if recs:
                rows[name] += [(s, f) for f in cut(recs, [(i, 0, len(r)) for i, r in enumerate(recs)], per, rng)]
        print(f"  {name}: {len(rows[name])} fragments from {len(ss)} species ({per}/species)", flush=True)

    for k, v in rows.items():
        df = pd.DataFrame(v, columns=["taxid", "sequence"])
        df["labels"] = df.taxid; df["split"] = k
        df = df.sample(frac=1, random_state=a.seed)[["labels", "sequence", "taxid", "split"]]
        df.to_csv(a.out / f"{k}.csv", index=False)
        print(f"wrote {k}: {len(df):,} fragments, {df.taxid.nunique()} species", flush=True)
        if k not in ("train", "val"):
            for rate in (0.005, 0.05):
                e = df.copy(); e["sequence"] = [mutate(x, rate, rng) for x in e.sequence]
                e.to_csv(a.out / f"{k}_err{rate*100:g}.csv", index=False)
    # lineage table for taxa_rank remapping, same format as before
    # same layout as the old file: unnamed integer index, `species` as a column
    sp.drop(columns=["species_str", "has_sequence", "split"]).reset_index(drop=True).to_csv(a.out / "full_basic_lineage.csv")


if __name__ == "__main__":
    main()
