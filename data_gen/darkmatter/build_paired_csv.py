#!/usr/bin/env python3
"""Extract MATCHED mate pairs from the raw dark matter fastqs.

build_csv.py cannot be used for this: it reservoir-samples each file with a
per-file seed, so its `paired_1` and `paired_2` rows are unrelated fragments.
The raw files are mate-synchronised (@S0R0/1 <-> @S0R0/2, same record order),
so pairs are recovered by streaming both files in lockstep and sampling the
same record indices from each.

Writes one row per FRAGMENT with both mates, so downstream code can test
whether 2x150bp of the same fragment helps -- either concatenated (mate1 +
revcomp(mate2), giving the model 300bp of context) or as two separate
embeddings averaged (2x the evidence, still 150bp of context). Those are
different hypotheses and the columns here support both.
"""
from __future__ import annotations
import argparse, gzip, random, re
from pathlib import Path
import pandas as pd

PAT = re.compile(r"up\.(?P<up>[0-9.]+)\.mut_rate[^_]*_sample_(?P<sample>\d+)_paired_1\.fastq\.gz$")
RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s): return s.translate(RC)[::-1]


def stream_pairs(f1: Path, f2: Path, target: int, seed: int):
    """Reservoir-sample `target` matched pairs in one pass over both files."""
    rng = random.Random(seed)
    res, n = [], 0
    with gzip.open(f1, "rt") as a, gzip.open(f2, "rt") as b:
        while True:
            h1 = a.readline()
            if not h1:
                break
            s1 = a.readline().strip(); a.readline(); a.readline()
            h2 = b.readline(); s2 = b.readline().strip(); b.readline(); b.readline()
            if not h2:
                break
            if h1.split("/")[0] != h2.split("/")[0]:      # mate sync guard
                raise RuntimeError(f"mates out of sync at record {n}: {h1!r} vs {h2!r}")
            n += 1
            if len(res) < target:
                res.append((s1, s2))
            else:
                j = rng.randrange(n)
                if j < target:
                    res[j] = (s1, s2)
    return res, n


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--src", type=Path, default=here)
    ap.add_argument("--per-file", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=here / "csv_data" / "darkmatter_pairs.csv")
    args = ap.parse_args()

    rows = []
    files = sorted(p for p in args.src.glob("*_paired_1.fastq.gz") if PAT.search(p.name))
    print(f"{len(files)} paired_1 files")
    for i, f1 in enumerate(files, 1):
        m = PAT.search(f1.name); up = float(m["up"]); samp = int(m["sample"])
        if up == 0.5:
            continue
        f2 = f1.with_name(f1.name.replace("_paired_1", "_paired_2"))
        if not f2.exists():
            print(f"  [skip] no mate file for {f1.name}"); continue
        pairs, total = stream_pairs(f1, f2, args.per_file, seed=hash((up, samp)) & 0xFFFF)
        for s1, s2 in pairs:
            rows.append({"up": up, "sample": samp, "mate1": s1, "mate2": s2,
                         "cat": s1 + revcomp(s2)})
        print(f"  [{i}/{len(files)}] up={up} sample={samp}: {len(pairs):,} pairs of {total:,}", flush=True)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  {len(df):,} fragments")
    print(df.groupby("up").size().to_string())
    print(f"mate1 len {df.mate1.str.len().mean():.1f}  concat len {df.cat.str.len().mean():.1f}")


if __name__ == "__main__":
    main()
