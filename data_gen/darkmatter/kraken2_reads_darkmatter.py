#!/usr/bin/env python3
"""Kraken2 on dark matter, with a database built from the SIMULATED READS.

The existing database (kraken2_db_train) was built from the training species'
reference GENOMES. That gives Kraken2 complete, error-free genome sequence,
while the GLM retrieval reference is pbsim reads -- so the two methods were
never given the same information.

This rebuilds the database from train.csv, i.e. the exact simulated reads the
GLM indexes. Both methods then see identical reference data and differ only in
how they match: exact k-mer membership versus embedding distance.

Reported per `up`: fraction of reads left UNCLASSIFIED, which is Kraken2's
novelty signal, plus the trend across `up` computed the same way as every GLM
result (replicate means, least squares).
"""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

KRAKEN = "/home/glh52/miniconda3/envs/kraken2/bin"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
RANKS = ["species", "genus", "family", "order", "class", "phylum", "kingdom", "superkingdom"]


def write_taxonomy(lin: pd.DataFrame, tax_dir: Path):
    tax_dir.mkdir(parents=True, exist_ok=True)
    parent, rank = {1: 1}, {1: "no rank"}
    for _, r in lin.iterrows():
        chain = [(int(r[c]), c) for c in RANKS if pd.notna(r[c]) and int(r[c]) > 0]
        for i, (t, rk) in enumerate(chain):
            rank.setdefault(t, rk)
            if t not in parent:
                parent[t] = chain[i + 1][0] if i + 1 < len(chain) else 1
    with (tax_dir / "nodes.dmp").open("w") as f:
        for t, p in parent.items():
            f.write(f"{t}\t|\t{p}\t|\t{rank.get(t,'no rank')}\t|\t-\t|\n")
    with (tax_dir / "names.dmp").open("w") as f:
        for t in parent:
            f.write(f"{t}\t|\ttaxon_{t}\t|\t-\t|\tscientific name\t|\n")
    print(f"taxonomy: {len(parent)} nodes", flush=True)


def classify(db: Path, seqs, tag: str, tmp: Path, threads: int):
    fa = tmp / f"{tag}.fa"
    with fa.open("w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{tag}{i}\n{s}\n")
    out = tmp / f"{tag}.kraken"
    subprocess.run([f"{KRAKEN}/kraken2", "--db", str(db), "--threads", str(threads),
                    "--output", str(out), "--report", str(tmp / f"{tag}.report"), str(fa)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cls = np.array([l.startswith("C") for l in out.open()])
    return cls


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--db", type=Path, default=Path("/scratch/home/glh52/kraken2_db_reads"))
    ap.add_argument("--tmp", type=Path, default=Path("/scratch/home/glh52/kraken2_tmp_reads"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)

    lin = pd.read_csv(D / "full_basic_lineage.csv", index_col=0)
    if not args.skip_build:
        write_taxonomy(lin, args.db / "taxonomy")
        lib = args.db / "library"; lib.mkdir(parents=True, exist_ok=True)
        n = 0
        with (lib / "reads.fna").open("w") as out:
            for chunk in pd.read_csv(D / "train.csv", usecols=["sequence", "taxid"], chunksize=50_000):
                for s, t in zip(chunk.sequence, chunk.taxid):
                    out.write(f">r{n}|kraken:taxid|{int(t)}\n{s}\n"); n += 1
        print(f"library: {n:,} SIMULATED READS (not genomes)", flush=True)
        subprocess.run([f"{KRAKEN}/kraken2-build", "--add-to-library", str(lib / "reads.fna"),
                        "--db", str(args.db), "--threads", str(args.threads)], check=True)
        subprocess.run([f"{KRAKEN}/kraken2-build", "--build", "--db", str(args.db),
                        "--threads", str(args.threads)], check=True)
        print("database built", flush=True)

    idt = pd.read_csv(D / "id_test.csv", usecols=["sequence"]).sequence
    cls_id = classify(args.db, idt.values, "id_test", args.tmp, args.threads)
    print(f"\nid_test: {cls_id.mean():.4f} classified ({len(cls_id):,} reads)", flush=True)

    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    dm = dm[dm.up != 0.5].reset_index(drop=True)
    cls_dm = classify(args.db, dm.sequence.values, "darkmatter", args.tmp, args.threads)
    dm = dm.assign(unclassified=(~cls_dm).astype(float))

    by = dm.groupby("up").unclassified.mean()
    rep = dm.groupby(["up", "sample"]).unclassified.mean().reset_index()
    sl, _, _, pv, se = stats.linregress(rep.up, rep.unclassified)
    dm[["up", "sample", "unclassified"]].to_csv("results/kraken2_reads_darkmatter.csv", index=False)
    print(f"\n=== Kraken2, database built from simulated reads ===")
    print(f"  id_test unclassified      {1-cls_id.mean():.4f}")
    for u, v in by.items():
        print(f"  dark matter up={u:<5} unclassified {v:.4f}")
    print(f"\n  slope of unclassified vs up = {sl:+.4f} +/- {se:.4f}  (p={pv:.4g})")
    print(f"  per-replicate means:")
    for _, r in rep.iterrows():
        print(f"    up={r.up:<5} rep={int(r['sample'])}  {r.unclassified:.4f}")


if __name__ == "__main__":
    main()
