#!/usr/bin/env python3
"""Kraken2 as a reference point for ID vs OOD detection.

The neural scores top out around 0.57 AUROC even for a superkingdom-level
shift, which raises the obvious question: is this task actually hard, or are
the models failing at something a standard tool solves easily? Kraken2 answers
it. Its classified/unclassified decision is exactly a novelty signal -- a read
whose k-mers are absent from the database gets no assignment.

The database is built from the reference genomes of the TRAINING species only,
so it has the same information the models were trained on. Queries are the same
three tiers:
  id_test  training species   -> should classify
  near     novel genus        -> known phylum, unseen organism
  far      non-bacterial      -> should be unclassified

Taxonomy is generated from full_basic_lineage.csv rather than downloaded from
NCBI: kraken2-build needs nodes.dmp/names.dmp, and the lineage table already
has species->genus->...->superkingdom for every genome here.
"""
from __future__ import annotations
import argparse, subprocess, shutil
from pathlib import Path
import numpy as np, pandas as pd

KRAKEN = "/home/glh52/miniconda3/envs/kraken2/bin"
RANKS = ["species", "genus", "family", "order", "class", "phylum", "kingdom", "superkingdom"]


def write_taxonomy(lin: pd.DataFrame, tax_dir: Path):
    """Minimal nodes.dmp / names.dmp covering every taxon in the lineage table."""
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
    print(f"taxonomy: {len(parent)} nodes -> {tax_dir}")


def build_db(species, raw_dir: Path, db: Path, threads: int):
    lib = db / "library"; lib.mkdir(parents=True, exist_ok=True)
    added = 0
    with (lib / "added.fna").open("w") as out:
        for sp in species:
            for fna in (raw_dir / sp).glob("*.fna"):
                for line in fna.open():
                    # kraken2 reads the taxid from the header tag
                    out.write(f">{line[1:].strip().split()[0]}|kraken:taxid|{sp}\n"
                              if line.startswith(">") else line)
                added += 1
    print(f"library: {added} genome files from {len(species)} species")
    subprocess.run([f"{KRAKEN}/kraken2-build", "--add-to-library", str(lib / "added.fna"),
                    "--db", str(db), "--threads", str(threads)], check=True)
    subprocess.run([f"{KRAKEN}/kraken2-build", "--build", "--db", str(db),
                    "--threads", str(threads)], check=True)


def classify(db: Path, seqs, tag: str, tmp: Path, threads: int):
    fa = tmp / f"{tag}.fa"
    with fa.open("w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{tag}{i}\n{s}\n")
    out = tmp / f"{tag}.kraken"
    subprocess.run([f"{KRAKEN}/kraken2", "--db", str(db), "--threads", str(threads),
                    "--output", str(out), "--report", str(tmp / f"{tag}.report"), str(fa)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cls, kmers = [], []
    for line in out.open():
        p = line.rstrip("\n").split("\t")
        cls.append(p[0] == "C")
        # LCA mapping field: count k-mers assigned to a real taxon vs to 0 (no hit)
        hit = tot = 0
        for chunk in p[4].split():
            t, n = chunk.rsplit(":", 1)
            if n.isdigit():
                tot += int(n)
                if t not in ("0", "A"):
                    hit += int(n)
        kmers.append(hit / tot if tot else 0.0)
    return np.array(cls), np.array(kmers)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--v", default="v3", choices=["v2", "v3"], help="which eval build to use")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--db", type=Path, default=Path("/scratch/home/glh52/kraken2_db_train"))
    args = ap.parse_args()

    root = here.parent / f"leave_phyla_out_{args.v}"
    L = "6000bp" if args.v == "v3" else "150bp"
    lin = pd.read_csv("/scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv")
    tmp = Path("/scratch/home/glh52/kraken2_tmp"); tmp.mkdir(exist_ok=True)

    if not (args.db / "hash.k2d").is_file():
        write_taxonomy(lin, args.db / "taxonomy")
        train_sp = sorted(set(open("data_gen/pbsim/splits/train_species.txt").read().split()))
        build_db(train_sp, Path("data_gen/pbsim/raw_sequences"), args.db, args.threads)
    else:
        print(f"using existing db at {args.db}")

    rows = {}
    for tier in ["id_test", "near", "far"]:
        d = pd.read_csv(root / f"eval_{tier}_{L}.csv")
        cls, km = classify(args.db, d.sequence.values, tier, tmp, args.threads)
        rows[tier] = dict(n=len(d), classified=cls.mean(), kmer_frac=km.mean())
        print(f"{tier:8s}: classified {cls.mean():.3f}  mean k-mer hit fraction {km.mean():.4f}")
        np.save(tmp / f"{tier}_km.npy", km)

    from sklearn.metrics import roc_auc_score
    idk = np.load(tmp / "id_test_km.npy")
    print("\nAUROC using Kraken2 k-mer hit fraction as the novelty score:")
    for tier in ["near", "far"]:
        k = np.load(tmp / f"{tier}_km.npy")
        y = np.r_[np.zeros(len(idk)), np.ones(len(k))]
        print(f"  id_test vs {tier:5s}: {roc_auc_score(y, -np.r_[idk, k]):.3f}")
    pd.DataFrame(rows).T.to_csv(f"results/kraken2_{args.v}.csv")


if __name__ == "__main__":
    main()
