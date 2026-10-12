#!/usr/bin/env python3
"""Alignment-based novelty baseline (MMseqs2) for the leave-phyla-out setup.

NOTE on reference construction: an earlier version built the reference from the
same 150bp windows the neural models train on and got zero alignments for every
query, ID and OOD alike. That is arithmetic rather than biology -- two short
reads from different organisms only align if they happen to cover the same
locus, and a sparse sample of 150bp windows covers almost none of a ~5Mb genome.
The reference here is therefore the FULL-LENGTH (~10kb) training reads for the
model's in-label phyla, which is both the fair analogue of what the network saw
and a reasonable stand-in for a reference genome database.

Best-hit sequence identity against a reference built from exactly the phyla a
given model was trained on. This is the field-standard way to flag novel
sequence, and it is the comparison that says whether the neural uncertainty
scores are competitive at all: same reference, same query reads, same ID/OOD
ground truth, same AUROC.

BLAST+ is not installed in this environment; MMseqs2 is, and is the faster
equivalent for this purpose.
"""
from __future__ import annotations
import argparse, subprocess, shutil, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

MMSEQS = shutil.which("mmseqs") or "/home/glh52/opt/mmseqs/bin/mmseqs"


def write_fasta(seqs, path, prefix):
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{prefix}{i}\n{s}\n")


def best_hits(ref_seqs, qry_seqs, tmp: Path, sens=5.7, threads=16):
    """Return best-hit identity per query (0.0 where nothing aligned)."""
    r, q = tmp / "ref.fa", tmp / "qry.fa"
    write_fasta(ref_seqs, r, "r"); write_fasta(qry_seqs, q, "q")
    out = tmp / "hits.m8"
    subprocess.run([MMSEQS, "easy-search", str(q), str(r), str(out), str(tmp / "tmp"),
                    "--search-type", "3", "-s", str(sens), "--threads", str(threads),
                    "--max-seqs", "10", "-v", "1"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ident = np.zeros(len(qry_seqs))
    if out.exists() and out.stat().st_size:
        h = pd.read_csv(out, sep="\t", header=None,
                        names=["q","t","pident","aln","mm","go","qs","qe","ts","te","e","bits"])
        h["qi"] = h.q.str.slice(1).astype(int)
        bb = h.groupby("qi").pident.max()
        ident[bb.index.values] = bb.values
    return ident


TRAIN_CSV = "/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
LINEAGE = "/scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv"
_CACHE = {}


def full_length_reference(phyla, per_phylum):
    """Full-length training reads for the given phyla (cached across subsets)."""
    if not _CACHE:
        lin = pd.read_csv(LINEAGE)
        sp2ph = dict(zip(lin.species.astype(str), lin.phylum.astype(str)))
        buckets = {}
        for ch in pd.read_csv(TRAIN_CSV, usecols=["sequence", "taxid"], chunksize=200_000):
            ch["ph"] = ch.taxid.astype(str).map(sp2ph)
            for p, g in ch.groupby("ph"):
                buckets.setdefault(p, []).append(g.sequence)
        for p, v in buckets.items():
            _CACHE[p] = pd.concat(v, ignore_index=True)
    out = []
    rng = np.random.default_rng(1)
    for p in phyla:
        s = _CACHE.get(p)
        if s is None or not len(s):
            continue
        idx = rng.choice(len(s), size=min(len(s), per_phylum), replace=False)
        out.extend(s.iloc[idx].tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--ref-per-phylum", type=int, default=20000,
                    help="full-length reference reads per in-label phylum")
    ap.add_argument("--subsets", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--out", type=Path, default=here.parent.parent / "results" / "leave_phyla_out" / "mmseqs_baseline.csv")
    args = ap.parse_args()

    design = pd.read_csv(here / "subsets" / "design.csv")
    ev = pd.read_csv(here / "eval_pbsim_150bp.csv")
    ev["true_phylum"] = ev.true_phylum.astype(str)
    rows = []
    for i in args.subsets:
        phyla = design.loc[design.subset == i, "phyla"].iloc[0].split("|")
        ref_seqs = full_length_reference(phyla, args.ref_per_phylum)
        with tempfile.TemporaryDirectory() as td:
            ident = best_hits(ref_seqs, ev.sequence.tolist(), Path(td))
        ref = ref_seqs
        is_id = ev.true_phylum.isin(phyla).values
        auroc = roc_auc_score((~is_id).astype(int), -ident)   # low identity => novel
        rows.append(dict(subset=i, n_ref=len(ref_seqs), n_id=int(is_id.sum()),
                         n_ood=int((~is_id).sum()),
                         ident_id=ident[is_id].mean(), ident_ood=ident[~is_id].mean(),
                         hit_rate_id=(ident[is_id] > 0).mean(),
                         hit_rate_ood=(ident[~is_id] > 0).mean(), auroc=auroc))
        print(f"subset_{i}: ident ID={ident[is_id].mean():.1f} OOD={ident[~is_id].mean():.1f} "
              f"| hit-rate ID={(ident[is_id]>0).mean():.3f} OOD={(ident[~is_id]>0).mean():.3f} "
              f"| AUROC={auroc:.3f}")
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nmean AUROC = {df.auroc.mean():.3f}  ->  {args.out}")


if __name__ == "__main__":
    main()
