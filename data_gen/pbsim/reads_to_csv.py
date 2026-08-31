#!/usr/bin/env python3
"""Convert simulated FASTQ reads into the CSV format the training code expects.

Step 3 of the simulated long-read pipeline, run after ``make_splits.py`` and
``generate_pbsim_reads.sh``. Reads the per-species FASTQ output and writes one
CSV per split with the columns ``nn_proj.common.datasets.load_local_dataset``
requires:

    label      family taxid of the source species, or -1 when unmapped
    sequence   the simulated read, uppercased with non-ACGT bases as N
    taxid      species taxid of the source genome
    split      which split the read belongs to

``label`` is family-level. Training and evaluation at coarser ranks remap it
through the lineage table via the ``--taxa_rank`` argument to the training
scripts, so this file does not need regenerating per rank.

Reads are subsampled per species with reservoir sampling so that abundant
genomes do not dominate. Sampling is seeded and the counts are recorded in the
run summary.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

SPLITS = ["train", "id_novel_genus", "ood_novel_family", "ood_nonbacterial"]

# Reads kept per species. Training gets more because it must cover the label
# space; the evaluation splits only need enough reads for a stable estimate.
DEFAULT_READS_PER_SPECIES = {
    "train": 100,
    "id_novel_genus": 10,
    "ood_novel_family": 10,
    "ood_nonbacterial": 10,
}

NON_ACGT = re.compile(r"[^ATCG]")


def load_species_to_family(splits_csv: Path) -> Dict[str, int]:
    """Map species taxid (as string) to family taxid, -1 when absent."""
    df = pd.read_csv(splits_csv)
    return dict(zip(
        df["species"].astype("Int64").astype(str),
        df["family"].fillna(-1).astype(int),
    ))


def iter_fastq(path: Path) -> Iterator[str]:
    """Yield sequences from a plain or gzipped FASTQ file."""
    opener, mode = (gzip.open, "rt") if path.suffix == ".gz" else (open, "r")
    with opener(path, mode) as f:
        while True:
            header = f.readline()
            if not header.strip():
                return
            seq = f.readline().rstrip()
            f.readline()          # '+' separator
            if not f.readline():  # quality line; truncated record if absent
                return
            yield seq


def clean(seq: str) -> str:
    """Uppercase and replace every non-ACGT character with N."""
    return NON_ACGT.sub("N", seq.upper())


def sample_reads(fastqs: List[Path], target: Optional[int], rng: random.Random) -> List[str]:
    """Up to ``target`` reads for one species, or all of them when target is None.

    Uses reservoir sampling so a species with millions of reads costs the same
    memory as one with a hundred, and every read has equal probability of
    selection without knowing the total up front.
    """
    if target is None:
        return [clean(s) for p in fastqs for s in iter_fastq(p)]

    reservoir: List[str] = []
    seen = 0
    for path in fastqs:
        for raw in iter_fastq(path):
            seen += 1
            if len(reservoir) < target:
                reservoir.append(clean(raw))
            else:
                j = rng.randrange(seen)
                if j < target:
                    reservoir[j] = clean(raw)
    return reservoir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--splits-csv", type=Path, default=here / "splits" / "species_splits.csv")
    ap.add_argument("--raw-dir", type=Path, default=here / "pbsim_raw",
                    help="PBSim output: <raw_dir>/<split>/<species_taxid>/*.fq")
    ap.add_argument("--outdir", type=Path, default=here / "csv_data")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--reads-per-species", type=int, default=None,
                    help="Override the per-split defaults with a single value "
                         "(omit for train=100, evaluation splits=10)")
    ap.add_argument("--no-combined", action="store_true",
                    help="Skip the combined reads_all_splits.csv (it is large)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    species_to_family = load_species_to_family(args.splits_csv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    combined = None
    if not args.no_combined:
        combined_path = args.outdir / "reads_all_splits.csv"
        combined_file = combined_path.open("w", newline="")
        combined = csv.writer(combined_file)
        combined.writerow(["label", "sequence", "taxid", "split"])

    summary = []
    for split in args.splits:
        split_dir = args.raw_dir / split
        if not split_dir.is_dir():
            print(f"[warn] no reads for split {split!r} at {split_dir}, skipping")
            continue

        target = (args.reads_per_species if args.reads_per_species is not None
                  else DEFAULT_READS_PER_SPECIES.get(split))
        out_path = args.outdir / f"reads_{split}.csv"
        print(f"\n{split}: target {target} reads/species -> {out_path}")

        n_species = n_reads = n_unmapped = 0
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "sequence", "taxid", "split"])

            for species_dir in sorted(split_dir.iterdir()):
                if not species_dir.is_dir():
                    continue
                species = species_dir.name
                fastqs = sorted(species_dir.glob("*.fq*"))
                if not fastqs:
                    print(f"  [warn] no FASTQ files for species {species}")
                    continue

                label = species_to_family.get(species, -1)
                if label == -1:
                    n_unmapped += 1

                seqs = sample_reads(fastqs, target, rng)
                for seq in seqs:
                    writer.writerow([label, seq, species, split])
                    if combined:
                        combined.writerow([label, seq, species, split])

                n_species += 1
                n_reads += len(seqs)

        print(f"  {n_species} species, {n_reads} reads")
        if n_unmapped:
            # These become label -1 and would be trained as a real class. The
            # training scripts drop them; see nn_proj/common/datasets.py.
            print(f"  [warn] {n_unmapped} species have no family in the lineage table "
                  f"(label -1); these are dropped at load time")
        summary.append({"split": split, "species": n_species, "reads": n_reads,
                        "reads_per_species": target, "unmapped_species": n_unmapped})

    if combined:
        combined_file.close()
        print(f"\nWrote {combined_path}")

    summary_df = pd.DataFrame(summary)
    summary_path = args.outdir / "generation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{summary_df.to_string(index=False)}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
