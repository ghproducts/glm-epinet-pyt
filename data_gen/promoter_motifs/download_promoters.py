#!/usr/bin/env python3
"""Download the promoter corpus and export FASTA files for motif discovery.

Step 1 of the promoter-motif pipeline (see README.md). Downloads
``human_nontata_promoters`` (version 0) via the ``genomic-benchmarks``
package and writes:

    promoters_all.csv                all sequences, flat CSV
    promoters_positive_train.fa      positive-class training sequences
    promoters_negative_train.fa      negative-class training sequences
    promoters_all.fa                 every sequence, for the FIMO scan

The two single-class FASTA files are AME's case/control inputs (step 2 in
README.md); ``promoters_all.fa`` is FIMO's scan target (step 3).

Requires network access and ``pip install genomic-benchmarks``; not run as
part of this repository's test suite for the same reason
``data_gen/pbsim``'s genome download isn't.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATASET_NAME = "human_nontata_promoters"
DATASET_VERSION = 0


def download(out_dir: Path) -> pd.DataFrame:
    """Download the dataset and flatten it into one dataframe.

    ``source_split``/``label`` follow the dataset's own train/test/
    positive/negative directory layout; that split is unrelated to this
    module's train/test_ID/test_matched_ID/test_OOD split (see README.md).
    """
    from genomic_benchmarks.loc2seq import download_dataset

    dataset_path = Path(download_dataset(
        DATASET_NAME, version=DATASET_VERSION, dest_path=out_dir / "genomic_benchmarks",
    ))

    records = []
    for source_split in ["train", "test"]:
        for label_name, label_value in [("negative", 0), ("positive", 1)]:
            split_dir = dataset_path / source_split / label_name
            for fp in sorted(split_dir.iterdir()):
                if not fp.is_file():
                    continue
                seq = "".join(fp.read_text().strip().upper().split())
                records.append({
                    "sequence_id": f"{source_split}_{label_name}_{fp.stem}",
                    "sequence": seq,
                    "label": label_value,
                    "label_name": label_name,
                    "source_split": source_split,
                })

    return pd.DataFrame(records)


def write_fasta(df: pd.DataFrame, out_path: Path) -> None:
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['sequence_id']}|label={row['label']}|split={row['source_split']}\n")
            f.write(f"{row['sequence']}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--out-dir", type=Path, default=here,
                     help="Directory to write promoters_all.csv and the FASTA files into")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = download(args.out_dir)
    csv_path = args.out_dir / "promoters_all.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} sequences)")

    train = df[df["source_split"] == "train"]
    write_fasta(train[train["label"] == 1], args.out_dir / "promoters_positive_train.fa")
    write_fasta(train[train["label"] == 0], args.out_dir / "promoters_negative_train.fa")
    write_fasta(df, args.out_dir / "promoters_all.fa")
    print(f"Wrote FASTA files to {args.out_dir}")


if __name__ == "__main__":
    main()
