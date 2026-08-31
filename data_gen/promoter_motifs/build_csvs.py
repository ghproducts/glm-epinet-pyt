#!/usr/bin/env python3
"""Join the split manifest onto sequence text to build model-ready CSVs.

Step 5 of the promoter-motif pipeline (see README.md), the local analogue of
``data_gen/pbsim/reads_to_csv.py``. Writes one CSV per split into
``--outdir`` (gitignored, like ``data_gen/pbsim/csv_data/`` -- regenerate
rather than commit) with the columns
``nn_proj.common.datasets.load_local_dataset`` expects:

    label               0/1 promoter presence (renamed to `labels` at load time)
    sequence            251bp nucleotide sequence
    sequence_id         carried through as metadata
    motif_combo         carried through as metadata
    num_selected_motifs carried through as metadata

The metadata columns are what let per-motif-combo stratified analysis
(Reviewer 2 point 3, Reviewer 1 point 3 in .docs/REVISION_PLAN.md) run
directly off prediction CSVs, once MODEL_CODE_FIXES.md item 2.3's metadata
passthrough in ``predict()`` lands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUT_COLS = ["label", "sequence", "sequence_id", "motif_combo", "num_selected_motifs"]
SPLITS = ["train", "test_ID", "test_matched_ID", "test_OOD"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--manifest", type=Path, default=here / "splits" / "motif_splits.csv")
    ap.add_argument("--annotated", type=Path, default=here / "promoters_annotated.csv",
                     help="Annotated corpus with sequence text (output of annotate_motifs.py)")
    ap.add_argument("--outdir", type=Path, default=here / "csv_data")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    corpus = pd.read_csv(args.annotated, usecols=["sequence_id", "sequence"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = manifest[manifest["split"] == split].merge(corpus, on="sequence_id", how="left")

        missing = rows["sequence"].isna().sum()
        if missing:
            raise SystemExit(
                f"{split}: {missing} sequence_id(s) in the manifest have no match in "
                f"{args.annotated}. Was it built from the same corpus as the manifest?"
            )

        out_path = args.outdir / f"{split}.csv"
        rows[OUT_COLS].to_csv(out_path, index=False)
        print(f"{split:<18} {len(rows):>6} rows -> {out_path}")


if __name__ == "__main__":
    main()
