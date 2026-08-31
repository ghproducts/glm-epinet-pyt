#!/usr/bin/env python3
"""Join FIMO motif hits onto the promoter corpus.

Step 3 of the promoter-motif pipeline (see README.md), run after FIMO scans
``promoters_all.fa`` against the 6 selected motifs. Adds one 0/1 presence
column per motif, ``num_selected_motifs`` (their row sum), and
``motif_combo`` (the sorted ``+``-joined names of the motifs present, or
``"none"``) — the columns ``make_splits.py`` assigns splits on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from select_motifs import MOTIF_NAME_MAP

MOTIF_COLS = list(MOTIF_NAME_MAP.values())


def annotate(promoters: pd.DataFrame, fimo: pd.DataFrame) -> pd.DataFrame:
    fimo = fimo.copy()
    fimo["sequence_id"] = fimo["sequence_name"].str.split("|").str[0]
    fimo["motif_short"] = fimo["motif_id"].map(MOTIF_NAME_MAP)

    presence = (
        fimo.dropna(subset=["motif_short"])
            .assign(present=1)
            .drop_duplicates(["sequence_id", "motif_short"])
            .pivot(index="sequence_id", columns="motif_short", values="present")
            .fillna(0).astype(int).reset_index()
    )

    annot = promoters.merge(presence, on="sequence_id", how="left")
    for col in MOTIF_COLS:
        if col not in annot.columns:
            annot[col] = 0
        annot[col] = annot[col].fillna(0).astype(int)

    annot["num_selected_motifs"] = annot[MOTIF_COLS].sum(axis=1)
    annot["motif_combo"] = annot.apply(
        lambda row: "+".join(sorted(m for m in MOTIF_COLS if row[m] == 1)) or "none", axis=1,
    )
    return annot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--promoters", type=Path, default=here / "promoters_all.csv")
    ap.add_argument("--fimo", type=Path, default=here / "fimo_out" / "fimo.tsv",
                     help="FIMO's fimo.tsv output")
    ap.add_argument("--out", type=Path, default=here / "promoters_annotated.csv")
    args = ap.parse_args()

    promoters = pd.read_csv(args.promoters)
    fimo = pd.read_csv(args.fimo, sep="\t", comment="#")

    annot = annotate(promoters, fimo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    annot.to_csv(args.out, index=False)

    print(f"Annotated {len(annot)} sequences -> {args.out}")
    print("\nmotif_combo counts (top 15):")
    print(annot["motif_combo"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
