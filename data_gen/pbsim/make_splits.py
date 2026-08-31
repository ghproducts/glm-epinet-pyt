#!/usr/bin/env python3
"""Assign reference species to the taxonomic train/test splits.

This is step 1 of the simulated long-read pipeline and the operational
definition of the taxonomic shift categories. Every species with a downloaded
reference genome is assigned to exactly one split:

    train             bacterial species used for fine-tuning
    id_novel_genus    bacterial species from genera held out of training,
                      within families that ARE in training  -> Near-ID
    ood_novel_family  bacterial species from families held out entirely,
                      -> Near-OOD
    ood_nonbacterial  every non-bacterial species (superkingdom != 2)
                      -> OOD

Holdout is at the level of the *taxon*, not the sequence, so no genome
contributes reads to more than one split.

The order matters: families are held out first, then genera are held out from
the families that remain. A species can therefore only be `id_novel_genus` if
its family survived the family holdout, which is what makes that split
"same label space, unseen organisms".

Outputs, written to ``--outdir``:
    species_splits.csv        full lineage table with a `split` column
    <split>_species.txt       one species taxid per line, per split

Both are committed to the repository, so the exact partition behind the
published results is reproducible without rerunning this script or
re-downloading RefSeq.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ["species", "genus", "family", "order", "class", "phylum",
                 "kingdom", "superkingdom"]
BACTERIA_SUPERKINGDOM = 2  # NCBI taxonomy id for Bacteria


def load_lineage(path: Path, sequences_dir: Path | None) -> pd.DataFrame:
    """Load the lineage table, optionally restricting to downloaded genomes."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")

    df["species"] = df["species"].astype("Int64")
    df = df.dropna(subset=["species"]).drop_duplicates(subset=["species"])

    if sequences_dir is not None:
        if not sequences_dir.is_dir():
            raise FileNotFoundError(f"Sequences directory not found: {sequences_dir}")
        available = {p.name for p in sequences_dir.iterdir() if p.is_dir()}
        before = len(df)
        df = df[df["species"].astype(int).astype(str).isin(available)]
        print(f"Species in lineage table      : {before}")
        print(f"Species with reference genomes: {len(df)}")

    return df.reset_index(drop=True)


def assign_splits(
    df: pd.DataFrame,
    seed: int,
    frac_heldout_families: float,
    frac_heldout_genera: float,
    min_genera_per_family: int,
) -> pd.DataFrame:
    """Assign each species to a split. Returns a copy with a `split` column."""
    rng = np.random.default_rng(seed)

    bact = df[df["superkingdom"] == BACTERIA_SUPERKINGDOM].copy()
    non_bact = df[df["superkingdom"] != BACTERIA_SUPERKINGDOM].copy()
    print(f"\nBacterial species     : {len(bact)}")
    print(f"Non-bacterial species : {len(non_bact)}  -> ood_nonbacterial")

    bact["split"] = "train"
    non_bact["split"] = "ood_nonbacterial"

    # --- Hold out whole families (Near-OOD) ------------------------------
    families = bact["family"].dropna().unique()
    if families.size == 0:
        raise ValueError("No family annotations among bacterial species.")

    n_hold = max(1, int(frac_heldout_families * families.size))
    n_hold = min(n_hold, families.size - 1)  # never hold out everything
    heldout_families = rng.choice(families, size=n_hold, replace=False)
    bact.loc[bact["family"].isin(heldout_families), "split"] = "ood_novel_family"

    print(f"\nBacterial families        : {families.size}")
    print(f"Held-out families         : {len(heldout_families)}  -> ood_novel_family")

    # --- Hold out genera within the remaining families (Near-ID) ---------
    # Only families that survived the family holdout are eligible, and only
    # those with enough genera that removing some still leaves the family
    # represented in training.
    train_bact = bact[bact["split"] == "train"]
    genera_per_family = train_bact.groupby("family")["genus"].nunique()
    eligible = genera_per_family[genera_per_family >= min_genera_per_family].index

    n_heldout_genera = 0
    for family in eligible:
        family_mask = (bact["family"] == family) & (bact["split"] == "train")
        genera = bact.loc[family_mask, "genus"].dropna().unique()
        if genera.size < min_genera_per_family:
            continue
        n = max(1, int(frac_heldout_genera * genera.size))
        n = min(n, genera.size - 1)  # keep the family present in training
        heldout = rng.choice(genera, size=n, replace=False)
        bact.loc[family_mask & bact["genus"].isin(heldout), "split"] = "id_novel_genus"
        n_heldout_genera += len(heldout)

    print(f"Families eligible for genus holdout: {len(eligible)}")
    print(f"Held-out genera                    : {n_heldout_genera}  -> id_novel_genus")

    out = pd.concat([bact, non_bact], ignore_index=True)
    out["split"] = out["split"].fillna("train")
    return out


def verify(df: pd.DataFrame) -> None:
    """Check the invariants the shift categories depend on."""
    train = df[df.split == "train"]
    problems = []

    train_families = set(train["family"].dropna())
    train_genera = set(train["genus"].dropna())

    novel_genus = df[df.split == "id_novel_genus"]
    leaked = set(novel_genus["genus"].dropna()) & train_genera
    if leaked:
        problems.append(f"id_novel_genus shares {len(leaked)} genus/genera with train")
    orphan = set(novel_genus["family"].dropna()) - train_families
    if orphan:
        problems.append(
            f"id_novel_genus has {len(orphan)} family/families absent from train; "
            "those species are not Near-ID"
        )

    novel_family = df[df.split == "ood_novel_family"]
    leaked = set(novel_family["family"].dropna()) & train_families
    if leaked:
        problems.append(f"ood_novel_family shares {len(leaked)} family/families with train")

    overlap = df.groupby("species")["split"].nunique()
    if (overlap > 1).any():
        problems.append(f"{int((overlap > 1).sum())} species appear in more than one split")

    print("\nSplit verification:")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        raise SystemExit("Split assignment violates its own definition; not writing output.")
    print("  ok  id_novel_genus: no genus overlap with train, all families present in train")
    print("  ok  ood_novel_family: no family overlap with train")
    print("  ok  every species belongs to exactly one split")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--lineage", type=Path, default=here / "full_basic_lineage.csv",
                    help="Lineage table with species/genus/family/.../superkingdom columns")
    ap.add_argument("--sequences", type=Path, default=None,
                    help="Directory of per-species reference genomes; when given, only "
                         "species with a downloaded genome are assigned")
    ap.add_argument("--outdir", type=Path, default=here / "splits")
    ap.add_argument("--seed", type=int, default=123,
                    help="RNG seed for the holdout draws (published splits used 123)")
    ap.add_argument("--frac-heldout-families", type=float, default=0.10)
    ap.add_argument("--frac-heldout-genera", type=float, default=0.25)
    ap.add_argument("--min-genera-per-family", type=int, default=3)
    args = ap.parse_args()

    print(f"Lineage table : {args.lineage}")
    df = load_lineage(args.lineage, args.sequences)

    splits = assign_splits(df, args.seed, args.frac_heldout_families,
                           args.frac_heldout_genera, args.min_genera_per_family)
    verify(splits)

    print("\nSpecies per split:")
    for name, count in splits["split"].value_counts().items():
        print(f"  {name:<20} {count}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_out = args.outdir / "species_splits.csv"
    splits.to_csv(csv_out, index=False)
    print(f"\nWrote {csv_out}")

    for name, subset in splits.groupby("split"):
        txt = args.outdir / f"{name}_species.txt"
        txt.write_text("\n".join(subset["species"].astype(int).astype(str)) + "\n")
        print(f"Wrote {txt} ({len(subset)} species)")


if __name__ == "__main__":
    main()
