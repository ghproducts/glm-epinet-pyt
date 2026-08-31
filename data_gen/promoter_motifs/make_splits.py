#!/usr/bin/env python3
"""Assign annotated promoters to the train/test_ID/test_matched_ID/test_OOD splits.

Step 4 of the promoter-motif pipeline (see README.md) and the operational
definition of the shift: every eligible sequence is assigned to exactly one
of

    train              80% of the non-held-out eligible pool (stratified
                        by label)
    test_ID            the other 20%
    test_matched_ID     resampled from test_ID to match test_OOD's per-label,
                        per-motif-count distribution -- a matched control,
                        not a shift condition
    test_OOD           every sequence carrying one of HELDOUT_COMBOS, held
                        out of training entirely

Ported from the DSCI-691 course project's ``data_pipeline.ipynb`` (cell 12),
generalized into a committed, argument-driven script with the verification
pass pbsim's ``make_splits.py`` runs for its own splits.

The 80/20 draw and the matched-control resampling are both seeded, but a
fresh run of this script against a newly downloaded corpus will not
reproduce the *committed* ``splits/motif_splits.csv`` bit-for-bit even with
the same seed, because the stratified shuffle is sensitive to input row
order and a fresh download will not enumerate sequences in the same order
the original one did. See README.md, "Regenerating vs. using the release".

Use ``--check`` to instead re-run verification against an already-written
manifest without recomputing anything -- this is how the committed manifest
here was actually validated, since it was derived from the DSCI-691 release
rather than from a fresh run of this script (see README.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Excluded from training entirely; become test_OOD.
HELDOUT_COMBOS = {"EGR1+SP2", "SP2+TFDP1", "SP2+ZNF131", "ZBED4+ZNF131"}

# Everything else is dropped for insufficient support in one label or the
# other. Includes the four held-out combos above (they still need to be
# eligible to end up in test_OOD) plus every combo with enough examples in
# both promoter and non-promoter sequences.
ELIGIBLE_COMBOS = HELDOUT_COMBOS | {
    "none", "SP2", "EGR1", "ZBED4", "ZNF131", "ELK3", "TFDP1",
    "SP2+ZBED4", "SP2+ZBED4+ZNF131", "EGR1+SP2+ZBED4+ZNF131",
}

MANIFEST_COLS = ["sequence_id", "label", "label_name", "motif_combo",
                  "num_selected_motifs", "SP2", "ZBED4", "EGR1", "ELK3",
                  "TFDP1", "ZNF131", "split"]


def assign_splits(annot: pd.DataFrame, seed: int, test_size: float) -> pd.DataFrame:
    """Port of data_pipeline.ipynb cell 12. Returns the manifest (no sequence text)."""
    df = annot[annot["motif_combo"].isin(ELIGIBLE_COMBOS)].copy()

    heldout_test = df[df["motif_combo"].isin(HELDOUT_COMBOS)].copy()
    nonheldout = df[~df["motif_combo"].isin(HELDOUT_COMBOS)].copy()

    train_idx, iid_idx = train_test_split(
        nonheldout.index, test_size=test_size, random_state=seed,
        stratify=nonheldout["label"],
    )
    train = nonheldout.loc[train_idx].copy()
    iid_test = nonheldout.loc[iid_idx].copy()

    matched_parts = []
    for (label, n_motifs), group in heldout_test.groupby(["label", "num_selected_motifs"]):
        pool = iid_test[(iid_test["label"] == label) &
                         (iid_test["num_selected_motifs"] == n_motifs)]
        matched_parts.append(
            pool.sample(n=len(group), replace=len(pool) < len(group), random_state=seed)
        )
    matched_test = pd.concat(matched_parts).sample(frac=1, random_state=seed) if matched_parts else pd.DataFrame(columns=iid_test.columns)

    train["split"] = "train"
    iid_test["split"] = "test_ID"
    matched_test["split"] = "test_matched_ID"
    heldout_test["split"] = "test_OOD"

    manifest = pd.concat([train, iid_test, matched_test, heldout_test], ignore_index=True)
    return manifest[MANIFEST_COLS]


def verify(manifest: pd.DataFrame) -> None:
    """Check the invariants the split depends on. Fails closed on real defects,
    warns (does not fail) on the known matched-ID resampling limitation."""
    problems = []

    train = manifest[manifest["split"] == "train"]
    ood = manifest[manifest["split"] == "test_OOD"]
    id_test = manifest[manifest["split"] == "test_ID"]
    matched = manifest[manifest["split"] == "test_matched_ID"]

    leaked = set(train["motif_combo"]) & HELDOUT_COMBOS
    if leaked:
        problems.append(f"train contains held-out combo(s) {sorted(leaked)}")

    # No sequence should span two splits, except test_ID <-> test_matched_ID,
    # which overlap by construction (the matched control is drawn from the ID
    # pool, not disjoint from it).
    by_split = {name: set(sub["sequence_id"]) for name, sub in manifest.groupby("split")}
    names = list(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = by_split[a] & by_split[b]
            if overlap and {a, b} != {"test_ID", "test_matched_ID"}:
                problems.append(f"{len(overlap)} sequence(s) appear in both {a!r} and {b!r}")

    for label in sorted(ood["label"].unique()):
        ood_counts = ood[ood["label"] == label]["num_selected_motifs"].value_counts()
        matched_counts = matched[matched["label"] == label]["num_selected_motifs"].value_counts()
        if not ood_counts.sort_index().equals(matched_counts.reindex(ood_counts.index).fillna(-1).sort_index()):
            mismatch = ood_counts.sort_index().compare(
                matched_counts.reindex(ood_counts.index).sort_index()
            )
            problems.append(f"label {label}: test_matched_ID motif-count distribution "
                             f"does not match test_OOD:\n{mismatch}")

    print("Split verification:")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        raise SystemExit("Split assignment violates its own definition; not writing output.")
    print("  ok  train: no held-out combination present")
    print("  ok  no cross-split sequence overlap other than test_ID <-> test_matched_ID")
    print("  ok  test_matched_ID motif-count distribution matches test_OOD, per label")

    n_unique = matched["sequence_id"].nunique()
    n_total = len(matched)
    if n_unique < n_total:
        max_repeat = matched["sequence_id"].value_counts().max()
        print(f"  WARN  test_matched_ID is resampled with replacement: "
              f"{n_unique} unique sequence(s) fill {n_total} rows "
              f"(max {max_repeat} repeats of one sequence). Treat this split as "
              f"~{n_unique} independent observations, not {n_total} -- see README.md.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--annotated", type=Path, default=here / "promoters_annotated.csv")
    ap.add_argument("--outdir", type=Path, default=here / "splits")
    ap.add_argument("--seed", type=int, default=42,
                     help="RNG seed for the 80/20 draw and matched-ID resampling "
                          "(published split used 42)")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--check", action="store_true",
                     help="Re-verify --outdir/motif_splits.csv without recomputing anything")
    args = ap.parse_args()

    manifest_path = args.outdir / "motif_splits.csv"

    if args.check:
        print(f"Checking {manifest_path} (no regeneration)")
        manifest = pd.read_csv(manifest_path)
        verify(manifest)
        return

    print(f"Annotated corpus: {args.annotated}")
    annot = pd.read_csv(args.annotated)

    manifest = assign_splits(annot, args.seed, args.test_size)
    verify(manifest)

    print("\nRows per split:")
    for name, count in manifest["split"].value_counts().items():
        print(f"  {name:<18} {count}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
