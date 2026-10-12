#!/usr/bin/env python3
"""v2 split: fixes the two confounds `.docs/PROMOTER_MOTIFS_RESULTS.md` found in v1.

v1 (``make_splits.py``) held out four motif combinations, all of them
2-motif pairs. Two things followed directly from that, verified against real
predictions: `test_OOD` came out class-imbalanced relative to training (61%
positive vs. training's 61% *negative*, because every motif-bearing
combination in this corpus skews positive -- "no motif" is the class-0-heavy
bucket), and `test_OOD` sat entirely in one motif-count stratum (2), which
happens to be an easy, atypical corner of the difficulty range (see the
`num_selected_motifs`-stratified error table in `.docs/REVISION_PLAN.md`).
Together they made plain `test_ID` vs. `test_OOD` comparisons read backwards
in some backbones -- OOD looking *easier* than ID.

v2 keeps the same "hold out specific motif combinations, evaluate whether
the model generalizes to them" story, and only changes two things:

1. Adds the standalone single-motif combination ``TFDP1`` to the held-out
   set (on top of the original four 2-motif pairs), so `test_OOD` now spans
   motif-count strata {1, 2} instead of only {2}. It doesn't reach 3/4 -- the
   annotated pool has exactly one eligible combination at each of those
   counts, and holding either out would erase that count from training
   entirely; not attempted here.
2. Downsamples the held-out pool's positive-class rows (stratified per
   combination, keeping every negative) so `test_OOD`'s label balance
   matches the non-held-out (train + test_ID) pool's, instead of whatever
   ratio the chosen combinations happen to carry.

Everything else -- the 80/20 train/test_ID draw, the `test_matched_ID`
per-(label, num_selected_motifs) resampling against `test_OOD`, the
verification pass -- follows v1's `make_splits.py` exactly, reusing the same
annotated corpus (no FIMO/AME rerun needed; MEME suite isn't available in
this environment, so this works only within the pool v1 already annotated).

Outputs, written to ``--outdir`` (default ``splits_v2/``):
    motif_splits.csv   same schema as v1's manifest, no sequence text
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Same four 2-motif pairs as v1, plus the standalone single-motif TFDP1 --
# spans motif-count strata {1, 2} instead of only {2}.
HELDOUT_COMBOS = {"TFDP1", "SP2+TFDP1", "EGR1+SP2", "SP2+ZNF131", "ZBED4+ZNF131"}

# Same eligibility rule as v1 (enough support in both classes), minus
# standalone TFDP1, which moved into HELDOUT_COMBOS above.
ELIGIBLE_COMBOS = HELDOUT_COMBOS | {
    "none", "SP2", "EGR1", "ZBED4", "ZNF131", "ELK3",
    "SP2+ZBED4", "SP2+ZBED4+ZNF131", "EGR1+SP2+ZBED4+ZNF131",
}

MANIFEST_COLS = ["sequence_id", "label", "label_name", "motif_combo",
                  "num_selected_motifs", "SP2", "ZBED4", "EGR1", "ELK3",
                  "TFDP1", "ZNF131", "split"]


def assign_splits(annot: pd.DataFrame, seed: int, test_size: float) -> pd.DataFrame:
    df = annot[annot["motif_combo"].isin(ELIGIBLE_COMBOS)].copy()

    heldout_pool = df[df["motif_combo"].isin(HELDOUT_COMBOS)].copy()
    nonheld = df[~df["motif_combo"].isin(HELDOUT_COMBOS)].copy()

    train_idx, iid_idx = train_test_split(
        nonheld.index, test_size=test_size, random_state=seed,
        stratify=nonheld["label"],
    )
    train = nonheld.loc[train_idx].copy()
    iid_test = nonheld.loc[iid_idx].copy()

    # --- balance-corrected test_OOD -----------------------------------
    # Keep every negative in the held-out pool; downsample positives,
    # stratified per combination so their relative mix is preserved, to hit
    # nonheld's own label balance exactly.
    target_pos_frac = nonheld["label"].mean()
    neg = heldout_pool[heldout_pool["label"] == 0]
    pos = heldout_pool[heldout_pool["label"] == 1]
    pos_needed = int(round(len(neg) * target_pos_frac / (1 - target_pos_frac)))
    frac = pos_needed / len(pos) if len(pos) else 0.0

    pos_parts = []
    for _, g in pos.groupby("motif_combo"):
        n_take = min(len(g), max(1, int(round(len(g) * frac))))
        pos_parts.append(g.sample(n=n_take, random_state=seed))
    heldout_test = pd.concat([neg, *pos_parts]).sample(frac=1, random_state=seed)

    # --- matched-ID control, same technique as v1 ----------------------
    matched_parts = []
    for (label, n_motifs), group in heldout_test.groupby(["label", "num_selected_motifs"]):
        pool = iid_test[(iid_test["label"] == label) &
                         (iid_test["num_selected_motifs"] == n_motifs)]
        matched_parts.append(
            pool.sample(n=len(group), replace=len(pool) < len(group), random_state=seed)
        )
    matched_test = (pd.concat(matched_parts).sample(frac=1, random_state=seed)
                     if matched_parts else pd.DataFrame(columns=iid_test.columns))

    train["split"] = "train"
    iid_test["split"] = "test_ID"
    matched_test["split"] = "test_matched_ID"
    heldout_test["split"] = "test_OOD"

    manifest = pd.concat([train, iid_test, matched_test, heldout_test], ignore_index=True)
    return manifest[MANIFEST_COLS]


def verify(manifest: pd.DataFrame, balance_tolerance: float = 0.02) -> None:
    problems = []

    train = manifest[manifest["split"] == "train"]
    ood = manifest[manifest["split"] == "test_OOD"]
    id_test = manifest[manifest["split"] == "test_ID"]
    matched = manifest[manifest["split"] == "test_matched_ID"]
    nonheld_ref = pd.concat([train, id_test])

    leaked = set(train["motif_combo"]) & HELDOUT_COMBOS
    if leaked:
        problems.append(f"train contains held-out combo(s) {sorted(leaked)}")

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

    # New check, not in v1: test_OOD's label balance vs. the non-held-out pool's.
    ref_pos = nonheld_ref["label"].mean()
    ood_pos = ood["label"].mean()
    if abs(ref_pos - ood_pos) > balance_tolerance:
        problems.append(f"test_OOD pos_frac {ood_pos:.4f} vs. train+test_ID pos_frac "
                         f"{ref_pos:.4f} differs by more than {balance_tolerance}")

    # New check: test_OOD spans more than one motif-count stratum.
    n_strata = ood["num_selected_motifs"].nunique()
    if n_strata < 2:
        problems.append(f"test_OOD sits in only {n_strata} motif-count stratum; "
                         "the whole point of v2 is spanning more than one")

    print("Split verification (v2):")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        raise SystemExit("Split assignment violates its own definition; not writing output.")
    print("  ok  train: no held-out combination present")
    print("  ok  no cross-split sequence overlap other than test_ID <-> test_matched_ID")
    print("  ok  test_matched_ID motif-count distribution matches test_OOD, per label")
    print(f"  ok  test_OOD pos_frac {ood_pos:.4f} matches train+test_ID pos_frac {ref_pos:.4f} "
          f"within {balance_tolerance}")
    print(f"  ok  test_OOD spans {n_strata} motif-count strata: "
          f"{sorted(ood['num_selected_motifs'].unique())}")

    n_unique = matched["sequence_id"].nunique()
    n_total = len(matched)
    if n_unique < n_total:
        max_repeat = matched["sequence_id"].value_counts().max()
        print(f"  WARN  test_matched_ID is resampled with replacement: "
              f"{n_unique} unique sequence(s) fill {n_total} rows "
              f"(max {max_repeat} repeats of one sequence).")
    else:
        print(f"  ok  test_matched_ID needed no resampling with replacement: "
              f"all {n_total} rows are unique sequences.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--annotated", type=Path, default=here / "promoters_annotated.csv")
    ap.add_argument("--outdir", type=Path, default=here / "splits_v2")
    ap.add_argument("--seed", type=int, default=42)
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
