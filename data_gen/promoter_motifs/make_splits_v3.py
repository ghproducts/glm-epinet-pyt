#!/usr/bin/env python3
"""v3 split: separates "novel combination" from "novel motif entirely".

v2 (``make_splits_v2.py``) fixed v1's class-balance and motif-count
confounds, but it only tested one kind of novelty: combinations of motifs
the model had each seen individually elsewhere. It can't say anything about
a motif the model has *never* seen in any context.

v3 adds that second axis as a genuinely separate condition, rather than
folding it into the existing one, because the two kinds of novelty aren't
interchangeable: a combination of familiar motifs lets the model draw on
independent prior exposure to each piece; an entirely unseen motif offers no
such foothold. Checked against the corpus before committing to this design:
``ELK3`` is the only one of the six JASPAR motifs that never co-occurs with
anything else here (every other motif is entangled across multiple eligible
combinations, so removing any of them entirely would also remove those
combinations from training -- for e.g. ``ZNF131`` that means losing the
*only* eligible combinations at motif-count 3 and 4). ``ELK3`` is the one
motif that can be held out completely without collateral damage.

Six splits come out of this, not four:

    train                    fine-tuning data
    test_ID                  general population baseline (not a shift
                              comparison anchor -- see docs/REVISION_PLAN.md
                              for why plain test_ID is confounded against
                              either shift condition)
    test_matched_ID_nearID   matched control for test_nearID: same label
                              balance, same motif-count profile (spans 1 & 2),
                              every combination in it was in training
    test_matched_ID_nearOOD  matched control for test_nearOOD: same label
                              balance, same motif-count profile (count 1
                              only), every combination in it was in training
    test_nearID              the five held-out *combinations* from v2
                              (EGR1+SP2, SP2+TFDP1, SP2+ZNF131, ZBED4+ZNF131,
                              TFDP1) -- motifs individually seen, this
                              pairing wasn't
    test_nearOOD             ELK3 only, held out of training entirely --
                              never seen alone or in combination

test_matched_ID_nearID and test_matched_ID_nearOOD are deliberately two
separate sets, not one shared one: test_nearID's held-out pool spans
motif-count strata {1, 2}, test_nearOOD's is entirely stratum {1}. A single
combined matched set would match neither profile exactly and would reopen
the motif-count confound this whole design exists to close. They may (and do)
share underlying sequences with each other -- that's expected, since neither
needs to be disjoint from the other, only from train.

Same technique as v2 throughout: stratified positive-class downsampling to
match train's label balance, seeded per-(label, num_selected_motifs)
resampling for the matched controls, the same verification discipline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

NEAR_ID_COMBOS = {"TFDP1", "SP2+TFDP1", "EGR1+SP2", "SP2+ZNF131", "ZBED4+ZNF131"}
NEAR_OOD_COMBOS = {"ELK3"}
NONHELD_COMBOS = {"none", "SP2", "EGR1", "ZBED4", "ZNF131",
                   "SP2+ZBED4", "SP2+ZBED4+ZNF131", "EGR1+SP2+ZBED4+ZNF131"}
ELIGIBLE_COMBOS = NEAR_ID_COMBOS | NEAR_OOD_COMBOS | NONHELD_COMBOS

MANIFEST_COLS = ["sequence_id", "label", "label_name", "motif_combo",
                  "num_selected_motifs", "SP2", "ZBED4", "EGR1", "ELK3",
                  "TFDP1", "ZNF131", "split"]


def _balance_correct(pool: pd.DataFrame, target_pos_frac: float, seed: int) -> pd.DataFrame:
    """Keep every negative; downsample positives, stratified per combination,
    to hit target_pos_frac exactly. Same technique as v2's assign_splits()."""
    neg = pool[pool["label"] == 0]
    pos = pool[pool["label"] == 1]
    pos_needed = int(round(len(neg) * target_pos_frac / (1 - target_pos_frac)))
    frac = pos_needed / len(pos) if len(pos) else 0.0

    parts = [neg]
    for _, g in pos.groupby("motif_combo"):
        n_take = min(len(g), max(1, int(round(len(g) * frac))))
        parts.append(g.sample(n=n_take, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed)


def _matched_control(target: pd.DataFrame, source_pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Resample source_pool to match target's exact per-(label, num_selected_motifs)
    group sizes -- and, within each such group, spread evenly across whichever
    distinct motif_combo values are available there, rather than drawing a
    plain random sample.

    A plain random draw is dominated by whichever combo happens to be most
    abundant in the corpus (e.g. SP2, the most common single motif). If that
    combo is an unusually weak or strong predictor of the label on its own,
    the "matched" control stops representing familiar material in general
    and starts reflecting one combo's idiosyncrasies -- this is what produced
    test_nearOOD (ELK3) reading as *easier* than test_matched_ID_nearOOD in
    the first version of this script: the control was 59% SP2 (pos_frac
    0.41, a near-chance predictor) purely because SP2 is abundant, not
    because it's representative. See docs/REVISION_PLAN.md.
    """
    parts = []
    for (label, n_motifs), group in target.groupby(["label", "num_selected_motifs"]):
        candidate = source_pool[(source_pool["label"] == label) &
                                 (source_pool["num_selected_motifs"] == n_motifs)]
        combos = sorted(candidate["motif_combo"].unique())
        n_needed = len(group)
        base, remainder = divmod(n_needed, len(combos))
        for i, combo in enumerate(combos):
            n_take = base + (1 if i < remainder else 0)
            if n_take == 0:
                continue
            pool_c = candidate[candidate["motif_combo"] == combo]
            parts.append(pool_c.sample(n=n_take, replace=len(pool_c) < n_take, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed) if parts else source_pool.iloc[0:0]


def assign_splits(annot: pd.DataFrame, seed: int, test_size: float) -> pd.DataFrame:
    df = annot[annot["motif_combo"].isin(ELIGIBLE_COMBOS)].copy()

    nonheld = df[df["motif_combo"].isin(NONHELD_COMBOS)].copy()
    near_id_pool = df[df["motif_combo"].isin(NEAR_ID_COMBOS)].copy()
    near_ood_pool = df[df["motif_combo"].isin(NEAR_OOD_COMBOS)].copy()

    train_idx, id_idx = train_test_split(
        nonheld.index, test_size=test_size, random_state=seed, stratify=nonheld["label"],
    )
    train = nonheld.loc[train_idx].copy()
    test_id = nonheld.loc[id_idx].copy()

    target_pos_frac = nonheld["label"].mean()
    test_near_id = _balance_correct(near_id_pool, target_pos_frac, seed)
    test_near_ood = _balance_correct(near_ood_pool, target_pos_frac, seed)

    matched_near_id = _matched_control(test_near_id, test_id, seed)
    matched_near_ood = _matched_control(test_near_ood, test_id, seed)

    train["split"] = "train"
    test_id["split"] = "test_ID"
    matched_near_id["split"] = "test_matched_ID_nearID"
    matched_near_ood["split"] = "test_matched_ID_nearOOD"
    test_near_id["split"] = "test_nearID"
    test_near_ood["split"] = "test_nearOOD"

    manifest = pd.concat([train, test_id, matched_near_id, matched_near_ood,
                           test_near_id, test_near_ood], ignore_index=True)
    return manifest[MANIFEST_COLS]


def verify(manifest: pd.DataFrame, balance_tolerance: float = 0.02) -> None:
    problems = []

    train = manifest[manifest["split"] == "train"]
    test_id = manifest[manifest["split"] == "test_ID"]
    matched_nid = manifest[manifest["split"] == "test_matched_ID_nearID"]
    matched_nood = manifest[manifest["split"] == "test_matched_ID_nearOOD"]
    near_id = manifest[manifest["split"] == "test_nearID"]
    near_ood = manifest[manifest["split"] == "test_nearOOD"]
    nonheld_ref = pd.concat([train, test_id])

    leaked = set(train["motif_combo"]) & (NEAR_ID_COMBOS | NEAR_OOD_COMBOS)
    if leaked:
        problems.append(f"train contains held-out combo(s) {sorted(leaked)}")

    # Cross-split overlap: allowed only where matched-ID controls draw from
    # test_ID (and from each other, since they're not required to be
    # disjoint from one another -- only from train).
    allowed_overlap_pairs = {
        frozenset({"test_ID", "test_matched_ID_nearID"}),
        frozenset({"test_ID", "test_matched_ID_nearOOD"}),
        frozenset({"test_matched_ID_nearID", "test_matched_ID_nearOOD"}),
    }
    by_split = {name: set(sub["sequence_id"]) for name, sub in manifest.groupby("split")}
    names = list(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = by_split[a] & by_split[b]
            if overlap and frozenset({a, b}) not in allowed_overlap_pairs:
                problems.append(f"{len(overlap)} sequence(s) appear in both {a!r} and {b!r}")

    def _check_matched(name, target, matched):
        for label in sorted(target["label"].unique()):
            t_counts = target[target["label"] == label]["num_selected_motifs"].value_counts()
            m_counts = matched[matched["label"] == label]["num_selected_motifs"].value_counts()
            if not t_counts.sort_index().equals(m_counts.reindex(t_counts.index).fillna(-1).sort_index()):
                problems.append(f"{name}: label {label} motif-count distribution "
                                 f"does not match its target")

    _check_matched("test_matched_ID_nearID", near_id, matched_nid)
    _check_matched("test_matched_ID_nearOOD", near_ood, matched_nood)

    ref_pos = nonheld_ref["label"].mean()
    for name, sub in [("test_nearID", near_id), ("test_nearOOD", near_ood)]:
        pos = sub["label"].mean()
        if abs(ref_pos - pos) > balance_tolerance:
            problems.append(f"{name} pos_frac {pos:.4f} vs. train+test_ID pos_frac "
                             f"{ref_pos:.4f} differs by more than {balance_tolerance}")

    n_strata_nid = near_id["num_selected_motifs"].nunique()
    if n_strata_nid < 2:
        problems.append(f"test_nearID sits in only {n_strata_nid} motif-count stratum; "
                         "the point of this split is spanning more than one")
    # test_nearOOD is expected to be single-stratum -- ELK3 only ever appears
    # as a singleton in this corpus, so unlike test_nearID this is not a
    # defect to check for.

    print("Split verification (v3):")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        raise SystemExit("Split assignment violates its own definition; not writing output.")
    print("  ok  train: no held-out combination present")
    print("  ok  no cross-split sequence overlap other than the matched-ID/test_ID trio")
    print("  ok  test_matched_ID_nearID matches test_nearID's motif-count distribution, per label")
    print("  ok  test_matched_ID_nearOOD matches test_nearOOD's motif-count distribution, per label")
    print(f"  ok  test_nearID pos_frac matches train+test_ID pos_frac ({ref_pos:.4f}) "
          f"within {balance_tolerance}")
    print(f"  ok  test_nearOOD pos_frac matches train+test_ID pos_frac ({ref_pos:.4f}) "
          f"within {balance_tolerance}")
    print(f"  ok  test_nearID spans {n_strata_nid} motif-count strata: "
          f"{sorted(near_id['num_selected_motifs'].unique())}")
    print(f"  ok  test_nearOOD is single-stratum by design: "
          f"{sorted(near_ood['num_selected_motifs'].unique())}")

    for name, sub in [("test_matched_ID_nearID", matched_nid),
                       ("test_matched_ID_nearOOD", matched_nood)]:
        n_unique, n_total = sub["sequence_id"].nunique(), len(sub)
        if n_unique < n_total:
            max_repeat = sub["sequence_id"].value_counts().max()
            print(f"  WARN  {name} is resampled with replacement: {n_unique} unique "
                  f"sequence(s) fill {n_total} rows (max {max_repeat} repeats).")
        else:
            print(f"  ok  {name} needed no resampling with replacement: "
                  f"all {n_total} rows are unique sequences.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--annotated", type=Path, default=here / "promoters_annotated.csv")
    ap.add_argument("--outdir", type=Path, default=here / "splits_v3")
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
        print(f"  {name:<24} {count}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
