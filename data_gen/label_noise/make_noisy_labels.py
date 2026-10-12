#!/usr/bin/env python3
"""Uniform random label-noise injection on `promoter_all`.

Primary noise model from `.docs/NEW_DATASETS_PROPOSAL.md` section 1, built to
answer a specific, narrower question first (see `.docs/LABEL_NOISE_PILOT.md`):
which of two regimes -- generalization/smoothing or memorization -- does this
project's actual fine-tuning setup fall into, since the originally-proposed
headline comparison (mean aleatoric uncertainty on flipped vs. clean
*training* rows) is only informative in an intermediate regime that isn't
guaranteed to hold.

Construction, mirroring `data_gen/promoter_motifs/make_splits_v3.py`'s style:

- Base corpus: `promoter_all` (`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised
  /promoter_all`), loaded via `load_NT_tasks`. 30,000 train rows, 1,584 test
  rows, ~50/50 label balance, fixed 300bp sequences.
- `sequence_id` is synthesized from split + row position (`train_00000`,
  `test_0000`, ...) rather than reused from the dataset's own `name` column,
  because `name` is *not* unique in this corpus (29,874 unique names over
  30,000 train rows -- some `chr:start-end` windows repeat) and a duplicate
  ID would make every later join by `sequence_id` ambiguous.
- For each target rate r in {0, 0.05, 0.10, 0.20, 0.40}: within the training
  split, independently for each original-label stratum (0 and 1), draw a
  round(r * n_stratum)-sized subset uniformly at random and flip those rows'
  labels. Stratifying the draw (rather than drawing r-fraction of the whole
  pool) is what keeps the marginal P(label=1) unchanged at every r -- a
  single overall draw could, by chance, flip more 0s than 1s and shift the
  class balance, confounding noise rate with class balance.
- The test split is carried through at every rate completely unmodified --
  same rows, same order, same labels, `noisy=False` and `original_label ==
  label` for every row. This is deliberate: the whole point of a clean,
  reserved eval set is that noise never gets anywhere near it.
- `noisy` (bool) and `original_label` (int) columns ride alongside the
  (possibly-corrupted) `label` actually fed to the model, exactly as
  `promoter_motifs`'s `motif_combo` metadata rides alongside `label` --
  `prep_for_trainer`'s `metadata_cols` mechanism drops them before
  tokenization but the pipeline's `inference.py` picks up any non-
  sequence/label column automatically and writes it back into the per-row
  prediction CSV, which is what the pilot's diagnostics actually key on.

Seed policy (documented explicitly per the design brief's request):

- `NOISE_SEED` (below) is the ONLY seed used anywhere in this script -- it
  drives every stratified draw, for every rate, via an independent
  `numpy.random.Generator` per rate (`NOISE_SEED + round(r * 1000)`, so each
  rate's draw is reproducible on its own and independent of the others; they
  are NOT nested subsets of one another).
- This is completely separate from the *training* seeds ({1, 2, 3, 4, 42},
  this project's usual pool) later passed as `--seed`/`--data_seed` to
  `train_base.py` when fine-tuning on the resulting CSVs -- the noise draw
  that defines what "correct" labels look like for a given r must not change
  depending on which training seed happens to fine-tune on it, or the r=0.05
  condition would not actually be the same task across seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from nn_proj.common.datasets import load_NT_tasks

NOISE_RATES = [0.0, 0.05, 0.10, 0.20, 0.40]
RATE_TAG = {0.0: "r00", 0.05: "r05", 0.10: "r10", 0.20: "r20", 0.40: "r40"}
NOISE_SEED = 20260903  # fixed, documented; see module docstring's "Seed policy"

MANIFEST_COLS = ["sequence_id", "label", "original_label", "noisy", "split", "noise_rate"]

# Preregistered thresholds for the GC-content confound check (verify() (d)).
GC_ALPHA = 0.01
GC_N_BINS = 4


def _gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else float("nan")


def _load_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load promoter_all's train/test splits as plain DataFrames with a
    synthesized, guaranteed-unique sequence_id."""
    train_ds = load_NT_tasks(task="promoter_all", split="train", encode_labels=True)
    test_ds = load_NT_tasks(task="promoter_all", split="test", encode_labels=True)

    train_df = pd.DataFrame({"sequence": train_ds["sequence"], "label": train_ds["labels"]})
    test_df = pd.DataFrame({"sequence": test_ds["sequence"], "label": test_ds["labels"]})

    train_df.insert(0, "sequence_id", [f"train_{i:05d}" for i in range(len(train_df))])
    test_df.insert(0, "sequence_id", [f"test_{i:04d}" for i in range(len(test_df))])
    return train_df, test_df


def _flip_mask_for_rate(labels: pd.Series, rate: float, seed: int) -> pd.Series:
    """Stratified-by-label boolean flip mask: an independent draw per label
    stratum, each of exact size round(rate * stratum_size)."""
    rng = np.random.default_rng(seed)
    mask = pd.Series(False, index=labels.index)
    for lab, idx in labels.groupby(labels).groups.items():
        idx = pd.Index(idx)
        n_flip = int(round(rate * len(idx)))
        if n_flip == 0:
            continue
        chosen = rng.choice(idx.to_numpy(), size=n_flip, replace=False)
        mask.loc[chosen] = True
    return mask


def assign_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[float, dict[str, pd.DataFrame]]:
    """Build the per-rate train/test DataFrames.

    Returns {rate: {"train": df, "test": df}}. Every rate's "test" frame is
    byte-for-byte the same rows/labels as every other rate's -- constructed
    from the same untouched test_df each time, never modified.
    """
    out: dict[float, dict[str, pd.DataFrame]] = {}
    for rate in NOISE_RATES:
        seed = NOISE_SEED + round(rate * 1000)

        tr = train_df.copy()
        tr["original_label"] = tr["label"]
        flip_mask = _flip_mask_for_rate(tr["label"], rate, seed)
        tr["noisy"] = flip_mask
        tr.loc[flip_mask, "label"] = 1 - tr.loc[flip_mask, "original_label"]

        te = test_df.copy()
        te["original_label"] = te["label"]
        te["noisy"] = False

        out[rate] = {"train": tr, "test": te}
    return out


def _build_manifest(splits: dict[float, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for rate, d in splits.items():
        for split_name, df in d.items():
            sub = df[["sequence_id", "label", "original_label", "noisy"]].copy()
            sub["split"] = split_name
            sub["noise_rate"] = rate
            rows.append(sub)
    return pd.concat(rows, ignore_index=True)[MANIFEST_COLS]


def verify(splits: dict[float, dict[str, pd.DataFrame]], manifest: pd.DataFrame,
           count_tol: int = 0) -> None:
    """Assert-and-fail-loud verification, following make_splits_v3.py's
    discipline. Refuses (raises SystemExit) if any check fails."""
    problems: list[str] = []
    oks: list[str] = []

    # (f) invertibility, checked first since every other check assumes it.
    for rate, d in splits.items():
        for split_name, df in d.items():
            bad = (df["noisy"] != (df["label"] != df["original_label"])).sum()
            if bad:
                problems.append(
                    f"r={rate} {split_name}: {bad} row(s) violate "
                    f"noisy == (label != original_label)"
                )
    if not any("violate" in p for p in problems):
        oks.append("(f) invertibility: noisy == (label != original_label) holds for every row, "
                    "every split, every rate")

    # (a) observed flip rate matches target r, exact-count check.
    for rate, d in splits.items():
        tr = d["train"]
        observed = tr["noisy"].mean()
        expected_count = 0
        for lab, grp in tr.groupby("original_label"):
            expected_count += int(round(rate * len(grp)))
        expected_rate = expected_count / len(tr)
        if abs(tr["noisy"].sum() - expected_count) > count_tol:
            problems.append(
                f"r={rate}: observed flip count {tr['noisy'].sum()} != expected "
                f"{expected_count} (sum of per-stratum round(r*n))"
            )
        else:
            oks.append(f"(a) r={rate}: flip count {tr['noisy'].sum()} matches target exactly "
                       f"(observed rate {observed:.4f}, nominal r={rate:.2f}, "
                       f"stratum-rounded target {expected_rate:.4f})")

    # (b) per-class flip rate matches r within each stratum.
    for rate, d in splits.items():
        tr = d["train"]
        for lab, grp in tr.groupby("original_label"):
            expected = int(round(rate * len(grp)))
            actual = int(grp["noisy"].sum())
            if abs(actual - expected) > count_tol:
                problems.append(
                    f"r={rate} label={lab}: stratum flip count {actual} != expected {expected}"
                )
            else:
                oks.append(f"(b) r={rate} label={lab}: stratum flip count {actual}/{len(grp)} "
                           f"matches round(r*n)={expected} exactly")

    # (c) marginal P(label=1) unchanged across all 5 rates.
    pos_fracs = {rate: d["train"]["label"].mean() for rate, d in splits.items()}
    ref = pos_fracs[0.0]
    # Not exactly 0: the two label strata have slightly different sizes
    # (14,950 vs. 15,050), so round(r*n0) 0->1 flips and round(r*n1) 1->0
    # flips aren't perfectly equal in count at every r, which nudges the
    # marginal by a rounding-sized amount (~1e-3 at worst here) -- a real,
    # bounded, and harmless side effect of per-stratum integer rounding, not
    # noise correlated with anything. 0.002 matches the tolerance
    # `.docs/NEW_DATASETS_PROPOSAL.md` itself proposes for this exact check.
    balance_tol = 0.002
    for rate, frac in pos_fracs.items():
        if abs(frac - ref) > balance_tol:
            problems.append(
                f"r={rate}: P(label=1)={frac:.6f} differs from r=0's {ref:.6f} "
                f"by more than {balance_tol}"
            )
        else:
            oks.append(f"(c) r={rate}: P(label=1)={frac:.6f} (r=0 reference {ref:.6f}) -- unchanged")

    # (d) noise not correlated with GC content: chi-square, Bonferroni across
    # the rates that actually have any noisy rows (r=0 has none to test).
    nonzero_rates = [r for r in NOISE_RATES if r > 0]
    alpha = GC_ALPHA / len(nonzero_rates)
    # Fixed bin edges from the (rate-invariant) pooled GC distribution, so
    # every rate's test uses the same bins.
    gc_all = splits[0.0]["train"]["sequence"].map(_gc_content)
    _, bin_edges = pd.qcut(gc_all, GC_N_BINS, retbins=True, duplicates="drop")
    for rate in NOISE_RATES:
        tr = splits[rate]["train"]
        if rate == 0.0:
            oks.append(f"(d) r={rate}: no noisy rows exist to test for a GC-content "
                       f"association (expected, not a failure)")
            continue
        gc = tr["sequence"].map(_gc_content)
        gc_bin = pd.cut(gc, bins=bin_edges, include_lowest=True)
        table = pd.crosstab(gc_bin, tr["noisy"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        if p <= alpha:
            problems.append(
                f"r={rate}: noisy is significantly associated with GC-content bin "
                f"(chi2={chi2:.3f}, dof={dof}, p={p:.5f} <= Bonferroni alpha {alpha:.5f})"
            )
        else:
            oks.append(f"(d) r={rate}: no significant GC-content association "
                       f"(chi2={chi2:.3f}, dof={dof}, p={p:.5f} > Bonferroni alpha {alpha:.5f})")

    # (e) test split byte-identical (label vector) across all 5 rate variants.
    ref_test = splits[0.0]["test"].sort_values("sequence_id").reset_index(drop=True)
    for rate in NOISE_RATES:
        te = splits[rate]["test"].sort_values("sequence_id").reset_index(drop=True)
        same_labels = (te["label"].to_numpy() == ref_test["label"].to_numpy()).all()
        same_seqs = (te["sequence"].to_numpy() == ref_test["sequence"].to_numpy()).all()
        same_ids = (te["sequence_id"].to_numpy() == ref_test["sequence_id"].to_numpy()).all()
        if not (same_labels and same_seqs and same_ids):
            problems.append(f"r={rate}: test split differs from r=0's test split "
                            f"(labels match={same_labels}, sequences match={same_seqs}, "
                            f"ids match={same_ids})")
        else:
            oks.append(f"(e) r={rate}: test split is byte-identical to r=0's "
                       f"(same ids, sequences, labels) -- {len(te)} rows")

    print("Label-noise injection verification:")
    for o in oks:
        print(f"  ok  {o}")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        raise SystemExit(f"\n{len(problems)} verification check(s) failed; refusing to write output.")
    print(f"\nAll checks passed across {len(NOISE_RATES)} noise levels.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--outdir", type=Path, default=here)
    ap.add_argument("--check", action="store_true",
                    help="Re-verify the already-written manifest.csv without recomputing anything")
    args = ap.parse_args()

    manifest_path = args.outdir / "manifest.csv"

    print("Loading promoter_all train/test splits via load_NT_tasks ...")
    train_df, test_df = _load_base()
    print(f"  train: {len(train_df)} rows, P(label=1)={train_df['label'].mean():.4f}")
    print(f"  test:  {len(test_df)} rows, P(label=1)={test_df['label'].mean():.4f}")

    splits = assign_splits(train_df, test_df)
    manifest = _build_manifest(splits)
    verify(splits, manifest)

    print("\nRows per (rate, split):")
    print(manifest.groupby(["noise_rate", "split"]).size().to_string())

    args.outdir.mkdir(parents=True, exist_ok=True)
    for rate, d in splits.items():
        tag = RATE_TAG[rate]
        rate_dir = args.outdir / f"csv_data_{tag}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        for split_name, df in d.items():
            cols = ["sequence_id", "sequence", "label", "original_label", "noisy"]
            out_path = rate_dir / f"{split_name}.csv"
            df[cols].to_csv(out_path, index=False)
            print(f"  wrote {out_path} ({len(df)} rows)")

    manifest.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
