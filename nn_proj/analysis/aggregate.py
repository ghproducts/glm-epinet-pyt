"""Build every reported table from the raw prediction files.

Two tables come out of here, both long-form and both per-seed:

  calibration   one row per (backbone, method, train, test, seed) with error
                rate, ECE, proper scoring rules, and selective-prediction
                summaries.

  ood_detection one row per (backbone, method, train, id_test, ood_test,
                score, seed) with AUROC and its delta against the base
                model's total uncertainty.

Every figure and every number quoted in the manuscript is derived from these
two files, so a reader can recompute any of them without rerunning inference.

Usage
-----
    python -m nn_proj.analysis.aggregate <results_root> -o tables/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .metrics import (
    compute_auroc,
    compute_brier,
    compute_ece,
    compute_nll,
    confidence_oracle_curve,
    correctness,
    ece_bin_sensitivity,
    higher_is_more_uncertain,
)
from .results import ResultsIndex, RunKey
from .tasks import TaskPair

# Score used as the reference when reporting AUROC deltas.
BASE_SCORE = ("base", "U_total")

# A score whose full range falls below this carries no ranking information.
# The epistemic component of a deterministic model is analytically zero but
# computed as a difference of two entropy estimates, so it lands within a few
# units in the last place of zero. Ranking on that noise produces an AUROC
# that looks meaningful and is not.
DEGENERATE_SCORE_RANGE = 1e-6


def _label_overlap(df: pd.DataFrame) -> float:
    """Fraction of test labels that the model is capable of predicting.

    Predicted classes are drawn from the training label space, so labels that
    never appear as a prediction anywhere are outside it. This is a coarse
    but automatic check on whether a pair's error rate is 1.0 by construction
    rather than by model failure, which is the distinction the Near-OOD and
    OOD conditions turn on.
    """
    labels = set(df["labels"].astype(str))
    preds = set(df["pred"].astype(str))
    return len(labels & preds) / max(len(labels), 1)


def _is_degenerate(id_scores: pd.Series, ood_scores: pd.Series) -> bool:
    """Whether a score varies too little to rank anything.

    Applied jointly across the ID and OOD sets, since AUROC ranks their union.
    """
    both = pd.concat([id_scores, ood_scores]).to_numpy(dtype=float)
    both = both[np.isfinite(both)]
    if both.size == 0:
        return True
    return float(both.max() - both.min()) < DEGENERATE_SCORE_RANGE


def calibration_table(
    index: ResultsIndex,
    ece_bins: int = 50,
    include_scoring_rules: bool = True,
) -> pd.DataFrame:
    """Per-run calibration and selective-prediction metrics.

    ECE and error are computed for every run, but the ``supports_calibration``
    column records whether the registry considers them interpretable for that
    pair. Downstream reporting filters on it; the raw values are kept so the
    filtering decision stays visible and reversible.
    """
    rows: List[dict] = []
    for key in index:
        try:
            pair = index.registry.get(key.train, key.test)
        except KeyError:
            continue
        df = index.frame(key)

        correct = correctness(df)
        error = float(1.0 - correct.mean())

        row = {
            "backbone": key.backbone,
            "method": key.method,
            "train": key.train,
            "test": key.test,
            "rank": pair.rank or "",
            "category": pair.category,
            "target_shift": pair.target_shift,
            "supports_calibration": pair.supports_calibration,
            "seed": key.seed,
            "n": len(df),
            "n_label_classes": int(df["labels"].nunique()),
            "label_overlap": _label_overlap(df),
            "error": error,
            "ece": compute_ece(df, n_bins=ece_bins),
            "mean_confidence": float(df["max_confidence"].mean()),
        }

        for score in ("U_total", "U_aleatoric", "U_epistemic"):
            if score in df.columns:
                row[f"mean_{score}"] = float(df[score].mean())

        if include_scoring_rules:
            row["nll"] = compute_nll(df)
            row["brier"] = compute_brier(df)

        # Selective prediction. Meaningless when nothing is ever correct, so
        # skip rather than emit a degenerate curve.
        if 0.0 < error < 1.0:
            for score in ("max_confidence", "U_total"):
                if score not in df.columns:
                    continue
                try:
                    _, auco, aurc = confidence_oracle_curve(df, score_col=score)
                    row[f"auco_{score}"] = auco
                    row[f"aurc_{score}"] = aurc
                except ValueError:
                    pass

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["backbone", "method", "train", "test", "seed"]
    ).reset_index(drop=True)


def ood_table(index: ResultsIndex, scores: Optional[List[str]] = None) -> pd.DataFrame:
    """Per-run OOD-detection AUROC for every ID/OOD pairing and score.

    The ID anchor for each training task comes from ``ood_anchors`` in the
    task config. Deltas are taken against the base model's total uncertainty
    on the same (backbone, train, id_test, ood_test, seed), which is the
    comparison the manuscript's heatmaps report.
    """
    scores = scores or index.registry.scores
    reg = index.registry

    rows: List[dict] = []
    for id_pair, ood_pair in reg.ood_pairs():
        for backbone in index.backbones:
            for method in index.methods:
                for seed in index.seeds:
                    id_key = RunKey(seed, backbone, method, id_pair.train, id_pair.test)
                    ood_key = RunKey(seed, backbone, method, ood_pair.train, ood_pair.test)
                    if id_key not in index or ood_key not in index:
                        continue
                    id_df, ood_df = index.frame(id_key), index.frame(ood_key)

                    for score in scores:
                        if score not in id_df.columns or score not in ood_df.columns:
                            continue
                        if _is_degenerate(id_df[score], ood_df[score]):
                            continue
                        rows.append({
                            "backbone": backbone,
                            "method": method,
                            "train": ood_pair.train,
                            "rank": ood_pair.rank or "",
                            "id_test": id_pair.test,
                            "ood_test": ood_pair.test,
                            "id_category": id_pair.category,
                            "ood_category": ood_pair.category,
                            "score": score,
                            "seed": seed,
                            "auroc": compute_auroc(id_df, ood_df, score_col=score),
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    base = (df[(df.method == BASE_SCORE[0]) & (df.score == BASE_SCORE[1])]
            .set_index(["backbone", "train", "id_test", "ood_test", "seed"])["auroc"]
            .rename("base_auroc"))
    df = df.join(base, on=["backbone", "train", "id_test", "ood_test", "seed"])
    df["delta_auroc"] = df["auroc"] - df["base_auroc"]

    return df.sort_values(
        ["backbone", "train", "ood_test", "method", "score", "seed"]
    ).reset_index(drop=True)


def summarise(df: pd.DataFrame, value_cols: List[str], group_cols: List[str]) -> pd.DataFrame:
    """Mean, std, and seed count for ``value_cols`` grouped by ``group_cols``.

    Reported as mean ± std across seeds. ``n_seeds`` is carried through so a
    cell backed by two seeds is never silently presented like one backed by
    five.
    """
    present = [c for c in value_cols if c in df.columns]
    agg = df.groupby(group_cols, dropna=False)[present].agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg["n_seeds"] = df.groupby(group_cols, dropna=False)["seed"].nunique()
    return agg.reset_index()


def bin_sensitivity_table(index: ResultsIndex) -> pd.DataFrame:
    """ECE under varying bin counts and binning schemes.

    Supports the claim that the reported ECE ordering between methods is not
    an artefact of the M = 50 equal-mass choice.
    """
    rows = []
    seen = set()
    for key in index:
        try:
            pair = index.registry.get(key.train, key.test)
        except KeyError:
            continue
        if not pair.supports_calibration:
            continue
        # One representative seed per cell keeps the sweep cheap; the point
        # is the ordering between methods, not a seeded estimate.
        cell = (key.backbone, key.method, key.train, key.test)
        if cell in seen:
            continue
        seen.add(cell)

        sens = ece_bin_sensitivity(index.frame(key))
        sens = sens.assign(backbone=key.backbone, method=key.method,
                           train=key.train, test=key.test, seed=key.seed)
        rows.append(sens)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_root", help="Directory holding inference_results_<seed>/ folders")
    ap.add_argument("-o", "--outdir", default="tables", help="Where to write the CSVs")
    ap.add_argument("--ece-bins", type=int, default=50)
    ap.add_argument("--all-runs", action="store_true",
                    help="Index every run present, not just the canonical grid")
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="Skip the ECE bin-sensitivity sweep")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    index = ResultsIndex(args.results_root, canonical_only=not args.all_runs)
    print(index.summary(), "\n")

    missing = index.missing()
    if len(missing):
        missing.to_csv(outdir / "missing_runs.csv", index=False)
        print(f"[warn] {len(missing)} registered cells have no prediction file "
              f"-> {outdir/'missing_runs.csv'}")

    short = index.row_count_check()
    if len(short):
        short.to_csv(outdir / "row_count_anomalies.csv", index=False)
        print(f"[warn] {len(short)} run(s) have fewer rows than the rest of their "
              f"test set -> {outdir/'row_count_anomalies.csv'}")
    if index.repaired:
        print(f"[warn] {len(index.repaired)} file(s) needed row repair on load")

    print("Computing calibration table ...")
    cal = calibration_table(index, ece_bins=args.ece_bins)
    cal.to_csv(outdir / "calibration.csv", index=False)
    print(f"  {len(cal)} rows -> {outdir/'calibration.csv'}")

    cal_summary = summarise(
        cal,
        value_cols=["error", "ece", "nll", "brier", "mean_confidence",
                    "auco_max_confidence", "aurc_max_confidence",
                    "auco_U_total", "aurc_U_total"],
        group_cols=["backbone", "method", "train", "test", "category",
                    "target_shift", "supports_calibration"],
    )
    cal_summary.to_csv(outdir / "calibration_summary.csv", index=False)
    print(f"  {len(cal_summary)} rows -> {outdir/'calibration_summary.csv'}")

    print("Computing OOD detection table ...")
    ood = ood_table(index)
    ood.to_csv(outdir / "ood_detection.csv", index=False)
    print(f"  {len(ood)} rows -> {outdir/'ood_detection.csv'}")

    ood_summary = summarise(
        ood,
        value_cols=["auroc", "delta_auroc"],
        group_cols=["backbone", "method", "train", "id_test", "ood_test",
                    "ood_category", "score"],
    )
    ood_summary.to_csv(outdir / "ood_detection_summary.csv", index=False)
    print(f"  {len(ood_summary)} rows -> {outdir/'ood_detection_summary.csv'}")

    if not args.skip_sensitivity:
        print("Computing ECE bin sensitivity ...")
        sens = bin_sensitivity_table(index)
        if len(sens):
            sens.to_csv(outdir / "ece_bin_sensitivity.csv", index=False)
            print(f"  {len(sens)} rows -> {outdir/'ece_bin_sensitivity.csv'}")

    index.registry.to_frame().to_csv(outdir / "task_registry.csv", index=False)
    print(f"\nTask registry -> {outdir/'task_registry.csv'}")


if __name__ == "__main__":
    main()
