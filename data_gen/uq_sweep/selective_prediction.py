#!/usr/bin/env python3
"""Does each method's own uncertainty score identify its own errors?

This is what R2-3 actually asks for ("accuracy stratified by uncertainty, or
prediction-set size and coverage"). It is a different question from the
margin-quartile sanity check, which validates the *dataset* rather than any
method's uncertainty estimate.

For each (dataset, backbone, method, seed) cell we rank examples by that
method's own U_total, then report:

  acc_lowU / acc_highU  accuracy in the least- and most-uncertain decile
  aurc                  area under the risk-coverage curve (lower is better):
                        mean error over all coverage levels when abstaining on
                        the most uncertain examples first
  aurc_random           the same for a random ranking -- the no-skill reference
  aurc_oracle           ranking errors last, the best achievable
  selective_ratio       (random - actual) / (random - oracle); 1.0 = oracle,
                        0.0 = no better than chance, <0 = actively harmful

selective_ratio is the headline: it says how much of the achievable
error-ranking benefit the method's uncertainty actually delivers.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep_lib import normalize_axis_cols  # noqa: E402

RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "summaries")


def risk_coverage(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    """Mean error across coverage levels, abstaining on highest uncertainty
    first. Ties broken deterministically so repeated runs agree."""
    order = np.lexsort((np.arange(len(uncertainty)), uncertainty))  # ascending U
    err = 1.0 - correct[order]
    # cumulative error rate over the k most-confident examples, k = 1..n
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(cum.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="boundary,alisim")
    ap.add_argument("--score", default="U_total",
                    help="uncertainty column to rank by (U_total, U_epistemic, ...)")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")]
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "*", "*.csv"))):
        parts = path.split(os.sep)
        dataset, backbone = parts[-3], parts[-2]
        if dataset not in datasets:
            continue
        df = normalize_axis_cols(pd.read_csv(path))
        if args.score not in df.columns or "correct" not in df.columns:
            continue
        u = pd.to_numeric(df[args.score], errors="coerce").to_numpy(dtype=float)
        c = df["correct"].to_numpy(dtype=float)
        ok = ~np.isnan(u)
        u, c = u[ok], c[ok]
        if len(u) < 50 or np.unique(u).size < 5:
            continue  # constant/degenerate score (e.g. evidential U_epistemic == 0)

        n = len(u)
        k = max(1, n // 10)
        order = np.lexsort((np.arange(n), u))
        acc_low = float(c[order[:k]].mean())     # least uncertain decile
        acc_high = float(c[order[-k:]].mean())   # most uncertain decile

        aurc = risk_coverage(u, c)
        aurc_oracle = risk_coverage(1.0 - c, c)          # errors ranked last
        rng = np.random.default_rng(0)
        aurc_random = float(np.mean([risk_coverage(rng.permutation(n).astype(float), c)
                                     for _ in range(5)]))
        denom = aurc_random - aurc_oracle
        ratio = (aurc_random - aurc) / denom if denom > 1e-12 else float("nan")

        rows.append(dict(
            dataset=dataset, backbone=backbone,
            method=str(df["method"].iloc[0]), seed=str(df["seed"].iloc[0]),
            score=args.score, n=n, accuracy=float(c.mean()),
            acc_lowU_decile=acc_low, acc_highU_decile=acc_high,
            decile_gap=acc_low - acc_high,
            aurc=aurc, aurc_random=aurc_random, aurc_oracle=aurc_oracle,
            selective_ratio=ratio,
        ))

    if not rows:
        raise SystemExit("no usable cells found")
    out = pd.DataFrame(rows)
    path = os.path.join(OUT, f"selective_prediction_{args.score}.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out)} cells)")

    agg = (out.groupby(["dataset", "backbone", "method"])
              .agg(n_seeds=("seed", "nunique"),
                   acc=("accuracy", "mean"),
                   acc_lowU=("acc_lowU_decile", "mean"),
                   acc_highU=("acc_highU_decile", "mean"),
                   gap=("decile_gap", "mean"),
                   sel_ratio=("selective_ratio", "mean"),
                   sel_ratio_sd=("selective_ratio", "std"))
              .round(4).reset_index())
    agg.to_csv(os.path.join(OUT, f"selective_prediction_{args.score}_agg.csv"), index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
