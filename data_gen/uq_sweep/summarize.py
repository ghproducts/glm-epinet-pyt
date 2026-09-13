#!/usr/bin/env python3
"""Aggregate sweep cells into calibration and decomposition tables.

Reads results/<dataset>/<backbone>/<method>__seed<N>.csv and writes tidy
long-format CSVs to summaries/. Because the sweep persists per-example
probabilities, ECE/NLL/Brier are computed directly rather than reconstructed by
inverting entropy, and because every cell is per-seed, everything is reported
as mean +/- sd across seeds rather than a single run.

    python summarize.py                 # everything present
    python summarize.py --datasets boundary

Outputs:
  summaries/calibration_by_stratum.csv   acc/ECE/NLL/Brier per stratum per seed
  summaries/calibration_agg.csv          the above, mean+/-sd across seeds
  summaries/decomposition_by_stratum.csv U_aleatoric/U_epistemic per stratum
  summaries/decomposition_agg.csv        the above, mean+/-sd across seeds
  summaries/response_stats.csv           per-seed Spearman rho vs the shift axis
  summaries/bin_sensitivity.csv          ECE at 10/15/20/25 bins + adaptive
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "summaries")

BIN_COUNTS = (10, 15, 20, 25)
# which column defines the shift axis for each dataset, and how strata are named
AXIS = {"boundary": "quartile", "alisim": "branch_length",
        "alisim_extended": "branch_length"}


def _prob_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("prob_") and c[5:].isdigit()],
                  key=lambda c: int(c[5:]))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    """Equal-width binned |accuracy - confidence|, weighted by bin population."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def adaptive_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    """Equal-mass (quantile) bins -- ACE. Less sensitive to the confidence
    distribution piling up near 1.0 than equal-width bins."""
    if len(conf) < n_bins:
        return float("nan")
    order = np.argsort(conf)
    total = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        total += (len(chunk) / len(conf)) * abs(correct[chunk].mean() - conf[chunk].mean())
    return float(total)


def scores(df: pd.DataFrame) -> dict:
    """Accuracy plus proper scoring rules, from the persisted probabilities.

    `brier` is the multi-class form, summed over classes
    (sum_c (p_c - y_c)^2), so for a binary task it is twice the single-class
    (p - y)^2 convention some papers report. Stated here because the two
    differ by exactly 2x and are easy to compare by mistake.
    """
    pcols = _prob_cols(df)
    P = df[pcols].to_numpy(dtype=float)
    y = df["labels"].to_numpy(dtype=int)
    correct = df["correct"].to_numpy(dtype=float)
    conf = P.max(axis=1)
    p_true = P[np.arange(len(y)), y]
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0

    out = {
        "n": len(df),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(p_true, 1e-12, None)).mean()),
        "brier": float(((P - onehot) ** 2).sum(axis=1).mean()),
        "mean_confidence": float(conf.mean()),
        "ace_15bin": adaptive_ece(conf, correct, 15),
    }
    for b in BIN_COUNTS:
        out[f"ece_{b}bin"] = ece(conf, correct, b)
    out["ece_range"] = max(out[f"ece_{b}bin"] for b in BIN_COUNTS) - \
        min(out[f"ece_{b}bin"] for b in BIN_COUNTS)
    for col in ("U_total", "U_aleatoric", "U_epistemic", "vacuity",
                "set_size", "covered"):
        if col in df.columns and df[col].notna().any():
            out[f"mean_{col}"] = float(pd.to_numeric(df[col], errors="coerce").mean())
    return out


def load_cells(datasets: list[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "*", "*.csv"))):
        parts = path.split(os.sep)
        dataset, backbone = parts[-3], parts[-2]
        if dataset not in datasets:
            continue
        df = pd.read_csv(path)
        if "correct" not in df.columns or not _prob_cols(df):
            print(f"[skip] {path}: missing correct/prob columns")
            continue
        method = str(df["method"].iloc[0])
        seed = str(df["seed"].iloc[0])
        axis = AXIS.get(dataset)
        df["_cell"] = f"{dataset}/{backbone}/{method}/{seed}"
        rows.append((dataset, backbone, method, seed, axis, df))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="boundary,alisim,alisim_extended")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")]
    os.makedirs(OUT, exist_ok=True)

    cells = load_cells(datasets)
    if not cells:
        raise SystemExit(f"no result cells found under {RESULTS}")
    print(f"loaded {len(cells)} cells")

    calib, decomp, resp = [], [], []
    for dataset, backbone, method, seed, axis, df in cells:
        key = dict(dataset=dataset, backbone=backbone, method=method, seed=seed)

        # pooled over the whole dataset, plus per-stratum
        strata = [("__pooled__", df)]
        if axis and axis in df.columns:
            strata += [(s, g) for s, g in df.groupby(axis, sort=True)]
        for stratum, g in strata:
            calib.append({**key, "stratum": stratum, **scores(g)})
            if {"U_aleatoric", "U_epistemic"} <= set(g.columns):
                d = {**key, "stratum": stratum, "n": len(g),
                     "u_aleatoric": float(g.U_aleatoric.mean()),
                     "u_epistemic": float(g.U_epistemic.mean()),
                     "u_total": float(g.U_total.mean()) if "U_total" in g else float("nan")}
                if "vacuity" in g.columns and g.vacuity.notna().any():
                    d["vacuity"] = float(g.vacuity.mean())
                decomp.append(d)

        # response of each uncertainty score to the continuous shift axis
        if axis and axis in df.columns:
            x = df[axis]
            if axis == "quartile":  # Q1..Q4 -> ordinal; use margin when present
                x = df["margin"] if "margin" in df.columns else x.astype("category").cat.codes
            x = pd.to_numeric(x, errors="coerce")
            row = {**key, "axis": axis, "n": len(df)}
            for col in ("U_total", "U_aleatoric", "U_epistemic", "vacuity"):
                if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
                    v = pd.to_numeric(df[col], errors="coerce")
                    ok = x.notna() & v.notna()
                    if ok.sum() > 2 and v[ok].nunique() > 1:
                        rho, p = spearmanr(x[ok], v[ok])
                        row[f"rho_{col}"] = float(rho)
                        row[f"p_{col}"] = float(p)
            resp.append(row)

    calib_df = pd.DataFrame(calib)
    decomp_df = pd.DataFrame(decomp)
    resp_df = pd.DataFrame(resp)
    calib_df.to_csv(os.path.join(OUT, "calibration_by_stratum.csv"), index=False)
    decomp_df.to_csv(os.path.join(OUT, "decomposition_by_stratum.csv"), index=False)
    resp_df.to_csv(os.path.join(OUT, "response_stats.csv"), index=False)

    # aggregate across seeds: mean +/- sd, with the seed count made explicit so a
    # single-seed cell can never be mistaken for a replicated one
    def agg(df, value_cols, group):
        present = [c for c in value_cols if c in df.columns]
        g = df.groupby(group, dropna=False)[present].agg(["mean", "std", "count"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        return g.reset_index()

    gkeys = ["dataset", "backbone", "method", "stratum"]
    agg(calib_df, ["accuracy", "nll", "brier", "ece_15bin", "ace_15bin",
                   "mean_confidence", "mean_set_size", "mean_covered"],
        gkeys).to_csv(os.path.join(OUT, "calibration_agg.csv"), index=False)
    agg(decomp_df, ["u_aleatoric", "u_epistemic", "u_total", "vacuity"],
        gkeys).to_csv(os.path.join(OUT, "decomposition_agg.csv"), index=False)

    bins = calib_df[calib_df.stratum == "__pooled__"][
        gkeys + ["seed"] + [f"ece_{b}bin" for b in BIN_COUNTS] + ["ace_15bin", "ece_range"]]
    bins.to_csv(os.path.join(OUT, "bin_sensitivity.csv"), index=False)

    print(f"wrote 6 files to {OUT}")
    nseeds = calib_df.groupby(["dataset", "backbone", "method"]).seed.nunique()
    single = nseeds[nseeds < 2]
    print(f"cells with >=2 seeds: {(nseeds >= 2).sum()} / {len(nseeds)}")
    if len(single):
        print("single-seed (no sd available):")
        for k, v in single.items():
            print(f"  {'/'.join(k)}: {v} seed")


if __name__ == "__main__":
    main()
