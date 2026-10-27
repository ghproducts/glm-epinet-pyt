#!/usr/bin/env python3
"""Markdown tables for the promoter UQ sweep, from summarize.py's summaries/ CSVs.

    SWEEP_SUMMARIES_DIR=... python render_promoter.py > tables.md

Tables: pooled calibration per dataset (mean +/- sd over seeds), accuracy and epistemic
uncertainty per branch length (AliSim) and margin quartile (boundary), Spearman rho of each
uncertainty component against the shift axis, and split-conformal coverage / set size.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUM = os.environ.get("SWEEP_SUMMARIES_DIR") or os.path.join(HERE, "summaries")
RES = os.environ.get("SWEEP_RESULTS_DIR") or os.path.join(HERE, "results")
BB = ["NT_transformer", "DNABERT2", "hyenaDNA", "CARMANIA", "nonpretrained"]
METHODS = ["base", "base_scaled", "mc_dropout", "conv_epinet", "laplace", "conformal",
           "evidential", "ensemble_k5", "rf", "cnn_mc_dropout", "cnn_ensemble"]
NAME = {"NT_transformer": "NT", "nonpretrained": "no pretraining"}
MNAME = {"base": "softmax", "base_scaled": "temperature", "mc_dropout": "MC dropout",
         "conv_epinet": "Epinet", "laplace": "Laplace", "conformal": "conformal",
         "evidential": "evidential", "ensemble_k5": "ensemble (5 seeds)", "rf": "k-mer RF",
         "cnn_mc_dropout": "CNN + MC dropout", "cnn_ensemble": "CNN ensemble"}
DSNAME = {"boundary": "Aleatoric shift (margin boundary)", "alisim": "Epistemic shift (AliSim, t <= 3)",
          "alisim_extended": "Epistemic shift, extended (AliSim, t = 5, 8, 12)"}


def order(df):
    df = df[df.method.isin(METHODS)].copy()
    df["_b"] = df.backbone.map(BB.index)
    df["_m"] = df.method.map(METHODS.index)
    return df.sort_values(["_b", "_m"])


def pm(mean, sd, nd=3):
    if pd.isna(mean):
        return "-"
    return f"{mean:.{nd}f}" if pd.isna(sd) else f"{mean:.{nd}f} ±{sd:.{nd}f}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    cal = pd.read_csv(os.path.join(SUM, "calibration_agg.csv"))
    dec = pd.read_csv(os.path.join(SUM, "decomposition_agg.csv"))
    rho = pd.read_csv(os.path.join(SUM, "response_stats.csv"))
    for ds in ["boundary", "alisim", "alisim_extended"]:
        c = order(cal[(cal.dataset == ds) & (cal.stratum == "__pooled__")])
        if c.empty:
            continue
        t = pd.DataFrame({
            "backbone": c.backbone.map(lambda b: NAME.get(b, b)), "method": c.method.map(MNAME),
            "seeds": c.accuracy_count.astype(int),
            "accuracy": [pm(a, b) for a, b in zip(c.accuracy_mean, c.accuracy_std)],
            "ECE": [pm(a, b) for a, b in zip(c.ece_15bin_mean, c.ece_15bin_std)],
            "NLL": [pm(a, b) for a, b in zip(c.nll_mean, c.nll_std)],
            "Brier": [pm(a, b) for a, b in zip(c.brier_mean, c.brier_std)]})
        print(f"### {DSNAME[ds]}: calibration (pooled; mean ±sd over seeds)\n\n{md(t)}\n")

    # accuracy / epistemic per stratum
    for ds, label in [("alisim", "branch length"), ("boundary", "margin quartile (Q1 = nearest the boundary)")]:
        for val, col, title in [("accuracy_mean", cal, "accuracy"), ("u_epistemic_mean", dec, "epistemic uncertainty"),
                                ("u_aleatoric_mean", dec, "aleatoric uncertainty")]:
            d = col[(col.dataset == ds) & (col.stratum != "__pooled__")]
            if d.empty:
                continue
            d = order(d)
            d["row"] = d.backbone.map(lambda b: NAME.get(b, b)) + " / " + d.method.map(MNAME)
            if ds == "alisim":
                d["stratum"] = d.stratum.astype(float)
            p = d.pivot_table(index="row", columns="stratum", values=val, sort=False)
            p = p[sorted(p.columns)]
            p.columns = [f"{x:g}" if isinstance(x, float) else str(x) for x in p.columns]
            p = p.round(3).reset_index().rename(columns={"row": "backbone / method"})
            print(f"### {DSNAME[ds]}: {title} by {label} (mean over seeds)\n\n{md(p)}\n")

    # Spearman rho vs the shift axis, mean over seeds
    r = rho[rho.dataset.isin(["boundary", "alisim"])]
    g = r.groupby(["dataset", "backbone", "method"])[["rho_U_aleatoric", "rho_U_epistemic", "rho_U_total"]].mean().reset_index()
    w = g.pivot_table(index=["backbone", "method"], columns="dataset",
                      values=["rho_U_aleatoric", "rho_U_epistemic"]).reset_index()
    w.columns = ["backbone", "method"] + [f"{ds} {v.split('_')[-1]}" for v, ds in w.columns[2:]]
    w = order(w)
    keep = ["boundary aleatoric", "boundary epistemic", "alisim aleatoric", "alisim epistemic"]
    t = w[["backbone", "method"] + [k for k in keep if k in w.columns]].copy()
    t["backbone"] = t.backbone.map(lambda b: NAME.get(b, b)); t["method"] = t.method.map(MNAME)
    t = t.round(3).rename(columns={"boundary aleatoric": "boundary: aleatoric ρ", "boundary epistemic": "boundary: epistemic ρ",
                                   "alisim aleatoric": "AliSim: aleatoric ρ", "alisim epistemic": "AliSim: epistemic ρ"})
    print("### Spearman ρ of each uncertainty component vs the shift axis (mean over seeds)\n\n"
          "Boundary: vs margin (negative = more uncertain nearer the boundary). AliSim: vs branch length "
          "(positive = more uncertain with divergence).\n\n" + md(t) + "\n")

    # conformal coverage and set size
    rows = []
    for ds in ["boundary", "alisim", "alisim_extended"]:
        for f in sorted(glob.glob(os.path.join(RES, ds, "*", "conformal__seed*.csv"))):
            d = pd.read_csv(f)
            rows.append({"dataset": ds, "backbone": f.split(os.sep)[-2], "coverage": d.covered.mean(),
                         "set size": d.set_size.mean(), "empty": (d.set_size == 0).mean(), "both classes": (d.set_size == 2).mean()})
    if rows:
        c = pd.DataFrame(rows).groupby(["dataset", "backbone"]).mean().reset_index()
        c["_b"] = c.backbone.map(BB.index); c = c.sort_values(["dataset", "_b"]).drop(columns="_b")
        c["backbone"] = c.backbone.map(lambda b: NAME.get(b, b))
        print("### Split conformal (alpha 0.1, calibrated on each seed's validation split; target coverage 0.90)\n\n"
              + md(c.round(3)) + "\n")


if __name__ == "__main__":
    main()
