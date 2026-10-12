#!/usr/bin/env python3
"""Figures for the dark matter section of the weekly report.

  fig1  darkmatter_trend_both.pdf   flagged fraction vs `up`, both encoders
  fig2  darkmatter_slope_vs_fpr.pdf slope of that fraction across the FPR sweep
  fig3  darkmatter_roc_by_up.pdf    ROC of id_test vs each `up` level

Scores come from results/darkmatter_all_methods_<rank>_scores.npz, which holds
one scalar per read for every method under both encoders. SCORE selects which;
k-NN (k=1) is used because it is the only family of scores that (a) is positive
at strict thresholds under BOTH encoders, (b) agrees across k=1/10/50, and (c)
does not depend on the classifier head, which for the 429-way family model is
near-uniform at 150bp and therefore unusable.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

SCORE = "knn_k1"
COL = {"phylum": "tab:purple", "family": "tab:cyan"}
UPS = [0.25, 0.75, 1.0]


def load(rank):
    z = np.load(f"results/darkmatter_all_methods_{rank}_scores.npz")
    return (pd.DataFrame({"up": z["up"], "sample": z["sample"], "v": z[f"dm__{SCORE}"]}),
            z[f"id__{SCORE}"])


def sweep(dm, si, fprs):
    sl, pv = [], []
    for f in fprs:
        t = np.quantile(si, 1 - f)
        rr = dm.assign(fl=(dm.v > t).astype(float)).groupby(["up", "sample"]).fl.mean().reset_index()
        s, _, _, p, _ = stats.linregress(rr.up, rr.fl)
        sl.append(s); pv.append(p)
    return np.array(sl), np.array(pv)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--outdir", type=Path, default=Path(".docs/latex/figs"))
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.outdir.mkdir(parents=True, exist_ok=True)
    R = {r: load(r) for r in ("phylum", "family")}

    # ---- fig 1: flagged fraction vs up, one panel per encoder ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
    for ax, rank in zip(axes, ("phylum", "family")):
        dm, si = R[rank]
        T = np.quantile(si, 1 - args.fpr)
        rep = dm.assign(fl=(dm.v > T).astype(float)).groupby(["up", "sample"]).fl.mean().reset_index()
        s, ic, _, p, _ = stats.linregress(rep.up, rep.fl)
        ax.scatter(rep.up * 100, rep.fl, s=62, color=COL[rank], edgecolors="black",
                   linewidths=.6, zorder=3)
        xs = np.linspace(0.20, 1.05, 50)
        ax.plot(xs * 100, ic + s * xs, ls="--", lw=2.2, color=COL[rank], zorder=2)
        ax.set_xlabel("Unknown percentage")
        ax.set_title(rank.capitalize(), fontsize=11, loc="left")
        print(f"  fig1 {rank}: slope={s:+.4f} p={p:.4f}")
        ax.set_xticks([25, 50, 75, 100]); ax.grid(alpha=.3, ls=":"); ax.set_axisbelow(True)
    axes[0].set_ylabel("Fraction flagged")
    fig.tight_layout(); fig.savefig(args.outdir / "darkmatter_trend_both.pdf")
    fig.savefig(args.outdir / "darkmatter_trend_both.png", dpi=140); plt.close(fig)

    # ---- fig 2: slope across the FPR sweep, both encoders on one axes --
    fprs = np.concatenate([np.arange(0.002, 0.05, 0.002), np.arange(0.05, 0.55, 0.025)])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0, color="black", lw=1)
    for rank in ("phylum", "family"):
        dm, si = R[rank]
        sl, pv = sweep(dm, si, fprs)
        ax.plot(fprs * 100, sl, color=COL[rank], lw=2, label=f"{rank} encoder")
        sig = pv < 0.05
        ax.scatter(fprs[sig] * 100, sl[sig], s=26, color=COL[rank], zorder=3)
        ax.scatter(fprs[~sig] * 100, sl[~sig], s=16, facecolors="none",
                   edgecolors=COL[rank], zorder=3)
        pos = sig & (sl > 0)
        if pos.any():
            print(f"{rank}: positive & significant for FPR "
                  f"{fprs[pos].min():.1%}-{fprs[pos].max():.1%}, peak {sl[pos].max():+.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("False positive rate (%)")
    ax.set_ylabel("Slope")
    ax.grid(alpha=.3, ls=":"); ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(args.outdir / "darkmatter_slope_vs_fpr.pdf")
    fig.savefig(args.outdir / "darkmatter_slope_vs_fpr.png", dpi=140); plt.close(fig)

    # ---- fig 3: all three ROC curves overlaid, plus a zoom on the strict corner
    dm, si = R["family"]
    curves = {}
    for u in UPS:
        b = dm[dm.up == u].v.values
        y = np.r_[np.zeros(len(si)), np.ones(len(b))]; sc = np.r_[si, b]
        fpr, tpr, _ = roc_curve(y, sc)
        curves[u] = (fpr, tpr, roc_auc_score(y, sc))
    # sequential shades so the `up` ordering reads off the colour directly;
    # up=0.75 and up=1.00 nearly coincide, so they need real hue separation.
    shade = {0.25: "#999999", 0.75: "#4292c6", 1.0: "#08306b"}
    ZOOM = 0.05

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for u in UPS:
        fpr, tpr, a = curves[u]
        ax.plot(fpr, tpr, color=shade[u], lw=2, label=f"{u:.2f}   (AUROC {a:.3f})")
        m = fpr <= ZOOM
        bx.plot(fpr[m], tpr[m], color=shade[u], lw=2, label=f"{u:.2f}")
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=1)
    bx.plot([0, ZOOM], [0, ZOOM], ls=":", color="gray", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    bx.set_xlim(0, ZOOM); bx.set_ylim(0, np.interp(ZOOM, *curves[1.0][:2]) * 1.08)
    for a_, t in ((ax, "Full range"), (bx, "Strict thresholds")):
        a_.set_xlabel("False positive rate"); a_.set_title(t, fontsize=11, loc="left")
        a_.grid(alpha=.3, ls=":"); a_.set_axisbelow(True)
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=9, title="Unknown percentage", title_fontsize=9,
              loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(args.outdir / "darkmatter_roc_by_up.pdf")
    fig.savefig(args.outdir / "darkmatter_roc_by_up.png", dpi=140); plt.close(fig)
    for u in UPS:
        fpr, tpr, a = curves[u]
        print(f"  up={u}: AUROC={a:.4f}  TPR@1%FPR={np.interp(0.01, fpr, tpr):.4f}")

    print("wrote 3 figures to", args.outdir)


if __name__ == "__main__":
    main()
