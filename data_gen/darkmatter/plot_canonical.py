#!/usr/bin/env python3
"""Dark matter figure from the canonical run (all 180k reads, FPR thresholds)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=Path(".docs/latex/figs/darkmatter_trend.pdf"))
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dm = pd.read_csv("results/darkmatter_canonical_scores.csv")
    si = np.load("results/dm_canon_id.npy")
    T = np.quantile(si, 1 - args.fpr)
    dm["flag"] = (dm.novelty > T).astype(float)
    rep = dm.groupby(["up", "sample"]).flag.mean().reset_index()
    sl, ic, r, p, se = stats.linregress(rep.up, rep.flag)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ax.scatter(rep.up * 100, rep.flag, s=62, color="tab:purple",
               edgecolors="black", linewidths=.6, zorder=3)
    xs = np.linspace(0.20, 1.05, 50)
    ax.plot(xs * 100, ic + sl * xs, ls="--", lw=2.2, color="tab:purple", zorder=2)
    ax.set_xlabel("Unknown percentage (`up`)")
    ax.set_ylabel(f"fraction of reads flagged novel")
    ax.set_title(f"A. at {args.fpr:.0%} FPR: slope = {sl:+.4f}, p = {p:.3f}",
                 fontsize=10.5, loc="left")
    ax.set_xticks([25, 50, 75, 100]); ax.grid(alpha=.3, ls=":"); ax.set_axisbelow(True)

    fp, sls, ps = [], [], []
    for f in np.concatenate([np.arange(0.002, 0.05, 0.002), np.arange(0.05, 0.55, 0.025)]):
        t = np.quantile(si, 1 - f)
        rr = (dm.assign(fl=(dm.novelty > t).astype(float))
                .groupby(["up", "sample"]).fl.mean().reset_index())
        s2, _, _, p2, _ = stats.linregress(rr.up, rr.fl)
        fp.append(f * 100); sls.append(s2); ps.append(p2)
    fp, sls, ps = np.array(fp), np.array(sls), np.array(ps)
    bx.axhline(0, color="black", lw=1)
    bx.plot(fp, sls, color="tab:purple", lw=2)
    sig = ps < 0.05
    bx.scatter(fp[sig], sls[sig], s=30, color="tab:red", zorder=3, label="p < 0.05")
    bx.scatter(fp[~sig], sls[~sig], s=18, facecolors="none", edgecolors="tab:purple", zorder=3,
               label="n.s.")
    bx.axvline(args.fpr * 100, color="tab:gray", ls=":", lw=1.5)
    bx.set_xscale("log")
    bx.set_xlabel("false positive rate on in-distribution reads (%)")
    bx.set_ylabel("slope of flagged fraction vs `up`")
    bx.set_title("B. positive only at strict thresholds", fontsize=10.5, loc="left")
    bx.grid(alpha=.3, ls=":"); bx.set_axisbelow(True); bx.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out); fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=140)
    pos = sig & (sls > 0)
    print(f"panel A: slope={sl:+.5f} se={se:.5f} p={p:.4f} n={len(rep)}")
    print(f"positive & significant for FPR in {fp[pos].min():.1f}%-{fp[pos].max():.1f}%" if pos.any() else "none")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
