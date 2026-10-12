#!/usr/bin/env python3
"""Darkmatter: flagged fraction vs `up`, and where in the score distribution
that trend actually holds.

Panel A is the trend itself at one threshold. Panel B exists because the trend
is threshold-dependent -- positive only among the most-novel-scoring reads and
flat-to-negative below that -- so reporting a single cut without showing the
sweep would be selective.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/darkmatter_clean_scores.csv"))
    ap.add_argument("--keep-frac", type=float, default=0.15,
                    help="flag the top this-fraction of reads by novelty (panel A)")
    ap.add_argument("--out", type=Path, default=Path(".docs/latex/figs/darkmatter_trend.pdf"))
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(args.scores)
    fit = d[d.up != 0.5]          # up=0.5 is one file, no replicates

    thr = np.quantile(d.novelty, 1 - args.keep_frac)
    d["flag"] = (d.novelty > thr).astype(float)
    # one point per input FASTQ file (up x sample x read_type x mate) = 61 points
    rep = d.groupby(["up", "sample", "read_type", "mate"]).flag.mean().reset_index()
    rep_fit = rep[rep.up != 0.5]
    sl, ic, r, p, se = stats.linregress(rep_fit.up, rep_fit.flag)
    # the four files of a replicate share a simulated community, so a per-file
    # fit is pseudo-replicated; the replicate-level fit is the conservative test
    repl = d[d.up != 0.5].groupby(["up", "sample"]).flag.mean().reset_index()
    sl_r, _, r_r, p_r, se_r = stats.linregress(repl.up, repl.flag)
    print(f"per-file   n={len(rep_fit)} slope={sl:+.5f} se={se:.5f} p={p:.4f}")
    print(f"replicate  n={len(repl)} slope={sl_r:+.5f} se={se_r:.5f} p={p_r:.4f}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # --- A: the trend -------------------------------------------------
    # spread the ~20 files sharing each `up` so they do not overplot
    # per-file points: each holds only ~250 reads, so its flagged fraction has a
    # binomial SE (~0.023) larger than the effect itself. Replicate means are
    # overlaid because the simulated community, not the file, is the unit the
    # regression is fit on.
    jit = (rep_fit.groupby("up").cumcount() - 9.5) * 0.55
    ax.scatter(rep_fit.up * 100 + jit, rep_fit.flag, s=20, color="tab:purple",
               alpha=.42, zorder=3, label="one point per file (n=60)")
    ax.scatter(repl.up * 100, repl.flag, s=70, color="tab:purple",
               edgecolors="black", linewidths=.7, zorder=4,
               label="replicate mean (n=15, fit)")
    solo = rep[rep.up == 0.5]
    if len(solo):
        ax.scatter(solo.up * 100, solo.flag, s=46, color="tab:gray", marker="^",
                   alpha=.85, zorder=3, label="up=50 (single file, excluded)")
    xs = np.linspace(0.20, 1.05, 50)
    ic_r = repl.flag.mean() - sl_r * repl.up.mean()
    ax.plot(xs * 100, ic_r + sl_r * xs, ls="--", lw=2.2, color="tab:purple", zorder=2)
    ax.set_xlabel("Unknown percentage (`up`)")
    ax.set_ylabel(f"fraction flagged novel (top {args.keep_frac:.0%})")
    ax.set_title(f"A. slope = {sl_r:+.4f}, p = {p_r:.3f} (fit on replicate means)",
                 fontsize=10.5, loc="left")
    ax.set_xticks([25, 50, 75, 100]); ax.grid(alpha=.3, ls=":"); ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left")

    # --- B: where the trend holds --------------------------------------
    fr, sls, ps = [], [], []
    for q in np.arange(0.02, 0.62, 0.02):
        t = np.quantile(d.novelty, 1 - q)
        f = fit.assign(flag=(fit.novelty > t).astype(float))
        rr = f.groupby(["up", "sample"]).flag.mean().reset_index()   # replicate level
        s2, _, _, p2, _ = stats.linregress(rr.up, rr.flag)
        fr.append(q * 100); sls.append(s2); ps.append(p2)
    fr, sls, ps = np.array(fr), np.array(sls), np.array(ps)
    bx.axhline(0, color="black", lw=1)
    bx.plot(fr, sls, color="tab:purple", lw=2)
    sig = ps < 0.05
    bx.scatter(fr[sig], sls[sig], s=34, color="tab:red", zorder=3, label="p < 0.05")
    bx.scatter(fr[~sig], sls[~sig], s=20, facecolors="none", edgecolors="tab:purple",
               zorder=3, label="n.s.")
    bx.axvline(args.keep_frac * 100, color="tab:gray", ls=":", lw=1.5)
    bx.set_xlabel("fraction of reads flagged (threshold)")
    bx.set_ylabel("slope of flagged fraction vs `up`")
    bx.set_title("B. trend is confined to the most-novel reads", fontsize=11, loc="left")
    bx.grid(alpha=.3, ls=":"); bx.set_axisbelow(True); bx.legend(fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=140)
    print(f"panel A: threshold={thr:.4f} slope={sl:+.5f} se={se:.5f} r={r:+.3f} p={p:.4f} n={len(rep_fit)}")
    print(f"positive & significant over flagged-fraction range: "
          f"{fr[sig & (sls>0)].min():.0f}%-{fr[sig & (sls>0)].max():.0f}%" if (sig & (sls>0)).any() else "none")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
