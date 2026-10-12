#!/usr/bin/env python3
"""Scatter figure: darkmatter `up` (inferred % unknown) vs. epinet U_epistemic.

One point per (up, sample) replicate (mean over that replicate's ~3000 reads),
epinet and mc_dropout overlaid for contrast, with a per-method linear trend
line. Built for the weekly-update writeup (.docs/latex/main.tex-style docs),
output alongside the other figures in .docs/latex/figs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-dir", type=Path, default=here.parent.parent / "results" / "darkmatter_nt")
    ap.add_argument("--title", default="Darkmatter: epistemic uncertainty vs. % unknown\n"
                                       "(NT_transformer, PBsim family, seed 1)")
    ap.add_argument("--out", type=Path,
                     default=here.parent.parent / ".docs" / "latex" / "figs" / "darkmatter_confidence_vs_up.pdf")
    ap.add_argument("--metric", default="U_epistemic")
    ap.add_argument("--methods", nargs="*", default=["epinet", "mc_dropout"],
                     help="Methods to plot. The default pair keeps the twin-axis layout "
                          "(their U_epistemic scales differ ~5x). Any other number of "
                          "methods -- e.g. adding `base`, which is meaningful for U_total "
                          "and max_confidence but has U_epistemic==0 by construction -- "
                          "switches to one shared axis.")
    ap.add_argument("--per-file", action="store_true",
                     help="Plot one point per input FASTQ file (up x sample x read_type x "
                          "mate, 61 points/method) instead of one per replicate (16 "
                          "points/method). Reads summary_per_file.csv.")
    args = ap.parse_args()

    if args.per_file:
        per_sample = pd.read_csv(args.results_dir / "summary_per_file.csv",
                                  header=[0, 1], index_col=[0, 1, 2, 3, 4])
        per_sample.index.names = ["method", "up", "sample", "read_type", "mate"]
    else:
        per_sample = pd.read_csv(args.results_dir / "summary_per_sample.csv",
                                  header=[0, 1], index_col=[0, 1, 2])
        per_sample.index.names = ["method", "up", "sample"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SPREAD = 14.0   # x-axis width each `up` group's points are spread across (--per-file)
    up_ticks = sorted(per_sample.index.get_level_values("up").unique() * 100)
    x_lo, x_hi = min(up_ticks) - 10, max(up_ticks) + 10

    colors = {"epinet": "tab:purple", "mc_dropout": "tab:green", "base": "tab:orange"}
    markers = {"epinet": "o", "mc_dropout": "^", "base": "s"}
    # Nudge each method's points apart horizontally so same-up markers don't overlap.
    x_offset = {"epinet": -0.015, "mc_dropout": 0.015, "base": 0.0}

    methods = [m for m in args.methods
               if m in per_sample.index.get_level_values("method").unique()]
    missing = [m for m in args.methods if m not in methods]
    if missing:
        print(f"[warn] not in {args.results_dir}, skipping: {', '.join(missing)}")
    if not methods:
        raise SystemExit("none of the requested methods are present")

    # Twin axes only make sense for exactly two series; with one or three+ the
    # comparison has to share a scale (which is honest here anyway -- U_total and
    # max_confidence are on the same footing across methods, unlike U_epistemic).
    twin = (len(methods) == 2)
    fig, ax_left = plt.subplots(figsize=(5.5, 4))
    if twin:
        axes = {methods[0]: ax_left, methods[1]: ax_left.twinx()}
    else:
        axes = {m: ax_left for m in methods}

    handles = []
    for method in methods:
        ax = axes[method]
        sub = per_sample.loc[method][(args.metric, "mean")].reset_index()
        if args.per_file:
            sub.columns = ["up", "sample", "read_type", "mate", "value"]
        else:
            sub.columns = ["up", "sample", "value"]
        sub["up_pct"] = sub["up"] * 100

        if args.per_file:
            # Up to 20 files share each `up` (5 replicates x 4 read files), so
            # spread them evenly across a fixed window rather than jittering by
            # replicate index alone -- otherwise the four files of one replicate
            # land on top of each other.
            sub = sub.sort_values(["up_pct", "sample", "read_type", "mate"])
            jitter = sub.groupby("up_pct").cumcount()
            group_n = sub.groupby("up_pct")["value"].transform("size")
            jitter = (jitter - (group_n - 1) / 2) * (SPREAD / group_n.clip(lower=2))
        else:
            jitter = (sub["sample"] - 2) * 0.6  # spread the 5 replicates apart horizontally

        h = ax.scatter(sub["up_pct"] + x_offset[method] + jitter, sub["value"], color=colors[method],
                       marker=markers[method], s=40 if not args.per_file else 22,
                       alpha=0.8 if not args.per_file else 0.65, label=method, zorder=3)
        handles.append(h)

        # Per-method linear trend. With --per-file the fit runs over every plotted
        # point, so the single up=0.5 file carries the weight of one file rather
        # than of a whole `up` level; otherwise fit the per-`up` means as before.
        if args.per_file:
            coef = np.polyfit(sub["up_pct"], sub["value"], 1)
        else:
            means = sub.groupby("up_pct")["value"].mean()
            coef = np.polyfit(means.index, means.values, 1)
        xs = np.linspace(x_lo, x_hi, 50)
        ax.plot(xs, np.polyval(coef, xs), color=colors[method], linestyle="--", linewidth=1, zorder=2)

        # Label from the metric itself -- "Uncertainty" is wrong for
        # max_confidence, which moves in the opposite direction.
        if twin:
            ax.set_ylabel(f"{args.metric} ({method})", color=colors[method])
            ax.tick_params(axis="y", labelcolor=colors[method])
        else:
            ax.set_ylabel(args.metric)

    ax_left.set_xlabel("Unknown Percentage")
    # Two-line titles overflow the 5.5in canvas at the default size; matplotlib
    # will not wrap, so scale the title down instead of clipping it.
    ax_left.set_title(args.title, fontsize=9)
    ax_left.set_xticks(up_ticks)
    ax_left.set_xlim(x_lo, x_hi)
    for a in {id(v): v for v in axes.values()}.values():
        a.set_axisbelow(True)
    ax_left.grid(True, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, zorder=0)
    ax_left.legend(handles=handles, labels=[h.get_label() for h in handles], loc="upper left")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
