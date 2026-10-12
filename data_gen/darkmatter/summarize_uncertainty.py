#!/usr/bin/env python3
"""Summarize uncertainty vs. darkmatter's `up` (percent-unknown) value.

Loads each UQ method's inference_uncertainty.csv (written by
nn_proj.models.<BACKBONE>.inference against data_gen/darkmatter/build_csv.py's
output), tags rows with their method, and reports mean/median/std of the
uncertainty/confidence columns grouped by (method, up). Writes a summary CSV
and a line plot so the trend across `up` is visible directly.

Caveats this script does NOT resolve (see data_gen/darkmatter/build_csv.py's
docstring and the plan this was built from):
  - `up`'s "percent unknown" meaning is inferred from the filename only, not
    documented or confirmed anywhere in this repo.
  - up in {0.25, 0.75, 1.0} have 5 replicates x 4 read files each. up=0.5
    is a single paired_1 file (sample 4 only) -- no 0.0 baseline anywhere,
    and the 0.5 point is not replicate-comparable to the others.
  - darkmatter reads are ~90-150bp; the model was fine-tuned on ~6000bp
    (truncated from 10-11kb PacBio-simulated) reads, a second, confounding
    distribution shift layered on top of whatever `up` represents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = ["max_confidence", "U_total", "U_aleatoric", "U_epistemic", "vote_pct"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--results-dir", type=Path, default=here.parent.parent / "results" / "darkmatter",
                     help="Directory containing <method>/inference_uncertainty.csv subdirs")
    ap.add_argument("--methods", nargs="*", default=["base", "mc_dropout", "epinet"])
    ap.add_argument("--title", default="Darkmatter: uncertainty/confidence vs. `up`",
                     help="Plot suptitle -- set this to match whichever backbone/checkpoint "
                          "actually produced --results-dir's CSVs.")
    args = ap.parse_args()

    frames = []
    for method in args.methods:
        path = args.results_dir / method / "inference_uncertainty.csv"
        if not path.is_file():
            print(f"[warn] missing {path}, skipping {method}")
            continue
        df = pd.read_csv(path)
        df["method"] = method
        frames.append(df)

    if not frames:
        raise SystemExit("No inference_uncertainty.csv files found -- run inference first.")

    all_df = pd.concat(frames, ignore_index=True)

    summary = (
        all_df.groupby(["method", "up"])[METRICS]
        .agg(["mean", "median", "std", "count"])
    )
    summary_path = args.results_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"Wrote {summary_path}\n")
    print(summary.to_string())

    # Also break out by sample (replicate) so within-`up` spread is visible,
    # not just the pooled mean/median across all 5 samples' reads.
    per_sample = (
        all_df.groupby(["method", "up", "sample"])[METRICS]
        .agg(["mean", "median"])
    )
    per_sample_path = args.results_dir / "summary_per_sample.csv"
    per_sample.to_csv(per_sample_path)
    print(f"\nWrote {per_sample_path}")

    # The 2026-09-14 file additions brought in paired_1/paired_2/unmatched_1
    # reads alongside the original unmatched_2 ones. The two summaries above
    # pool them (and stay the shape plot_epinet_scatter.py expects); this one
    # keeps them apart, so a read_type effect can't masquerade as an `up`
    # effect. Skipped on older result sets that have no read_type column.
    if "read_type" in all_df.columns:
        by_read_type = (
            all_df.groupby(["method", "up", "read_type", "mate"])[METRICS]
            .agg(["mean", "median", "std", "count"])
        )
        by_read_type_path = args.results_dir / "summary_by_read_type.csv"
        by_read_type.to_csv(by_read_type_path)
        print(f"\nWrote {by_read_type_path}")
        print(by_read_type.to_string())

        # One row per input FASTQ file (up x sample x read_type x mate). This is
        # the finest grouping that is still a real experimental unit, and it is
        # what plot_epinet_scatter.py --per-file plots: 61 points per method
        # instead of the 16 replicate-level points summary_per_sample.csv gives.
        per_file = (
            all_df.groupby(["method", "up", "sample", "read_type", "mate"])[METRICS]
            .agg(["mean", "median", "std", "count"])
        )
        per_file_path = args.results_dir / "summary_per_file.csv"
        per_file.to_csv(per_file_path)
        print(f"\nWrote {per_file_path} ({len(per_file)} rows)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plot")
        return

    plot_metrics = ["max_confidence", "U_total", "U_aleatoric", "U_epistemic"]
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 4), sharex=True)
    for ax, metric in zip(axes, plot_metrics):
        for method in args.methods:
            if method not in all_df["method"].unique():
                continue
            sub = all_df[all_df["method"] == method]
            stats = sub.groupby("up")[metric].agg(["mean", "median"]).sort_index()
            ax.plot(stats.index, stats["mean"], marker="o", label=f"{method} (mean)")
            ax.plot(stats.index, stats["median"], marker="x", linestyle="--",
                    label=f"{method} (median)")
        ax.set_xlabel("up (inferred % unknown)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=7)

    fig.suptitle(args.title)
    fig.tight_layout()
    plot_path = args.results_dir / "uncertainty_vs_up.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
