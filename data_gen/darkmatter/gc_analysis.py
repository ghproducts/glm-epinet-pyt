#!/usr/bin/env python3
"""GC-confound analysis for the darkmatter `up` (fraction-unknown) experiment.

The headline `up` effect at phylum rank -- model confidence RISES as the
fraction of unknown organisms rises -- is confounded: GC content also rises
with `up` (the novel taxa in this simulation are higher-GC), and model
confidence is strongly GC-dependent. Stratifying by GC removes the `up`
effect. This script produces the four-panel figure and the tables behind that
conclusion.

Run after scripts/run_darkmatter_nt_rank.sh phylum and the phylum-rank
truncation control arms have populated results/.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["base", "mc_dropout", "epinet"]
COLORS = {"base": "tab:orange", "epinet": "tab:purple", "mc_dropout": "tab:green"}
MARKERS = {"base": "s", "epinet": "o", "mc_dropout": "^"}
GC_BINS = [0, .40, .44, .48, .52, .56, 1.0]


def gc_of(seq: pd.Series) -> pd.Series:
    return (seq.str.count("G") + seq.str.count("C")) / seq.str.len()


def load(results_dir: Path, reads_csv: Path):
    reads = pd.read_csv(reads_csv)
    gc = gc_of(reads.sequence)
    out = {}
    for m in METHODS:
        d = pd.read_csv(results_dir / m / "inference_uncertainty.csv")
        d["gc"] = gc.values
        out[m] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    ap.add_argument("--results-dir", type=Path, default=root / "results" / "darkmatter_nt_phylum")
    ap.add_argument("--reads", type=Path,
                    default=here / "csv_data" / "darkmatter_reads_all.csv")
    ap.add_argument("--id-dir", type=Path,
                    default=root / "results" / "truncation_control_phylum" / "150bp")
    ap.add_argument("--id-reads", type=Path,
                    default=root / "data_gen" / "pbsim" / "truncation_control" / "id_novel_genus_150bp.csv")
    ap.add_argument("--outdir", type=Path, default=root / ".docs" / "latex" / "figs")
    ap.add_argument("--png-dir", type=Path, default=None)
    args = ap.parse_args()

    data = load(args.results_dir, args.reads)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axA, axB, axC, axD = axes.ravel()

    # ---- A: the confound itself -- GC rises with `up` -------------------
    b = data["base"]
    per_file = b.groupby(["up", "sample", "read_type", "mate"]).gc.mean().reset_index()
    jit = (per_file.groupby("up").cumcount() - 9.5) * 0.55
    axA.scatter(per_file.up * 100 + jit, per_file.gc, s=18, alpha=.7, color="tab:red")
    lv = b.groupby("up").gc.mean()
    axA.plot(lv.index * 100, lv.values, color="black", lw=1.5, marker="D", ms=5, zorder=5,
             label="mean per `up`")
    axA.set_xlabel("Unknown percentage (`up`)"); axA.set_ylabel("GC content")
    axA.set_title("A. The confound: GC rises with unknown fraction", fontsize=10, loc="left")
    axA.legend(fontsize=8); axA.grid(alpha=.3, ls=":")

    # ---- B: what confidence actually tracks -- GC ----------------------
    for m in METHODS:
        d = data[m]
        d["gcbin"] = pd.cut(d.gc, GC_BINS)
        g = d.groupby("gcbin", observed=True).max_confidence.agg(["mean", "sem"])
        x = [iv.mid for iv in g.index]
        axB.errorbar(x, g["mean"], yerr=g["sem"] * 1.96, color=COLORS[m], marker=MARKERS[m],
                     ms=5, lw=1.5, capsize=3, label=m)
    axB.set_xlabel("GC content"); axB.set_ylabel("max_confidence")
    axB.set_title("B. Confidence tracks GC strongly (0.54 → 0.73)", fontsize=10, loc="left")
    axB.legend(fontsize=8); axB.grid(alpha=.3, ls=":")

    # ---- C: the apparent `up` effect (uncorrected) ---------------------
    for m in METHODS:
        d = data[m]
        pf = d.groupby(["up", "sample", "read_type", "mate"]).max_confidence.mean().reset_index()
        jit = (pf.groupby("up").cumcount() - 9.5) * 0.55
        axC.scatter(pf.up * 100 + jit, pf.max_confidence, s=14, alpha=.55,
                    color=COLORS[m], marker=MARKERS[m], label=m)
        rep = d[d.up != .5].groupby(["up", "sample"]).max_confidence.mean().reset_index()
        sl, ic, *_ = stats.linregress(rep.up, rep.max_confidence)
        xs = np.linspace(.15, 1.10, 20)
        axC.plot(xs * 100, ic + sl * xs, color=COLORS[m], ls="--", lw=1)
    axC.set_xlabel("Unknown percentage (`up`)"); axC.set_ylabel("max_confidence")
    axC.set_title("C. Uncorrected: confidence RISES with unknown fraction", fontsize=10, loc="left")
    axC.legend(fontsize=8); axC.grid(alpha=.3, ls=":")

    # ---- D: stratify by GC and the effect disappears -------------------
    d = data["base"].copy()
    d["gcbin"] = pd.cut(d.gc, GC_BINS)
    cmap = plt.get_cmap("viridis")
    bins_used = [gb for gb, g in d.groupby("gcbin", observed=True) if len(g) > 2000]
    for i, gb in enumerate(bins_used):
        g = d[d.gcbin == gb]
        lv = g.groupby("up").max_confidence.mean()
        axD.plot(lv.index * 100, lv.values, marker="o", ms=4, lw=1.4,
                 color=cmap(i / max(1, len(bins_used) - 1)),
                 label=f"GC {gb.left:.2f}-{gb.right:.2f}")
    axD.set_xlabel("Unknown percentage (`up`)"); axD.set_ylabel("max_confidence (base)")
    axD.set_title("D. Within GC bands the `up` effect vanishes", fontsize=10, loc="left")
    axD.legend(fontsize=7, ncol=2); axD.grid(alpha=.3, ls=":")

    fig.suptitle("Darkmatter at phylum rank: uncertainty tracks GC content, not taxonomic novelty\n"
                 "(NT_transformer, PBsim phylum checkpoint, 36 classes, seed 1, 183k reads)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outdir / "darkmatter_gc_confound.pdf")
    print(f"Wrote {args.outdir / 'darkmatter_gc_confound.pdf'}")
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png_dir / "darkmatter_gc_confound.png", dpi=140)
        print(f"Wrote {args.png_dir / 'darkmatter_gc_confound.png'}")


if __name__ == "__main__":
    main()
