"""Manuscript figures, generated from the aggregated tables.

Every figure here reads ``tables/*.csv`` produced by
``nn_proj.analysis.aggregate``. No values are hard-coded: changing a seed set
or rerunning inference changes the figures without any edit here.

Figure map
----------
    ece_vs_error     ECE against classification error, one panel per backbone,
                     split into ID and OOD rows.               (Figs 4, 7)
    reliability      Reliability diagrams comparing methods.    (Figs 5, 8)
    auroc_heatmap    AUROC deltas against the base model.       (Figs 6, 9)
    risk_coverage    Selective-prediction curves.               (new)
    ece_sensitivity  ECE against bin count and binning scheme.  (new)

Usage
-----
    python -m nn_proj.analysis.figures tables/ -o plots/ --results <results_root>

``--results`` is only needed for the reliability and risk-coverage figures,
which read per-example predictions rather than summary rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from .metrics import confidence_oracle_curve, reliability_bins
from .results import ResultsIndex, RunKey
from .tasks import load_registry

# Display names and a stable plotting order for the uncertainty methods.
METHOD_ORDER = ["base", "base_scaled", "mc_dropout", "conv_epinet"]
METHOD_LABEL = {
    "base": "Base",
    "base_scaled": "Temp Scaling",
    "mc_dropout": "MC Dropout",
    "conv_epinet": "Epinet",
}
METHOD_MARKER = {"base": "o", "base_scaled": "P", "mc_dropout": "D", "conv_epinet": "*"}
METHOD_COLOR = {
    "base": "#1f77b4",
    "base_scaled": "#ff7f0e",
    "mc_dropout": "#2ca02c",
    "conv_epinet": "#d62728",
}
BACKBONE_ORDER = ["NT_transformer", "DNABERT2", "hyenaDNA", "CARMANIA"]
BACKBONE_LABEL = {"NT_transformer": "NT", "DNABERT2": "DNABERT2",
                  "hyenaDNA": "HyenaDNA", "CARMANIA": "CARMANIA"}

# Which categories count as "shifted" when splitting a figure into ID/OOD rows.
SHIFTED = ("Near-OOD", "OOD")


def _order(values: Sequence[str], reference: Sequence[str]) -> List[str]:
    """Sort ``values`` by ``reference``, appending anything unlisted."""
    known = [v for v in reference if v in values]
    return known + sorted(set(values) - set(reference))


def _save(fig, outpath: Optional[Path]) -> None:
    if outpath is None:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# --------------------------------------------------------------------------
# ECE vs error scatter (Figs 4 and 7)
# --------------------------------------------------------------------------

def ece_vs_error(
    summary: pd.DataFrame,
    trains: Sequence[str],
    outpath: Optional[Path] = None,
    title: str = "",
    calibration_only: bool = True,
    figsize_per_panel: tuple = (3.6, 3.4),
):
    """ECE against classification error, one column per backbone.

    Rows split matched evaluations from shifted ones. Each point is a
    (task, method) cell; a line joins every method back to its Base model so
    the direction of the change is readable.

    ``calibration_only`` drops pairs the registry marks as not supporting
    calibration — those that change the prediction target, where ECE is not
    interpretable. The manuscript's originals included them.
    """
    df = summary[summary.train.isin(trains)].copy()
    if calibration_only:
        df = df[df.supports_calibration]
    if df.empty:
        print(f"  [skip] no calibration-supporting rows for {list(trains)}")
        return None

    df["pair"] = df.train + "→" + df.test
    df["shifted"] = df.category.isin(SHIFTED)

    backbones = _order(df.backbone.unique(), BACKBONE_ORDER)
    rows = [False, True] if df.shifted.any() and (~df.shifted).any() else [df.shifted.iloc[0]]
    pairs = sorted(df.pair.unique())
    pair_color = {p: plt.get_cmap("tab10")(i % 10) for i, p in enumerate(pairs)}

    fig, axes = plt.subplots(
        len(rows), len(backbones), squeeze=False,
        figsize=(figsize_per_panel[0] * len(backbones), figsize_per_panel[1] * len(rows)),
    )

    for r, shifted in enumerate(rows):
        for c, backbone in enumerate(backbones):
            ax = axes[r][c]
            cell = df[(df.backbone == backbone) & (df.shifted == shifted)]
            for pair, grp in cell.groupby("pair"):
                grp = grp.set_index("method")
                if "base" in grp.index:
                    b = grp.loc["base"]
                    for method in grp.index:
                        if method == "base":
                            continue
                        m = grp.loc[method]
                        ax.plot([b.error_mean * 100, m.error_mean * 100],
                                [b.ece_mean * 100, m.ece_mean * 100],
                                color=pair_color[pair], lw=1.6, alpha=0.7, zorder=1)
                for method, row in grp.iterrows():
                    ax.scatter(row.error_mean * 100, row.ece_mean * 100,
                               marker=METHOD_MARKER.get(method, "o"), s=70,
                               color=pair_color[pair], edgecolor="k", lw=0.8, zorder=2)
            ax.set_title(f"{BACKBONE_LABEL.get(backbone, backbone)} — "
                         f"{'OOD' if shifted else 'ID'}", fontsize=11)
            ax.grid(True, ls="--", alpha=0.4)
            if c == 0:
                ax.set_ylabel("ECE [%]")
            if r == len(rows) - 1:
                ax.set_xlabel("Classification Error [%]")

    handles = [Line2D([], [], marker=METHOD_MARKER.get(m, "o"), ls="", color="k",
                      markerfacecolor="w", markersize=9, label=METHOD_LABEL.get(m, m))
               for m in _order(df.method.unique(), METHOD_ORDER)]
    handles += [Line2D([], [], marker="s", ls="", color=pair_color[p], markersize=9, label=p)
                for p in pairs]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
               frameon=True, bbox_to_anchor=(0.5, -0.10), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, outpath)
    return fig


# --------------------------------------------------------------------------
# Reliability diagrams (Figs 5 and 8)
# --------------------------------------------------------------------------

def reliability(
    index: ResultsIndex,
    backbone: str,
    train: str,
    test: str,
    methods: Sequence[str] = tuple(METHOD_ORDER),
    n_bins_visual: int = 20,
    min_count: int = 5,
    ax=None,
    outpath: Optional[Path] = None,
    show_ece: bool = True,
    ece_summary: Optional[pd.DataFrame] = None,
):
    """Reliability diagram pooling every available seed.

    Each method's curve is the per-seed mean accuracy in a fixed grid of
    equal-width confidence bins, with a shaded band at +/- one standard
    deviation across seeds. Bins holding fewer than ``min_count`` examples in a
    seed do not contribute for that seed.

    Legend ECE values are the seeded mean from ``ece_summary`` when supplied,
    so the diagram and the tables cannot disagree.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], "--", color="k", lw=1.8, zorder=1)

    for method in methods:
        keys = index.select(backbone=backbone, method=method, train=train, test=test)
        if not keys:
            continue

        per_seed = []
        for key in keys:
            bins = reliability_bins(index.frame(key), n_bins=n_bins_visual,
                                    binning="equal_width", min_count=min_count)
            per_seed.append(bins.set_index(bins.index)[["confidence", "accuracy"]])

        acc = pd.concat([s["accuracy"] for s in per_seed], axis=1)
        conf = pd.concat([s["confidence"] for s in per_seed], axis=1)
        mean_acc, std_acc = acc.mean(axis=1), acc.std(axis=1)
        mean_conf = conf.mean(axis=1)
        ok = mean_acc.notna() & mean_conf.notna()
        if not ok.any():
            continue

        label = METHOD_LABEL.get(method, method)
        if show_ece:
            ece = _lookup_ece(ece_summary, backbone, method, train, test, len(keys))
            if ece is not None:
                label = f"{label}  ECE {ece}"

        color = METHOD_COLOR.get(method)
        ax.plot(mean_conf[ok], mean_acc[ok], lw=2.0, color=color, label=label, zorder=3)
        ax.fill_between(mean_conf[ok], (mean_acc - std_acc)[ok], (mean_acc + std_acc)[ok],
                        color=color, alpha=0.18, lw=0, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_title(f"{BACKBONE_LABEL.get(backbone, backbone)}\n{train}→{test}", fontsize=10)

    if own_fig:
        fig.tight_layout()
        _save(fig, outpath)
    return fig, ax


def _lookup_ece(summary, backbone, method, train, test, n_keys) -> Optional[str]:
    """Formatted 'mean +/- std' ECE for a cell, or None when unavailable."""
    if summary is None:
        return None
    row = summary[(summary.backbone == backbone) & (summary.method == method)
                  & (summary.train == train) & (summary.test == test)]
    if row.empty:
        return None
    mean = row.ece_mean.iloc[0] * 100
    std = row.ece_std.iloc[0] * 100
    n = int(row.n_seeds.iloc[0])
    if np.isnan(std):
        return f"{mean:.1f}% (n={n})"
    return f"{mean:.1f}±{std:.1f}% (n={n})"


def reliability_grid(
    index: ResultsIndex,
    pairs: Sequence[tuple],
    backbones: Sequence[str],
    ece_summary: Optional[pd.DataFrame] = None,
    outpath: Optional[Path] = None,
    **kwargs,
):
    """Grid of reliability diagrams: one row per backbone, one column per pair."""
    fig, axes = plt.subplots(len(backbones), len(pairs), squeeze=False,
                             figsize=(4.2 * len(pairs), 4.2 * len(backbones)))
    for r, backbone in enumerate(backbones):
        for c, (train, test) in enumerate(pairs):
            reliability(index, backbone, train, test, ax=axes[r][c],
                        ece_summary=ece_summary, **kwargs)
    fig.tight_layout()
    _save(fig, outpath)
    return fig


# --------------------------------------------------------------------------
# AUROC delta heatmaps (Figs 6 and 9)
# --------------------------------------------------------------------------

def auroc_heatmap(
    ood_summary: pd.DataFrame,
    backbone: str,
    trains: Optional[Sequence[str]] = None,
    outpath: Optional[Path] = None,
    vlim: Optional[float] = None,
):
    """AUROC change against the base model, per ID/OOD pairing and score.

    Rows are ID->OOD evaluation pairs; columns group (method, score). The
    rightmost annotation carries the absolute base AUROC, without which a
    delta cannot be read: +0.08 over a chance-level 0.48 baseline is a very
    different claim from +0.08 over 0.90.
    """
    df = ood_summary[ood_summary.backbone == backbone].copy()
    if trains is not None:
        df = df[df.train.isin(trains)]
    if df.empty:
        print(f"  [skip] no OOD rows for {backbone}")
        return None

    df = df[~((df.method == "base") & (df.score == "U_total"))]
    df["col"] = df.method + "\n" + df.score.str.replace("U_", "", regex=False)
    df["row"] = "ID: " + df.id_test + "\nOOD: " + df.ood_test

    pivot = df.pivot_table(index="row", columns="col", values="delta_auroc_mean")
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    pivot = pivot[sorted(pivot.columns, key=lambda c: (method_rank.get(c.split("\n")[0], 99), c))]

    base = (ood_summary[(ood_summary.backbone == backbone)
                        & (ood_summary.method == "base")
                        & (ood_summary.score == "U_total")]
            .assign(row=lambda d: "ID: " + d.id_test + "\nOOD: " + d.ood_test)
            .set_index("row")["auroc_mean"])

    v = vlim if vlim is not None else float(np.nanmax(np.abs(pivot.to_numpy())))
    v = max(v, 1e-3)

    fig, ax = plt.subplots(figsize=(1.15 * len(pivot.columns) + 3.5,
                                    0.85 * len(pivot.index) + 2.0))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis",
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v), aspect="auto")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > v * 0.6 else "black")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i, row in enumerate(pivot.index):
        b = base.get(row, np.nan)
        ax.text(len(pivot.columns) - 0.35, i, f"  {b:.3f}" if not np.isnan(b) else "  --",
                ha="left", va="center", fontsize=9, fontweight="bold")
    ax.text(len(pivot.columns) - 0.35, -0.75, "  Base\n  AUROC",
            ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_title(f"{BACKBONE_LABEL.get(backbone, backbone)} OOD detection "
                 f"(ΔAUROC vs Base)", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.12, label="ΔAUROC")
    fig.tight_layout()
    _save(fig, outpath)
    return fig


# --------------------------------------------------------------------------
# Risk-coverage curves
# --------------------------------------------------------------------------

def risk_coverage(
    index: ResultsIndex,
    backbone: str,
    train: str,
    test: str,
    methods: Sequence[str] = tuple(METHOD_ORDER),
    score_col: str = "max_confidence",
    ax=None,
    outpath: Optional[Path] = None,
):
    """Selective-prediction curves: error on the retained set against coverage.

    This is the direct test of whether uncertainty is *adaptive*. A method that
    reduces ECE by shrinking every prediction equally leaves this curve on top
    of the Base curve; a method that knows which specific examples are
    unreliable pulls it down toward the oracle.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
    else:
        fig = ax.figure

    oracle_drawn = False
    for method in methods:
        keys = index.select(backbone=backbone, method=method, train=train, test=test)
        if not keys:
            continue

        grid = np.linspace(0.02, 1.0, 100)
        curves, oracles = [], []
        for key in keys:
            df = index.frame(key)
            err = 1.0 - df["correct"].mean()
            if not (0.0 < err < 1.0):
                continue
            curve, _, _ = confidence_oracle_curve(df, score_col=score_col)
            curves.append(np.interp(grid, curve.coverage, curve.confidence_error))
            oracles.append(np.interp(grid, curve.coverage, curve.oracle_error))
        if not curves:
            continue

        arr = np.vstack(curves)
        mean, std = arr.mean(0), arr.std(0)
        color = METHOD_COLOR.get(method)
        ax.plot(grid, mean, lw=2.0, color=color, label=METHOD_LABEL.get(method, method))
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, lw=0)

        if not oracle_drawn:
            ax.plot(grid, np.vstack(oracles).mean(0), ls="--", lw=1.8,
                    color="k", label="Oracle")
            oracle_drawn = True

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("Error on retained set")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title(f"{BACKBONE_LABEL.get(backbone, backbone)}\n{train}→{test}", fontsize=10)

    if own_fig:
        fig.tight_layout()
        _save(fig, outpath)
    return fig, ax


# --------------------------------------------------------------------------
# ECE bin sensitivity
# --------------------------------------------------------------------------

def ece_sensitivity(
    sensitivity: pd.DataFrame,
    backbone: str,
    train: str,
    test: str,
    outpath: Optional[Path] = None,
):
    """ECE against bin count, for both binning schemes.

    Shows whether the ordering between methods is stable, or an artefact of the
    M = 50 equal-mass choice used in the reported numbers.
    """
    df = sensitivity[(sensitivity.backbone == backbone) & (sensitivity.train == train)
                     & (sensitivity.test == test)]
    if df.empty:
        return None

    schemes = sorted(df.binning.unique())
    fig, axes = plt.subplots(1, len(schemes), squeeze=False, figsize=(4.4 * len(schemes), 3.8))
    for j, scheme in enumerate(schemes):
        ax = axes[0][j]
        for method in _order(df.method.unique(), METHOD_ORDER):
            sub = df[(df.binning == scheme) & (df.method == method)].sort_values("n_bins")
            if sub.empty:
                continue
            ax.plot(sub.n_bins, sub.ece * 100, marker="o", lw=1.8,
                    color=METHOD_COLOR.get(method), label=METHOD_LABEL.get(method, method))
        ax.axvline(50, color="k", ls=":", lw=1.2, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Number of bins")
        ax.set_title(scheme.replace("_", " "), fontsize=10)
        ax.grid(True, ls="--", alpha=0.4)
        if j == 0:
            ax.set_ylabel("ECE [%]")
            ax.legend(fontsize=8)
    fig.suptitle(f"{BACKBONE_LABEL.get(backbone, backbone)}  {train}→{test}"
                 f"   (dotted line = reported M=50)", fontsize=11)
    fig.tight_layout()
    _save(fig, outpath)
    return fig


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", help="Directory of CSVs from nn_proj.analysis.aggregate")
    ap.add_argument("-o", "--outdir", default="plots")
    ap.add_argument("--results", default=None,
                    help="Results root, needed for reliability and risk-coverage figures")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of figures to build: scatter reliability heatmap risk sensitivity")
    args = ap.parse_args()

    tables, out = Path(args.tables), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else {"scatter", "reliability", "heatmap",
                                               "risk", "sensitivity"}

    cal = pd.read_csv(tables / "calibration_summary.csv")
    ood = pd.read_csv(tables / "ood_detection_summary.csv")
    reg = load_registry()

    regulatory = [t for t in reg.train_tasks if not t.startswith(("pbsim", "gene_taxa"))]
    metagenomic = [t for t in reg.train_tasks if t.startswith(("pbsim", "gene_taxa"))]

    if "scatter" in wanted:
        print("ECE vs error scatters:")
        ece_vs_error(cal, regulatory, out / "regulatory_scatter.pdf",
                     title="Regulatory classification")
        ece_vs_error(cal, metagenomic, out / "metagenomic_scatter.pdf",
                     title="Metagenomic classification")

    if "heatmap" in wanted:
        print("AUROC heatmaps:")
        for backbone in _order(ood.backbone.unique(), BACKBONE_ORDER):
            auroc_heatmap(ood, backbone, regulatory,
                          out / f"auroc_regulatory_{backbone}.pdf")
            auroc_heatmap(ood, backbone, metagenomic,
                          out / f"auroc_metagenomic_{backbone}.pdf")

    if "sensitivity" in wanted and (tables / "ece_bin_sensitivity.csv").exists():
        print("ECE bin sensitivity:")
        sens = pd.read_csv(tables / "ece_bin_sensitivity.csv")
        for backbone in _order(sens.backbone.unique(), BACKBONE_ORDER):
            ece_sensitivity(sens, backbone, "promoter_all", "promoter_all",
                            out / f"ece_sensitivity_{backbone}.pdf")

    if args.results and wanted & {"reliability", "risk"}:
        index = ResultsIndex(args.results)
        backbones = _order(index.backbones, BACKBONE_ORDER)

        if "reliability" in wanted:
            print("Reliability diagrams:")
            reliability_grid(index, [("promoter_all", "promoter_all"),
                                     ("promoter_all", "enhancers")],
                             backbones, ece_summary=cal,
                             outpath=out / "reliability_regulatory.pdf")
            reliability_grid(index, [("pbsim_family", "id_novel_genus_family"),
                                     ("pbsim_family", "ood_novel_family_family")],
                             backbones, ece_summary=cal,
                             outpath=out / "reliability_metagenomic.pdf")

        if "risk" in wanted:
            print("Risk-coverage curves:")
            for train, test in [("promoter_all", "promoter_all"),
                                ("gene_taxa", "test"),
                                ("pbsim_family", "id_novel_genus_family")]:
                fig, axes = plt.subplots(1, len(backbones), squeeze=False,
                                         figsize=(4.6 * len(backbones), 4.2))
                for i, backbone in enumerate(backbones):
                    risk_coverage(index, backbone, train, test, ax=axes[0][i])
                fig.tight_layout()
                _save(fig, out / f"risk_coverage_{train}_{test}.pdf")
    elif wanted & {"reliability", "risk"}:
        print("[skip] reliability and risk-coverage need --results <results_root>")


if __name__ == "__main__":
    main()
