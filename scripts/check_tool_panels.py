#!/usr/bin/env python3
"""Reproduce the Kraken2 / MMseqs2 comparison, and show what it depends on.

Recomputes the tool-based confidence panels from the stored Kraken2 and
MMseqs2 output and prints them beside the published values. Two of the choices
buried in the original analysis change the answer a lot, so both are exposed
here rather than fixed silently:

  --hits {all,best}
      `all` keeps every alignment row, so a query with ten hits is weighted ten
      times as heavily; `best` keeps one row per query. The published panels
      used `all`. Queries with no hit are absent under either, which for the
      regulatory tasks removes most of the test set.

  --taxid-map {species,full}
      How a Kraken2 taxid is projected onto the target rank. `species` uses
      only the lineage table's species column; `full` uses every lineage
      column. Kraken2 assigns reads to internal nodes, so `species` discards
      most reads and `full` keeps them. This single choice moves ECE by up to
      40 percentage points on the same input.

Usage
-----
    python scripts/check_tool_panels.py \\
        --kraken2 <dir>/kraken2 --mmseqs <dir>/mmseqs \\
        --lineage data_gen/pbsim/full_basic_lineage.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nn_proj.analysis.metrics import compute_ece  # noqa: E402

MMSEQS_COLS = ["query", "target", "pident", "evalue", "bits", "qcov"]

# Published values: (ECE %, weighted reliability slope).
PUBLISHED = {
    ("mmseqs", "promoter_all", "promoter_all", None, "pident"): (10.3, -0.175),
    ("mmseqs", "promoter_all", "promoter_all", None, "qcov"): (30.8, +0.264),
    ("mmseqs", "promoter_all", "enhancers", None, "pident"): (15.3, -0.045),
    ("mmseqs", "promoter_all", "enhancers", None, "qcov"): (26.3, +0.713),
    ("mmseqs", "pbsim", "id_novel_genus", "class", "pident"): (18.7, -1.789),
    ("mmseqs", "pbsim", "id_novel_genus", "class", "qcov"): (61.7, +2.826),
    ("mmseqs", "pbsim", "ood_novel_family", "class", "pident"): (34.8, -1.475),
    ("mmseqs", "pbsim", "ood_novel_family", "class", "qcov"): (41.2, +4.932),
    ("mmseqs", "pbsim", "id_novel_genus", "phylum", "pident"): (15.7, -1.485),
    ("mmseqs", "pbsim", "id_novel_genus", "phylum", "qcov"): (77.2, +1.955),
    ("mmseqs", "pbsim", "ood_novel_family", "phylum", "pident"): (17.2, -1.550),
    ("mmseqs", "pbsim", "ood_novel_family", "phylum", "qcov"): (65.2, -1.447),
    ("kraken2", "pbsim", "id_novel_genus", "class", "conf"): (27.6, +0.557),
    ("kraken2", "pbsim", "ood_novel_family", "class", "conf"): (41.7, +0.290),
    ("kraken2", "pbsim", "id_novel_genus", "phylum", "conf"): (18.2, +0.752),
    ("kraken2", "pbsim", "ood_novel_family", "phylum", "conf"): (32.2, +0.315),
}


def build_taxid_map(lineage: pd.DataFrame, rank: str, mode: str) -> Dict[int, int]:
    """Project taxids onto ``rank``. See --taxid-map in the module docstring."""
    if mode == "species":
        pairs = lineage[["species", rank]].dropna()
        return dict(zip(pairs["species"].astype(int), pairs[rank].astype(int)))

    out: Dict[int, int] = {}
    for _, row in lineage.iterrows():
        target = row[rank]
        if pd.isna(target):
            continue
        target = int(target)
        for col in lineage.columns:
            value = row[col]
            if pd.isna(value):
                continue
            try:
                out.setdefault(int(value), target)
            except (TypeError, ValueError):
                continue
    return out


def weighted_slope(df: pd.DataFrame, conf_col: str, n_bins: int = 20,
                   min_count: int = 10) -> float:
    """Weighted least-squares slope through the reliability points.

    A slope near +1 means confidence tracks accuracy; near 0 means the score
    carries no calibration information; negative means it is inverted.
    """
    conf = df[conf_col].to_numpy(dtype=float)
    correct = (df["labels"] == df["pred"]).to_numpy(dtype=float)
    ok = np.isfinite(conf)
    conf, correct = np.clip(conf[ok], 0.0, 1.0), correct[ok]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)

    x, y, w = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.sum() >= min_count:
            x.append(conf[m].mean()); y.append(correct[m].mean()); w.append(m.sum())
    if len(x) < 2:
        return float("nan")

    x, y, w = np.array(x), np.array(y), np.array(w, dtype=float)
    total = w.sum()
    mx, my = (w * x).sum() / total, (w * y).sum() / total
    denom = (w * (x - mx) ** 2).sum()
    return float((w * (x - mx) * (y - my)).sum() / denom) if denom > 0 else float("nan")


def load_mmseqs(path: Path, rank_map: Optional[Dict[int, int]], hits: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=MMSEQS_COLS)
    df["labels"] = df["query"].astype(str).str.split("|").str[-1]
    df["pred"] = df["target"].astype(str).str.split("|").str[-1]
    df["pident"] = df["pident"].astype(float) / 100.0
    df["qcov"] = df["qcov"].astype(float)
    if hits == "best":
        df = df.sort_values("bits", ascending=False).groupby("query", as_index=False).first()
    if rank_map is not None:
        # pbsim labels are family taxids; project them onto the target rank.
        for col in ("labels", "pred"):
            df[col] = df[col].astype("Int64").map(rank_map)
        df = df.dropna(subset=["labels", "pred"])
    return df


def load_kraken2(path: Path, rank_map: Dict[int, int]) -> pd.DataFrame:
    rows = []
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        try:
            pred_raw = int(parts[2])
            true_species = int(parts[1].split("|")[1])
        except (ValueError, IndexError):
            continue
        pred, true = rank_map.get(pred_raw), rank_map.get(true_species)
        if pred is None or true is None:
            continue

        support_pred = support_informative = 0
        for token in parts[4].split():
            if ":" not in token:
                continue
            tax_s, count_s = token.split(":", 1)
            if not tax_s.isdigit():
                continue
            tax_raw, count = int(tax_s), int(count_s)
            if tax_raw == 0:
                continue
            mapped = rank_map.get(tax_raw)
            if mapped is None:
                continue
            if mapped > 0:
                support_informative += count
            if mapped == pred:
                support_pred += count

        rows.append({"labels": int(true), "pred": int(pred),
                     "conf": support_pred / support_informative if support_informative else 0.0})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kraken2", type=Path, required=True, help="Directory of Kraken2 output")
    ap.add_argument("--mmseqs", type=Path, required=True, help="Directory of MMseqs2 TSVs")
    ap.add_argument("--lineage", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data_gen/pbsim/full_basic_lineage.csv")
    ap.add_argument("--hits", choices=["all", "best"], default="all",
                    help="Keep every alignment row, or the best per query (published: all)")
    ap.add_argument("--taxid-map", choices=["species", "full"], default="species",
                    help="How Kraken2 taxids are projected onto the target rank")
    ap.add_argument("--ece-bins", type=int, default=50)
    args = ap.parse_args()

    lineage = pd.read_csv(args.lineage, index_col=0)
    maps = {r: build_taxid_map(lineage, r, args.taxid_map) for r in ("class", "phylum")}
    fam_maps = {
        r: dict(zip(lineage["family"].astype("Int64"), lineage[r].astype("Int64")))
        for r in ("class", "phylum")
    }

    print(f"hits={args.hits}  taxid-map={args.taxid_map}  ece-bins={args.ece_bins}\n")
    print(f"{'tool':<9}{'panel':<38}{'score':<8}{'n':>7}"
          f"{'ECE pub':>9}{'ECE now':>9}{'d':>7}{'slope pub':>10}{'slope now':>10}")

    worst = 0.0
    for (tool, train, test, rank, score), (pub_ece, pub_slope) in PUBLISHED.items():
        if tool == "mmseqs":
            path = args.mmseqs / train / f"{test}.tsv"
            if not path.exists():
                continue
            df = load_mmseqs(path, fam_maps.get(rank) if rank else None, args.hits)
            col = score
        else:
            path = args.kraken2 / train / f"{test}.txt"
            if not path.exists():
                continue
            df = load_kraken2(path, maps[rank])
            col = "conf"

        ece = compute_ece(df, conf_col=col, n_bins=args.ece_bins) * 100
        slope = weighted_slope(df, col)
        worst = max(worst, abs(ece - pub_ece))
        panel = f"{train}->{test}" + (f" ({rank})" if rank else "")
        print(f"{tool:<9}{panel:<38}{score:<8}{len(df):>7}"
              f"{pub_ece:>9.1f}{ece:>9.1f}{ece - pub_ece:>+7.1f}"
              f"{pub_slope:>+10.3f}{slope:>+10.3f}")

    print(f"\nmax |ECE difference| = {worst:.1f} pp")
    print("Re-run with --taxid-map full to see how far the Kraken2 rows move; "
          "see docs/FINDINGS.md item 9.")


if __name__ == "__main__":
    main()
