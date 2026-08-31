#!/usr/bin/env python3
"""Turn Kraken2 and MMseqs2 output into standard prediction frames.

Both tools produce a per-read taxonomic call plus a score, but neither score is
a calibrated probability, and neither tool emits one. This script derives a
confidence surrogate from each and writes it in exactly the format
`nn_proj.models.epinet.predict` produces, so the same analysis code handles
tools and models without a parallel code path.

**On interpreting the result.** These surrogates were not designed to be
probabilities of correctness. Showing that an uncalibrated similarity score is
poorly calibrated is close to a tautology and is not evidence that deep models
are better. To make a fair comparison, either

  * fit a temperature or isotonic regression on the same validation split the
    models use, and report post-calibration numbers; or
  * compare the scores on their ability to *rank* errors — AUROC, or the
    risk-coverage curves in `nn_proj.analysis.metrics` — which is what these
    scores are actually built for.

This script does the second: its output is a standard prediction frame, so
`nn_proj.analysis.metrics` gives AUROC and risk-coverage directly. It does not
fit a calibrator; that has to be done against the models' own validation split
and is not implemented here.

Note on comparability: the tools are run on the full simulated read, while the
models see only the first ~6 kbp after tokenizer truncation. Any comparison
should either truncate the reads before running the tool or state the
asymmetry, since the tool is otherwise given roughly 1.8x more sequence.

Kraken2 confidence
------------------
Kraken2's per-read output carries a k-mer hit list of `taxid:count` tokens. The
surrogate is the share of k-mer support behind the predicted taxon:

    conf_informative = support(pred) / support(all taxa mapping to the rank)
    conf_all         = support(pred) / support(every token, unclassified too)

`conf_informative` conditions on the read being classifiable at all;
`conf_all` folds in the unclassified fraction. They differ substantially for
reads with low overall support, so both are emitted and which one is used must
be stated.

MMseqs2 confidence
------------------
Percent identity (`pident`) and query coverage (`qcov`) of the best hit,
rescaled to [0, 1]. Reads with no hit have no score and are recorded with
`has_hit = False` rather than being silently dropped, since discarding them
inflates apparent accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

RANKS = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]


def load_rank_map(lineage_csv: Path, target_rank: str) -> Dict[int, int]:
    """Map any taxid appearing as a species to its ancestor at ``target_rank``."""
    df = pd.read_csv(lineage_csv)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
    if target_rank not in df.columns:
        raise ValueError(f"{lineage_csv} has no '{target_rank}' column")
    pairs = df[["species", target_rank]].dropna()
    return dict(zip(pairs["species"].astype(int), pairs[target_rank].astype(int)))


def label_from_read_id(read_id: str, field: int = 2) -> Optional[int]:
    """Pull the true label out of a pipe-delimited read id.

    The simulated reads carry their provenance in the FASTA header, e.g.
    ``kraken:taxid|1002689|204434|id_novel_genus|2`` where field 1 is the
    source species and field 2 the family. Reading the label from the read id
    is safer than aligning a separate label file by row order, which silently
    mis-pairs everything if the tool reordered or dropped reads.
    """
    parts = read_id.split("|")
    if len(parts) <= field:
        return None
    try:
        return int(parts[field])
    except ValueError:
        return None


def parse_kraken2(
    report_path: Path,
    rank_map: Optional[Dict[int, int]] = None,
    label_field: Optional[int] = None,
) -> pd.DataFrame:
    """Parse Kraken2 standard output into a prediction frame.

    Expected columns (tab separated):
        classified_flag, read_id, taxid, length, hit_list
    """
    rows = []
    with open(report_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            _, read_id, taxid_s, _, hit_list = parts[:5]
            try:
                pred_raw = int(taxid_s)
            except ValueError:
                continue

            pred = rank_map.get(pred_raw, 0) if rank_map is not None else pred_raw

            support_pred = support_informative = support_all = 0
            for token in hit_list.split():
                if ":" not in token:
                    continue
                tax_s, count_s = token.split(":", 1)
                if not tax_s.isdigit():
                    continue  # 'A' marks an ambiguous-nucleotide run
                try:
                    tax_raw, count = int(tax_s), int(count_s)
                except ValueError:
                    continue

                support_all += count
                if tax_raw == 0:
                    continue  # unclassified segment: never informative

                if rank_map is not None:
                    mapped = rank_map.get(tax_raw)
                    if mapped is None:
                        continue  # unmappable at this rank
                    tax = int(mapped)
                else:
                    tax = tax_raw

                if tax > 0:
                    support_informative += count
                if tax == pred:
                    support_pred += count

            row = {
                "read_id": read_id,
                "pred": pred,
                "conf_informative": support_pred / support_informative if support_informative else 0.0,
                "conf_all": support_pred / support_all if support_all else 0.0,
                "has_hit": parts[0] == "C",
                "read_length": int(parts[3]) if parts[3].isdigit() else None,
            }
            if label_field is not None:
                row["labels"] = label_from_read_id(read_id, label_field)
            rows.append(row)
    return pd.DataFrame(rows)


def parse_mmseqs(tsv_path: Path, n_queries: Optional[int] = None) -> pd.DataFrame:
    """Parse MMseqs2 easy-search output, keeping the best hit per query.

    Expected columns: query, target, pident, alnlen, ..., with qcov last if
    the search was run with the coverage output format.
    """
    df = pd.read_csv(tsv_path, sep="\t", header=None)
    df.columns = ([f"c{i}" for i in range(df.shape[1])])
    df = df.rename(columns={"c0": "query", "c1": "target", "c2": "pident"})
    if "c10" in df.columns:
        df = df.rename(columns={"c10": "evalue"})
    qcov_col = df.columns[-1]
    df = df.rename(columns={qcov_col: "qcov"})

    # MMseqs reports hits sorted by score; the first per query is the best.
    best = df.groupby("query", as_index=False).first()

    # pident is reported as a fraction in some versions and a percentage in
    # others; normalise to [0, 1] either way.
    pident = best["pident"].astype(float)
    if pident.max() > 1.0:
        pident = pident / 100.0
    qcov = best["qcov"].astype(float)
    if qcov.max() > 1.0:
        qcov = qcov / 100.0

    out = pd.DataFrame({
        "read_id": best["query"],
        "pred": best["target"],
        "conf_pident": pident.clip(0, 1),
        "conf_qcov": qcov.clip(0, 1),
        "has_hit": True,
    })

    if n_queries is not None and len(out) < n_queries:
        print(f"[mmseqs] {n_queries - len(out)} of {n_queries} queries had no hit; "
              "recorded with confidence 0 rather than dropped")
    return out


def to_prediction_frame(df: pd.DataFrame, labels: pd.Series, conf_col: str) -> pd.DataFrame:
    """Shape a tool's output like an inference_uncertainty.csv.

    Total uncertainty is defined as 1 - confidence, which is the only
    uncertainty a point-estimate score supports. There is no aleatoric or
    epistemic decomposition, so those columns are deliberately absent rather
    than filled with zeros that would be read as a measurement.
    """
    out = pd.DataFrame({
        "labels": labels.values,
        "pred": df["pred"].values,
        "max_confidence": df[conf_col].astype(float).clip(0, 1).values,
        "has_hit": df.get("has_hit", True),
    })
    out["U_total"] = 1.0 - out["max_confidence"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", choices=["kraken2", "mmseqs"])
    ap.add_argument("input", type=Path, help="Kraken2 output file or MMseqs2 result TSV")
    ap.add_argument("--labels", type=Path, default=None,
                    help="CSV of the evaluated reads with a `labels` column. Aligned by row "
                         "order, so prefer --label-field when the read ids carry the label.")
    ap.add_argument("--label-field", type=int, default=None,
                    help="Zero-based field of the pipe-delimited read id holding the true "
                         "label (2 for the simulated reads). Safer than --labels: it cannot "
                         "mis-pair if the tool reordered or dropped reads.")
    ap.add_argument("--lineage", type=Path, default=None,
                    help="Lineage CSV, required when --rank is given")
    ap.add_argument("--rank", choices=RANKS, default=None,
                    help="Map predictions and hit-list taxids up to this rank")
    ap.add_argument("--conf-col", default=None,
                    help="Which surrogate to use as confidence "
                         "(kraken2: conf_informative|conf_all, mmseqs: conf_pident|conf_qcov)")
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    args = ap.parse_args()

    rank_map = None
    if args.rank:
        if not args.lineage:
            ap.error("--rank requires --lineage")
        rank_map = load_rank_map(args.lineage, args.rank)
        print(f"Mapping taxids to {args.rank} ({len(rank_map)} species mapped)")

    if args.tool == "kraken2":
        parsed = parse_kraken2(args.input, rank_map, args.label_field)
        conf_col = args.conf_col or "conf_informative"
    else:
        parsed = parse_mmseqs(args.input)
        conf_col = args.conf_col or "conf_pident"
    print(f"Parsed {len(parsed)} prediction(s) from {args.input}")

    if "labels" in parsed.columns and parsed["labels"].notna().any():
        labels = parsed["labels"]
        n = len(parsed)
        print(f"Labels taken from read ids (field {args.label_field})")
    elif args.labels:
        truth = pd.read_csv(args.labels)
        label_col = "labels" if "labels" in truth.columns else "label"
        if len(truth) != len(parsed):
            print(f"[warn] {len(truth)} labelled reads but {len(parsed)} tool predictions; "
                  "aligning on row order, which assumes the tool preserved input order")
        n = min(len(truth), len(parsed))
        labels = truth[label_col].iloc[:n]
    else:
        ap.error("provide --label-field or --labels")

    if "read_length" in parsed.columns and parsed["read_length"].notna().any():
        lengths = parsed["read_length"].dropna()
        print(f"Read length seen by the tool: median {int(lengths.median())} bp "
              f"(range {int(lengths.min())}-{int(lengths.max())})")

    frame = to_prediction_frame(parsed.iloc[:n], labels, conf_col)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "inference_uncertainty.csv"
    frame.to_csv(out, index=False)

    acc = (frame["labels"].astype(str) == frame["pred"].astype(str)).mean()
    print(f"\nconfidence surrogate : {conf_col}")
    print(f"accuracy             : {acc:.4f}")
    print(f"mean confidence      : {frame['max_confidence'].mean():.4f}")
    print(f"reads with no hit    : {int((~frame['has_hit'].astype(bool)).sum())}")
    print(f"\nWrote {out}")
    print("\nThis score is uncalibrated by construction. Report its ECE only "
          "alongside a calibrated version or a ranking metric; see the module "
          "docstring.")


if __name__ == "__main__":
    main()
