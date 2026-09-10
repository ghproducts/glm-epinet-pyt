#!/usr/bin/env python3
"""
make_alisim_data_dense.py
==========================

Denser follow-up to `make_alisim_data.py`: the original 8-point branch-length
grid ({0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5}) showed a real but weak,
**non-monotonic** epistemic-uncertainty signal for 2 of 3 UQ methods
(mc_dropout and evidential peaked around branch_length 0.2-0.4 and declined
toward 1.5, rather than rising monotonically) -- too coarse a grid to
characterize the actual shape. This script reruns the identical anchor
sampling and AliSim simulation procedure over a 16-point grid that adds
resolution in the 0.1-1.0 region (where the non-monotonic behavior was seen)
and extends further (2.0, 3.0) to check whether uncertainty plateaus at
substitution saturation.

Everything else is unchanged from `make_alisim_data.py` (same module,
same anchor-sampling function/seed, same HKY+F+G4 model, same per-anchor
star-tree batching trick, same verify() sanity checks) -- see
`data_gen/promoter_alisim/README.md` for the full methodology writeup this
follows. Only the branch-length grid and default output paths differ, so
results are directly comparable to (and extend) the original 8-point run:
the SAME 400 anchors (seed 20260907, stratified 200 pos / 200 neg from the
promoter_all TEST split, deduplicated by name) are reused here.

Usage
-----
    python make_alisim_data_dense.py --pilot          # 5 anchors x 16 branch lengths
    python make_alisim_data_dense.py                  # full 400 x 16 = 6400 grid

Run with a Python environment that has `datasets`, `transformers`, `numpy`,
and `pandas` installed. Requires an IQ-TREE2 binary with AliSim support
(found automatically, or pass --iqtree-bin / set ALISIM_IQTREE_BIN).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make the repo importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nn_proj.common.datasets import load_NT_tasks  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed design constants -- identical to make_alisim_data.py except for the
# branch-length grid and output paths. See README.md for full rationale.
# ---------------------------------------------------------------------------

TASK = "promoter_all"
SEED = 20260907  # identical anchor-sampling seed as the original 8-point run

N_POS_ANCHORS = 200
N_NEG_ANCHORS = 200

# Dense 16-point branch-length grid, in substitutions/site. Adds resolution
# in the 0.1-1.0 region where the coarse 8-point grid showed non-monotonic
# behavior for mc_dropout/evidential, and extends to 2.0/3.0 to probe for a
# saturation plateau. Do not change without a strong documented reason.
BRANCH_LENGTHS: List[float] = [
    0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3,
    0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0,
]

KAPPA = 2.0   # transition/transversion ratio; literature default for mammalian genomic DNA
ALPHA = 1.0   # Gamma(4) rate-heterogeneity shape; literature default

THIS_DIR = Path(__file__).resolve().parent
CSV_OUT_DIR = THIS_DIR / "csv_data"
PILOT_DIR = THIS_DIR / "pilot_dense"

# Leaf names used in every per-anchor star tree. t<i> corresponds to
# BRANCH_LENGTHS[i]; "ref" is always pinned at branch length 0.0 as a
# per-run sanity check that must reproduce the anchor exactly.
LEAF_NAMES = [f"t{i}" for i in range(len(BRANCH_LENGTHS))]
REF_NAME = "ref"
ROOT_SEQ_NAME = "anchor"


def find_iqtree_binary(explicit: Optional[str] = None) -> str:
    """Locate an IQ-TREE2 binary with AliSim support.

    Search order: explicit CLI arg > $ALISIM_IQTREE_BIN env var > known
    install location used when this dataset was built > `iqtree2`/`iqtree`
    on PATH.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("ALISIM_IQTREE_BIN"):
        candidates.append(os.environ["ALISIM_IQTREE_BIN"])
    candidates.append("/scratch/home/glh52/tools/iqtree-2.4.0-Linux-intel/bin/iqtree2")
    which_iqtree2 = shutil.which("iqtree2")
    if which_iqtree2:
        candidates.append(which_iqtree2)
    which_iqtree = shutil.which("iqtree")
    if which_iqtree:
        candidates.append(which_iqtree)

    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        "Could not find an IQ-TREE2 binary. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ". Pass --iqtree-bin or set ALISIM_IQTREE_BIN."
    )


# ---------------------------------------------------------------------------
# Step 1: empirical base frequencies from the promoter_all TRAIN split.
# (Identical to make_alisim_data.py -- recomputed here for a self-contained
# script; deterministic given the same corpus, so this reproduces the same
# frequencies as the original run.)
# ---------------------------------------------------------------------------

def compute_empirical_base_frequencies() -> Dict[str, float]:
    ds = load_NT_tasks(task=TASK, split="train")
    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    other = 0
    for seq in ds["sequence"]:
        for ch in seq.upper():
            if ch in counts:
                counts[ch] += 1
            else:
                other += 1
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("No ACGT bases found in promoter_all train split.")
    if other:
        print(f"[warn] {other} non-ACGT characters found in train split and ignored "
              f"for frequency estimation.")

    freqs = {k: v / total for k, v in counts.items()}
    fa = round(freqs["A"], 6)
    fc = round(freqs["C"], 6)
    fg = round(freqs["G"], 6)
    ft = round(1.0 - fa - fc - fg, 6)
    return {"A": fa, "C": fc, "G": fg, "T": ft}


# ---------------------------------------------------------------------------
# Step 2: stratified anchor sampling from the promoter_all TEST split.
# IDENTICAL logic/seed to make_alisim_data.py's sample_anchors() -- this is
# what guarantees the SAME 400 anchors are used here as in the original
# 8-point run, making the two datasets directly comparable.
# ---------------------------------------------------------------------------

@dataclass
class Anchor:
    anchor_id: str
    sequence: str
    original_label: int


def sample_anchors(n_pos: int, n_neg: int, seed: int) -> List[Anchor]:
    ds = load_NT_tasks(task=TASK, split="test")
    names = ds["name"]
    seqs = ds["sequence"]
    labels = ds["labels"]

    seen_names = set()
    unique_idx = []
    for i, n in enumerate(names):
        if n not in seen_names:
            seen_names.add(n)
            unique_idx.append(i)
    n_dupes = len(names) - len(unique_idx)
    if n_dupes:
        print(f"[warn] promoter_all test split contains {n_dupes} duplicate-name rows; "
              f"deduplicated before anchor sampling.")

    pos_idx = [i for i in unique_idx if labels[i] == 1]
    neg_idx = [i for i in unique_idx if labels[i] == 0]

    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    if len(pos_idx) < n_pos or len(neg_idx) < n_neg:
        raise RuntimeError(
            f"Not enough test examples to sample {n_pos} pos / {n_neg} neg "
            f"(have {len(pos_idx)} pos, {len(neg_idx)} neg)."
        )

    chosen = sorted(pos_idx[:n_pos] + neg_idx[:n_neg])
    anchors = [
        Anchor(anchor_id=names[i], sequence=seqs[i].upper(), original_label=int(labels[i]))
        for i in chosen
    ]
    return anchors


# ---------------------------------------------------------------------------
# Step 3: AliSim invocation.
# ---------------------------------------------------------------------------

def build_star_tree() -> str:
    """Newick star tree: 16 branch-length leaves + a ref-check leaf, all
    attached directly to the (unnamed) root that AliSim will seed with the
    real anchor sequence via --root-seq."""
    parts = [f"{name}:{bl}" for name, bl in zip(LEAF_NAMES, BRANCH_LENGTHS)]
    parts.append(f"{REF_NAME}:0.0")
    return "(" + ",".join(parts) + ");"


def parse_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = ""
            else:
                seqs[name] += line.strip()
    return seqs


def run_alisim_for_anchor(
    iqtree_bin: str,
    anchor: Anchor,
    anchor_index: int,
    work_dir: Path,
    freqs: Dict[str, float],
    keep_files: bool = False,
) -> Dict[str, str]:
    """Run one iqtree2 --alisim call for a single anchor's star tree.

    Returns {leaf_name: simulated_sequence} for all leaves in LEAF_NAMES + ref.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = work_dir / "root.fasta"
    tree_path = work_dir / "star.tree"
    out_prefix = work_dir / "out"

    with open(fasta_path, "w") as f:
        f.write(f">{ROOT_SEQ_NAME}\n{anchor.sequence}\n")
    with open(tree_path, "w") as f:
        f.write(build_star_tree() + "\n")

    model = f"HKY{{{KAPPA}}}+F{{{freqs['A']}/{freqs['C']}/{freqs['G']}/{freqs['T']}}}+G4{{{ALPHA}}}"
    # Deterministic per-anchor seed derived from the global seed. Same
    # convention as make_alisim_data.py; the larger leaf set here draws more
    # random numbers per anchor so shared branch lengths won't numerically
    # match the original run row-for-row, but the ANCHOR identities are
    # identical, which is what makes the two runs comparable.
    iqtree_seed = SEED + anchor_index

    cmd = [
        iqtree_bin,
        "--alisim", str(out_prefix),
        "-t", str(tree_path),
        "-m", model,
        "--root-seq", f"{fasta_path},{ROOT_SEQ_NAME}",
        "--seed", str(iqtree_seed),
        "-af", "fasta",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"AliSim failed for anchor {anchor.anchor_id!r}:\n"
            f"cmd: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    out_fa = out_prefix.with_suffix(".fa")
    if not out_fa.exists():
        alt = out_prefix.with_suffix(".fasta")
        if alt.exists():
            out_fa = alt
        else:
            raise RuntimeError(f"Expected AliSim output not found: {out_fa} (or .fasta)")

    seqs = parse_fasta(out_fa)

    if not keep_files:
        shutil.rmtree(work_dir, ignore_errors=True)

    return seqs


# ---------------------------------------------------------------------------
# Step 4: identity computation + row assembly.
# ---------------------------------------------------------------------------

def sequence_identity(a: str, b: str) -> float:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return 1.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def generate_rows_for_anchors(
    anchors: List[Anchor],
    iqtree_bin: str,
    freqs: Dict[str, float],
    base_work_dir: Path,
    keep_files: bool = False,
) -> List[dict]:
    rows = []
    for i, anchor in enumerate(anchors):
        work_dir = base_work_dir / f"anchor_{i:04d}"
        seqs = run_alisim_for_anchor(
            iqtree_bin, anchor, i, work_dir, freqs, keep_files=keep_files
        )

        if seqs[REF_NAME] != anchor.sequence:
            raise RuntimeError(
                f"Sanity check FAILED for anchor {anchor.anchor_id!r}: "
                f"'ref' leaf (branch length 0.0) does not exactly match the "
                f"anchor sequence. This indicates --root-seq is not behaving "
                f"as expected."
            )

        for leaf_name, bl in zip(LEAF_NAMES, BRANCH_LENGTHS):
            sim_seq = seqs[leaf_name]
            identity = sequence_identity(anchor.sequence, sim_seq)
            rows.append(
                {
                    "sequence": sim_seq,
                    "labels": anchor.original_label,
                    "anchor_id": anchor.anchor_id,
                    "original_label": anchor.original_label,
                    "branch_length": bl,
                    "realized_identity_to_anchor": identity,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Sanity checks (identical to make_alisim_data.py's verify()).
# ---------------------------------------------------------------------------

def verify(df: pd.DataFrame, anchors: List[Anchor], strict_alphabet: bool = True) -> None:
    anchor_by_id = {a.anchor_id: a for a in anchors}

    if strict_alphabet:
        bad = df[~df["sequence"].apply(lambda s: set(s.upper()) <= {"A", "C", "G", "T"})]
        assert len(bad) == 0, f"{len(bad)} rows contain non-ACGT characters."

    for anchor_id, group in df.groupby("anchor_id"):
        anchor_len = len(anchor_by_id[anchor_id].sequence)
        bad_len = group[group["sequence"].str.len() != anchor_len]
        assert len(bad_len) == 0, (
            f"Anchor {anchor_id}: {len(bad_len)} sequences have the wrong length."
        )

    zero_bl = df[df["branch_length"] == 0.0]
    bad_zero = zero_bl[zero_bl["realized_identity_to_anchor"] != 1.0]
    assert len(bad_zero) == 0, (
        f"{len(bad_zero)} branch_length=0.0 rows are not identical to their anchor."
    )

    mean_by_bl = df.groupby("branch_length")["realized_identity_to_anchor"].mean().sort_index()
    vals = mean_by_bl.values
    for j in range(1, len(vals)):
        assert vals[j] <= vals[j - 1] + 1e-9, (
            f"Mean identity increased with branch length: {mean_by_bl}"
        )

    print("[verify] All sanity checks passed.")
    print("[verify] Mean identity by branch length:")
    print(mean_by_bl.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true",
                         help="Run only a small pilot (5 anchors x 16 branch lengths = 80 rows) "
                              "written to data_gen/promoter_alisim/pilot_dense/, instead of the full grid.")
    parser.add_argument("--n-pilot-anchors", type=int, default=5)
    parser.add_argument("--n-pos", type=int, default=N_POS_ANCHORS)
    parser.add_argument("--n-neg", type=int, default=N_NEG_ANCHORS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--iqtree-bin", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None,
                         help="Override output CSV path (default: csv_data/promoter_alisim_dense.csv, "
                              "or pilot_dense/pilot_alisim_dense.csv with --pilot).")
    parser.add_argument("--keep-work-files", action="store_true",
                         help="Keep per-anchor AliSim scratch files (fasta/tree/log) instead of "
                              "deleting them after each anchor. Always kept for --pilot.")
    args = parser.parse_args()

    iqtree_bin = find_iqtree_binary(args.iqtree_bin)
    print(f"[info] Using IQ-TREE2 binary: {iqtree_bin}")
    version = subprocess.run([iqtree_bin, "--version"], capture_output=True, text=True)
    print(f"[info] {version.stdout.splitlines()[0] if version.stdout else version.stderr.splitlines()[0]}")

    print("[info] Computing empirical base frequencies from promoter_all TRAIN split...")
    freqs = compute_empirical_base_frequencies()
    print(f"[info] Empirical base frequencies: {freqs} (sum={sum(freqs.values()):.6f})")
    print(f"[info] kappa (literature default) = {KAPPA}, alpha (literature default) = {ALPHA}")
    print(f"[info] Dense branch-length grid ({len(BRANCH_LENGTHS)} points): {BRANCH_LENGTHS}")

    print(f"[info] Sampling anchors from promoter_all TEST split (seed={args.seed})...")
    all_anchors = sample_anchors(args.n_pos, args.n_neg, args.seed)
    print(f"[info] Sampled {len(all_anchors)} anchors "
          f"({sum(a.original_label == 1 for a in all_anchors)} pos / "
          f"{sum(a.original_label == 0 for a in all_anchors)} neg).")

    if args.pilot:
        anchors = all_anchors[: args.n_pilot_anchors]
        out_csv = Path(args.out_csv) if args.out_csv else PILOT_DIR / "pilot_alisim_dense.csv"
        work_dir = PILOT_DIR / "iqtree_work"
        keep_files = True
        print(f"[info] PILOT mode: {len(anchors)} anchors x {len(BRANCH_LENGTHS)} "
              f"branch lengths = {len(anchors) * len(BRANCH_LENGTHS)} rows.")
    else:
        anchors = all_anchors
        out_csv = Path(args.out_csv) if args.out_csv else CSV_OUT_DIR / "promoter_alisim_dense.csv"
        work_dir = Path(tempfile.mkdtemp(prefix="alisim_dense_full_"))
        keep_files = args.keep_work_files
        print(f"[info] FULL mode: {len(anchors)} anchors x {len(BRANCH_LENGTHS)} "
              f"branch lengths = {len(anchors) * len(BRANCH_LENGTHS)} rows.")

    rows = generate_rows_for_anchors(anchors, iqtree_bin, freqs, work_dir, keep_files=keep_files)
    df = pd.DataFrame(rows)

    verify(df, anchors)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[info] Wrote {len(df)} rows to {out_csv}")

    params_path = out_csv.parent / (out_csv.stem + "_params.json")
    with open(params_path, "w") as f:
        json.dump(
            {
                "task": TASK,
                "seed": args.seed,
                "n_pos_anchors": len(anchors) if args.pilot else args.n_pos,
                "n_neg_anchors": None if args.pilot else args.n_neg,
                "branch_lengths": BRANCH_LENGTHS,
                "kappa": KAPPA,
                "alpha": ALPHA,
                "base_frequencies": freqs,
                "iqtree_bin": iqtree_bin,
                "iqtree_version": version.stdout.strip() if version.stdout else version.stderr.strip(),
                "note": "Dense follow-up grid; same 400 anchors (seed 20260907) as "
                        "csv_data/promoter_alisim.csv (the original 8-point grid).",
            },
            f,
            indent=2,
        )
    print(f"[info] Wrote run parameters to {params_path}")

    if not args.pilot and not keep_files:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
