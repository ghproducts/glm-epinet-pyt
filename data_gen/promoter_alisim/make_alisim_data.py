#!/usr/bin/env python3
"""
make_alisim_data.py
====================

Builds a *continuously controlled* epistemic/OOD severity axis for the
`promoter_all` task by simulating sequence evolution away from real anchor
sequences using AliSim (bundled with IQ-TREE2; Ly-Trinh & Minh, MBE 2022).

For each of 400 real `promoter_all` test-split anchors (200 positive-label,
200 negative-label, stratified, fixed seed 20260907), we simulate a
descendant sequence under an HKY+Gamma4 substitution model at each of 8
branch lengths (substitutions/site), sweeping from "nearly identical"
(t=0.0, an exact-reproduction sanity check) to "well past saturation"
(t=1.5). The real anchor sequence is used as the literal root of the
simulation (via AliSim's `--root-seq FILE,SEQ_NAME` flag) -- descendants are
NOT redrawn from the model's stationary distribution, they are evolved
starting from the real sequence.

Design (see data_gen/promoter_alisim/README.md for full justification):
  - Data source: `promoter_all` via `nn_proj.common.datasets.load_NT_tasks`
    (InstaDeepAI/nucleotide_transformer_downstream_tasks_revised). NOT the
    data_gen/promoter_motifs/ FIMO/JASPAR pipeline -- unrelated, unused here.
  - Substitution model: HKY85 + Gamma(4 categories).
      * Base frequencies (piA, piC, piG, piT): estimated empirically by
        pooling nucleotide counts over the promoter_all TRAIN split. This is
        a legitimate corpus-derived statistic.
      * kappa (transition/transversion ratio): literature default of 2.0 for
        mammalian genomic sequence. NOT fit to this corpus -- promoter_all
        sequences are unrelated (non-orthologous) human promoter loci, so
        treating them as a real alignment and running IQ-TREE ModelFinder on
        them would misapply a phylogenetic-alignment assumption.
      * alpha (Gamma shape): literature default of 1.0. Also not corpus-fit.
  - Root: the real anchor sequence, fixed via `--root-seq`.
  - Tree: for each anchor, one call to `iqtree2 --alisim` simulates a
    *star tree* rooted at the anchor sequence, with 9 leaves attached
    directly to the root: `t0`..`t7` at the 8 branch lengths in
    BRANCH_LENGTHS (each an independent evolutionary draw from the anchor),
    plus a `ref` leaf fixed at branch length 0.0 as a built-in per-run
    sanity check (it must always come back identical to the anchor). This
    is mathematically identical to running 8 independent two-taxon
    (root -> descendant) simulations per anchor, batched into one process
    call for efficiency (400 iqtree2 invocations total instead of 3200).
  - No indels in this first pass (documented future extension).

Usage
-----
    python make_alisim_data.py --pilot          # 5 anchors x 8 branch lengths
    python make_alisim_data.py                  # full 400 x 8 = 3200 grid

Run with a Python environment that has `datasets`, `transformers`, `numpy`,
and `pandas` installed (this repo's `glm_epinet_venv` was used originally).
Requires an IQ-TREE2 binary with AliSim support (found automatically, or
pass --iqtree-bin / set ALISIM_IQTREE_BIN).
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
# Fixed design constants -- see module docstring / README.md for rationale.
# ---------------------------------------------------------------------------

TASK = "promoter_all"
SEED = 20260907  # fixed anchor-sampling seed, per task spec

N_POS_ANCHORS = 200
N_NEG_ANCHORS = 200

# Branch length grid, in substitutions/site. Fixed severity axis: spans
# "nearly identical" (0.0) to "well past saturation" (1.5). Do not change
# without a strong documented reason (see README.md).
BRANCH_LENGTHS: List[float] = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]

KAPPA = 2.0   # transition/transversion ratio; literature default for mammalian genomic DNA
ALPHA = 1.0   # Gamma(4) rate-heterogeneity shape; literature default

THIS_DIR = Path(__file__).resolve().parent
CSV_OUT_DIR = THIS_DIR / "csv_data"
PILOT_DIR = THIS_DIR / "pilot"

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
# ---------------------------------------------------------------------------

def compute_empirical_base_frequencies() -> Dict[str, float]:
    """Pool nucleotide counts across the promoter_all TRAIN split.

    Returns a dict {A,C,G,T: freq} that sums to exactly 1.0 (the last
    frequency absorbs rounding so the model string always sums to 1.0).
    """
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
    # Force exact sum-to-1 (avoid floating point drift breaking AliSim's
    # +F{...} parser expectations) by rounding A/C/G and setting T as the
    # remainder.
    fa = round(freqs["A"], 6)
    fc = round(freqs["C"], 6)
    fg = round(freqs["G"], 6)
    ft = round(1.0 - fa - fc - fg, 6)
    return {"A": fa, "C": fc, "G": fg, "T": ft}


# ---------------------------------------------------------------------------
# Step 2: stratified anchor sampling from the promoter_all TEST split.
# ---------------------------------------------------------------------------

@dataclass
class Anchor:
    anchor_id: str
    sequence: str
    original_label: int


def sample_anchors(n_pos: int, n_neg: int, seed: int) -> List[Anchor]:
    """Stratified random sample of anchors from the promoter_all TEST split.

    The upstream promoter_all TEST split contains a small number of exact
    duplicate rows (identical `name`, sequence, and label -- an upstream
    data artifact, not introduced here; 13/1584 rows as of this writing).
    We deduplicate by `name` before sampling so all requested anchors are
    guaranteed to be distinct sequences.
    """
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
    """Newick star tree: 8 branch-length leaves + a ref-check leaf, all
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
    # Deterministic per-anchor seed derived from the global seed.
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
        # Some IQ-TREE versions append ".fasta" instead of ".fa".
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

        # Built-in sanity check: the ref leaf (branch length 0.0) must
        # exactly reproduce the anchor, every single time.
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
# Sanity checks
# ---------------------------------------------------------------------------

def verify(df: pd.DataFrame, anchors: List[Anchor], strict_alphabet: bool = True) -> None:
    """Sanity checks over a generated rows DataFrame. Raises AssertionError
    on any failure."""
    anchor_by_id = {a.anchor_id: a for a in anchors}

    # 1) Alphabet check: ACGT only.
    if strict_alphabet:
        bad = df[~df["sequence"].apply(lambda s: set(s.upper()) <= {"A", "C", "G", "T"})]
        assert len(bad) == 0, f"{len(bad)} rows contain non-ACGT characters."

    # 2) Length check: every simulated sequence matches its anchor's length.
    for anchor_id, group in df.groupby("anchor_id"):
        anchor_len = len(anchor_by_id[anchor_id].sequence)
        bad_len = group[group["sequence"].str.len() != anchor_len]
        assert len(bad_len) == 0, (
            f"Anchor {anchor_id}: {len(bad_len)} sequences have the wrong length."
        )

    # 3) t=0.0 rows must be identical to the anchor (identity == 1.0 exactly).
    zero_bl = df[df["branch_length"] == 0.0]
    bad_zero = zero_bl[zero_bl["realized_identity_to_anchor"] != 1.0]
    assert len(bad_zero) == 0, (
        f"{len(bad_zero)} branch_length=0.0 rows are not identical to their anchor."
    )

    # 4) Identity should (on average, monotonically) decrease as branch
    # length increases. Check the mean identity per branch length is
    # non-increasing (allow tiny numerical slack).
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
                         help="Run only a small pilot (5 anchors x 8 branch lengths = 40 rows) "
                              "written to data_gen/promoter_alisim/pilot/, instead of the full grid.")
    parser.add_argument("--n-pilot-anchors", type=int, default=5)
    parser.add_argument("--n-pos", type=int, default=N_POS_ANCHORS)
    parser.add_argument("--n-neg", type=int, default=N_NEG_ANCHORS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--iqtree-bin", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None,
                         help="Override output CSV path (default: csv_data/promoter_alisim.csv, "
                              "or pilot/pilot_alisim.csv with --pilot).")
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

    print(f"[info] Sampling anchors from promoter_all TEST split (seed={args.seed})...")
    all_anchors = sample_anchors(args.n_pos, args.n_neg, args.seed)
    print(f"[info] Sampled {len(all_anchors)} anchors "
          f"({sum(a.original_label == 1 for a in all_anchors)} pos / "
          f"{sum(a.original_label == 0 for a in all_anchors)} neg).")

    if args.pilot:
        anchors = all_anchors[: args.n_pilot_anchors]
        out_csv = Path(args.out_csv) if args.out_csv else PILOT_DIR / "pilot_alisim.csv"
        work_dir = PILOT_DIR / "iqtree_work"
        keep_files = True
        print(f"[info] PILOT mode: {len(anchors)} anchors x {len(BRANCH_LENGTHS)} "
              f"branch lengths = {len(anchors) * len(BRANCH_LENGTHS)} rows.")
    else:
        anchors = all_anchors
        out_csv = Path(args.out_csv) if args.out_csv else CSV_OUT_DIR / "promoter_alisim.csv"
        work_dir = Path(tempfile.mkdtemp(prefix="alisim_full_"))
        keep_files = args.keep_work_files
        print(f"[info] FULL mode: {len(anchors)} anchors x {len(BRANCH_LENGTHS)} "
              f"branch lengths = {len(anchors) * len(BRANCH_LENGTHS)} rows.")

    rows = generate_rows_for_anchors(anchors, iqtree_bin, freqs, work_dir, keep_files=keep_files)
    df = pd.DataFrame(rows)

    verify(df, anchors)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[info] Wrote {len(df)} rows to {out_csv}")

    # Persist the exact frequencies/params used, for the README / provenance.
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
            },
            f,
            indent=2,
        )
    print(f"[info] Wrote run parameters to {params_path}")

    if not args.pilot and not keep_files:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
