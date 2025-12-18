#!/usr/bin/env python

import argparse
import os
import subprocess
import shutil
from typing import Dict, List

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer  # noqa: F401

from dataclasses import dataclass, field

import transformers
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn as nn
import torch 

def check_binary(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable '{name}' not found on PATH. "
            "Make sure BLAST+ is installed and on your PATH."
        )

def load_dataset_from_path(path: str, split: str):
    if path is None:
        raise ValueError("data_path must be specified.")
    if path.startswith("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
        task = path.split("/")[-1]
        return load_NT_tasks(task=task, split=split)
    else:
        return load_local_dataset(path=path)


def write_fasta_from_dataset(ds, fasta_path: str, prefix: str) -> Dict[str, str]:
    """Write a Dataset with 'sequence' and 'labels' to FASTA.

    Returns:
        id_to_label: mapping from sequence ID (header) to label.
    """
    id_to_label: Dict[str, str] = {}
    sequences = ds["sequence"]
    labels = ds["labels"]

    with open(fasta_path, "w") as f:
        for i, (seq, lab) in enumerate(zip(sequences, labels)):
            seq_id = f"{prefix}_{i}"
            seq = str(seq).replace(" ", "").upper()
            lab_str = str(lab)
            f.write(f">{seq_id}\n")
            f.write(seq + "\n")
            id_to_label[seq_id] = lab_str

    return id_to_label


def make_blast_db(fasta_path: str, db_prefix: str, dbtype: str = "nucl"):
    check_binary("makeblastdb")
    cmd = [
        "makeblastdb",
        "-in", fasta_path,
        "-dbtype", dbtype,
        "-out", db_prefix,
    ]
    subprocess.run(cmd, check=True)


def run_blast(query_fasta: str, db_prefix: str, out_tsv: str, program: str, num_threads: int):
    check_binary(program)
    outfmt = "6 qseqid sseqid pident evalue bitscore qcovs"
    cmd = [
        program,
        "-query", query_fasta,
        "-db", db_prefix,
        "-outfmt", outfmt,
        "-out", out_tsv,
        "-max_target_seqs", "1",  # best hit only
        "-num_threads", str(num_threads),
    ]
    subprocess.run(cmd, check=True)

def parse_blast_tsv(path: str):
    """Parse the simple 6-column BLAST outfmt 6 tsv.

    Returns:
        hits_by_q: dict qseqid -> hit dict
    """
    hits_by_q: Dict[str, Dict] = {}
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            qseqid, sseqid = parts[0], parts[1]
            pident = float(parts[2])
            evalue = float(parts[3])
            bitscore = float(parts[4])
            qcovs = float(parts[5])

            hits_by_q[qseqid] = {
                "sseqid": sseqid,
                "pident": pident,
                "evalue": evalue,
                "bitscore": bitscore,
                "qcovs": qcovs,
            }

    return hits_by_q


def evaluate_best_hit(test_ds, hits_by_q: Dict[str, Dict], train_labels: Dict[str, str]):
    """1-NN classifier using the best BLAST hit.

    test_ds:
        Dataset with columns 'sequence', 'labels'. We assume that the
        FASTA headers were in the form 'test_{i}' (matching write_fasta_from_dataset).
    hits_by_q:
        Mapping from 'test_{i}' -> BLAST hit info.
    train_labels:
        Mapping from subject seq IDs ('train_{j}') to labels (strings).

    Returns:
        metrics dict
    """
    sequences = test_ds["sequence"]
    labels = [str(l) for l in test_ds["labels"]]

    n = len(sequences)
    n_with_hits = 0
    n_correct = 0
    qcovs: List[float] = []
    pidents: List[float] = []
    evalues: List[float] = []

    for i, true_lab in enumerate(labels):
        qseqid = f"test_{i}"
        hit = hits_by_q.get(qseqid)
        if hit is None:
            continue

        n_with_hits += 1
        sseqid = hit["sseqid"]
        pred_lab = train_labels.get(sseqid)

        if pred_lab is not None and pred_lab == true_lab:
            n_correct += 1

        qcovs.append(hit["qcovs"])
        pidents.append(hit["pident"])
        evalues.append(hit["evalue"])

    precision = n_correct / n_with_hits if n > 0 else float("nan")
    mean_qcov = sum(qcovs) / len(qcovs) if qcovs else float("nan")
    mean_pident = sum(pidents) / len(pidents) if pidents else float("nan")
    mean_evalue = sum(evalues) / len(evalues) if evalues else float("nan")

    return {
        "n_queries": n,
        "n_with_hits": n_with_hits,
        "n_correct": n_correct,
        "precision": precision,
        "mean_qcov": mean_qcov,
        "mean_pident": mean_pident,
        "mean_evalue": mean_evalue,
    }


def main():
    parser = argparse.ArgumentParser(description="Simple BLAST evaluation between two NT-style datasets.")
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help=(
            "Path to training dataset. Either a local file, or "
            "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/<task>"
        ),
    )
    parser.add_argument("--test_data_path",type=str,required=True,help="Path to test dataset. Same conventions as train_data_path.",)
    parser.add_argument("--train_split",type=str,    default="train",help="Split name for NT tasks when loading train_data_path (default: train).",)
    parser.add_argument("--test_split",type=str,default="test",help="Split name for NT tasks when loading test_data_path (default: test).",)
    parser.add_argument("--outdir",type=str,default="blast_eval",help="Directory to write FASTA, BLAST db, and results.",)
    parser.add_argument("--dbtype",choices=["nucl", "prot"],default="nucl",help="BLAST database type (nucl -> blastn, prot -> blastp).",)
    parser.add_argument("--threads",type=int,default=4,help="Number of BLAST threads.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading training dataset from {args.train_data_path}")
    train_ds = load_dataset_from_path(args.train_data_path, split=args.train_split)
    split = train_ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="labels")

    train_ds = split["train"] 

    print(f"Loading test dataset from {args.test_data_path}")
    test_ds = load_dataset_from_path(args.test_data_path, split=args.test_split)

    train_fasta = os.path.join(args.outdir, "train.fa")
    test_fasta = os.path.join(args.outdir, "test.fa")
    db_prefix = os.path.join(args.outdir, "train_db")
    blast_tsv = os.path.join(args.outdir, "blast_hits.tsv")

    print("Writing training FASTA...")
    train_id_to_label = write_fasta_from_dataset(train_ds, train_fasta, prefix="train")

    print("Writing test FASTA...")
    _ = write_fasta_from_dataset(test_ds, test_fasta, prefix="test")

    print("Building BLAST database from training sequences...")
    make_blast_db(train_fasta, db_prefix, dbtype=args.dbtype)

    print("Running BLAST (test -> train)...")
    run_blast(test_fasta, db_prefix, blast_tsv, program="blastn", num_threads=args.threads)

    print("Parsing BLAST output...")
    hits_by_q = parse_blast_tsv(blast_tsv)

    print("Evaluating best-hit classifier (1-NN)...")
    metrics = evaluate_best_hit(test_ds, hits_by_q, train_id_to_label)

    print("\n=== BLAST 1-NN summary ===")
    print(f"Queries (test sequences):     {metrics['n_queries']}")
    print(f"Queries with at least 1 hit:  {metrics['n_with_hits']}")
    print(f"Correct predictions:          {metrics['n_correct']}")
    print(f"Precision:                    {metrics['precision']:.4f}")
    print(f"Mean qcovs (query coverage):  {metrics['mean_qcov']:.2f}")
    print(f"Mean pident:                  {metrics['mean_pident']:.2f}")
    print(f"Mean e-value:                 {metrics['mean_evalue']:.3e}")
    print(f"BLAST hits written to:        {blast_tsv}")


if __name__ == "__main__":
    main()
