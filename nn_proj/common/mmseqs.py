import argparse
import os
import subprocess
import shutil
from typing import Dict, List

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer  
from sklearn.model_selection import train_test_split

def check_binary(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable '{name}' not found on PATH. "
            "Make sure MMseqs2 is installed and on your PATH."
        )


def write_labeled_mmseqs_tsv(mmseqs_tsv: str, out_tsv: str,
                            test_id_to_label: dict, train_id_to_label: dict):
    with open(mmseqs_tsv, "r") as fin, open(out_tsv, "w") as fout:
        fout.write("query_id\ttarget_id\ttrue_label\tpred_label\tpident\tevalue\tbits\tqcov\n")
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            qid, sid, pident, evalue, bits, qcov = parts[:6]
            true_lab = test_id_to_label.get(qid, "NA")
            pred_lab = train_id_to_label.get(sid, "NA")
            fout.write(f"{qid}\t{sid}\t{true_lab}\t{pred_lab}\t{pident}\t{evalue}\t{bits}\t{qcov}\n")


def load_dataset_from_path(path: str, split: str):
    if path is None:
        raise ValueError("data_path must be specified.")
    if path.startswith("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
        task = path.split("/")[-1]
        return load_NT_tasks(task=task, split=split)
    else:
        return load_local_dataset(path=path, encode_labels=False)


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
            seq_id = f"{prefix}_{i}|{lab}"
            seq = str(seq).replace(" ", "").upper()
            lab_str = str(lab)
            f.write(f">{seq_id}\n")
            f.write(seq + "\n")
            id_to_label[seq_id] = lab_str

    return id_to_label


def run_mmseqs_easy_search(
    query_fasta: str,
    target_fasta: str,
    out_tsv: str,
    tmp_dir: str,
    threads: int,
):
    """Run mmseqs easy-search (1 best hit per query) and write TSV.

    We request 6 columns matching the BLAST script:
      query, target, pident, evalue, bits, qcov
    """
    check_binary("mmseqs")

    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "mmseqs",
        "easy-search",
        query_fasta,
        target_fasta,
        out_tsv,
        tmp_dir,
        "--threads", str(threads),
        "--max-seqs", "1",  # best hit only
        "--search-type", "3",
        "--format-output", "query,target,pident,evalue,bits,qcov",
        # default format-mode (0) = plain TSV, no header
    ]
    subprocess.run(cmd, check=True)


def parse_mmseqs_tsv(path: str):
    """Parse the simple 6-column MMseqs out TSV.

    Columns (in order):
        query, target, pident, evalue, bits, qcov

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
    """1-NN classifier using the best MMseqs hit.

    test_ds:
        Dataset with columns 'sequence', 'labels'. We assume that the
        FASTA headers were in the form 'test_{i}' (matching write_fasta_from_dataset).
    hits_by_q:
        Mapping from 'test_{i}' -> hit info.
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
        qseqid = f"test_{i}|{true_lab}"
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

    precision = n_correct / n_with_hits if n_with_hits > 0 else float("nan")
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
    parser = argparse.ArgumentParser(description="Simple MMseqs2 evaluation between two NT-style datasets.")
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help=(
            "Path to training dataset. Either a local file, or "
            "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/<task>"
        ),
    )
    parser.add_argument("--test_data_path",type=str,required=True,help="Path to test dataset. Same conventions as train_data_path.",
)
    parser.add_argument("--train_split",type=str,default="train",help="Split name for NT tasks when loading train_data_path (default: train).",)
    parser.add_argument("--test_split",type=str,default="test",help="Split name for NT tasks when loading test_data_path (default: test).",)
    parser.add_argument("--outdir",type=str,default="mmseqs_eval",help="Directory to write FASTA, MMseqs output, and results.",)
    parser.add_argument("--threads",type=int,default=4,help="Number of MMseqs threads.",)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading training dataset from {args.train_data_path}")
    train_ds = load_dataset_from_path(args.train_data_path, split=args.train_split)

    #split = train_ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="labels")
    #train_ds = split["train"]
    
    labels = train_ds["labels"]
    idx = list(range(len(train_ds)))

    train_idx, _ = train_test_split(
        idx, test_size=0.1, random_state=42, stratify=labels
    )

    train_ds = train_ds.select(train_idx)

    

    print(f"Loading test dataset from {args.test_data_path}")
    test_ds = load_dataset_from_path(args.test_data_path, split=args.test_split)

    train_fasta = os.path.join(args.outdir, "train.fa")
    test_fasta = os.path.join(args.outdir, "test.fa")
    mmseqs_tsv = os.path.join(args.outdir, "mmseqs_hits.tsv")
    tmp_dir = os.path.join(args.outdir, "mmseqs_tmp")

    print("Writing training FASTA...")
    train_id_to_label = write_fasta_from_dataset(train_ds, train_fasta, prefix="train")

    print("Writing test FASTA...")
    _ = write_fasta_from_dataset(test_ds, test_fasta, prefix="test")

    print("Running MMseqs2 easy-search (test -> train)...")
    run_mmseqs_easy_search(
        query_fasta=test_fasta,
        target_fasta=train_fasta,
        out_tsv=mmseqs_tsv,
        tmp_dir=tmp_dir,
        threads=args.threads,
    )

    print("Parsing MMseqs2 output...")
    hits_by_q = parse_mmseqs_tsv(mmseqs_tsv)

    print("Evaluating best-hit classifier (1-NN)...")
    metrics = evaluate_best_hit(test_ds, hits_by_q, train_id_to_label)

    print("\n=== MMseqs2 1-NN summary ===")
    print(f"Queries (test sequences):     {metrics['n_queries']}")
    print(f"Queries with at least 1 hit:  {metrics['n_with_hits']}")
    print(f"Correct predictions:          {metrics['n_correct']}")
    print(f"Precision:                    {metrics['precision']:.4f}")
    print(f"Mean qcov (fraction):         {metrics['mean_qcov']:.3f}")
    print(f"Mean pident:                  {metrics['mean_pident']:.2f}")
    print(f"Mean e-value:                 {metrics['mean_evalue']:.3e}")
    print(f"MMseqs hits written to:       {mmseqs_tsv}")


if __name__ == "__main__":
    main()
