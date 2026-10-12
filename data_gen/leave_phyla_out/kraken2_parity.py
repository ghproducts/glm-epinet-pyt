#!/usr/bin/env python3
"""Kraken2 given EXACTLY the data each neural model was trained on.

The first Kraken2 run indexed reference genomes: 1,406 species across all 43
training phyla, 5.71GB of error-free complete sequence. Each neural model saw
0.12GB of windowed PBsim reads from 5 phyla, with simulated sequencing error --
a 48x difference. That comparison measures the database, not the method.

Here the database for subset i is built from subset_i.csv itself: the same
windows, the same 5 phyla, the same simulated error. Anything Kraken2 achieves
over the neural scores is then attributable to exact k-mer matching rather than
to having more or cleaner data.

Sequences are labelled with their phylum taxid, which is the models' prediction
target; for a hit/no-hit novelty score the label granularity is irrelevant
anyway.
"""
from __future__ import annotations
import argparse, subprocess, shutil
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

KRAKEN = "/home/glh52/miniconda3/envs/kraken2/bin"
TAXO = Path("/scratch/home/glh52/kraken2_db_train/taxonomy")   # built by kraken2_baseline.py


def kmer_frac(db: Path, seqs, tag: str, tmp: Path, threads: int):
    fa = tmp / f"{tag}.fa"
    with fa.open("w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{tag}{i}\n{s}\n")
    out = tmp / f"{tag}.kraken"
    subprocess.run([f"{KRAKEN}/kraken2", "--db", str(db), "--threads", str(threads),
                    "--output", str(out), "--report", str(tmp / f"{tag}.rep"), str(fa)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cls, frac = [], []
    for line in out.open():
        p = line.rstrip("\n").split("\t")
        cls.append(p[0] == "C")
        hit = tot = 0
        for ch in p[4].split():
            t, n = ch.rsplit(":", 1)
            if n.isdigit():
                tot += int(n)
                if t not in ("0", "A"):
                    hit += int(n)
        frac.append(hit / tot if tot else 0.0)
    return np.array(cls), np.array(frac)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--v", default="v3", choices=["v2", "v3"])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--work", type=Path, default=Path("/scratch/home/glh52/kraken2_parity"))
    args = ap.parse_args()

    root = here.parent / f"leave_phyla_out_{args.v}"
    L = "6000bp" if args.v == "v3" else "150bp"
    design = pd.read_csv(root / "design.csv")
    tiers = {t: pd.read_csv(root / f"eval_{t}_{L}.csv") for t in ["id_test", "near", "far"]}
    for d in tiers.values():
        d["true_phylum"] = d.true_phylum.astype(str)
    args.work.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(10):
        phyla = design.loc[design.subset == i, "phyla"].iloc[0].split("|")
        tr = pd.read_csv(root / "subsets" / f"subset_{i}.csv")
        db = args.work / f"db_{args.v}_{i}"
        if db.exists():
            shutil.rmtree(db)
        (db / "taxonomy").mkdir(parents=True)
        for f in ("nodes.dmp", "names.dmp"):
            shutil.copy2(TAXO / f, db / "taxonomy" / f)
        lib = db / "lib.fna"
        with lib.open("w") as f:
            for j, (s, p) in enumerate(zip(tr.sequence, tr.phylum.astype(str))):
                f.write(f">w{j}|kraken:taxid|{p}\n{s}\n")
        subprocess.run([f"{KRAKEN}/kraken2-build", "--add-to-library", str(lib),
                        "--db", str(db), "--threads", str(args.threads)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([f"{KRAKEN}/kraken2-build", "--build", "--db", str(db),
                        "--threads", str(args.threads)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        tmp = args.work / f"tmp_{i}"; tmp.mkdir(exist_ok=True)
        idt = tiers["id_test"][tiers["id_test"].true_phylum.isin(phyla)]
        near = tiers["near"]; in_lab = near.true_phylum.isin(phyla).values
        c_i, f_i = kmer_frac(db, idt.sequence.values, "idt", tmp, args.threads)
        c_n, f_n = kmer_frac(db, near.sequence.values, "near", tmp, args.threads)
        c_f, f_f = kmer_frac(db, tiers["far"].sequence.values, "far", tmp, args.threads)

        def auc(a, b):
            return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], -np.r_[a, b])

        r = dict(subset=i, cls_idtest=c_i.mean(), cls_nearid=c_n[in_lab].mean(),
                 cls_nearood=c_n[~in_lab].mean(), cls_far=c_f.mean(),
                 idtest_v_nearid=auc(f_i, f_n[in_lab]),
                 idtest_v_nearood=auc(f_i, f_n[~in_lab]),
                 idtest_v_farood=auc(f_i, f_f),
                 nearid_v_nearood=auc(f_n[in_lab], f_n[~in_lab]))
        rows.append(r)
        print(f"subset_{i}: classified idtest={r['cls_idtest']:.3f} nearID={r['cls_nearid']:.3f} "
              f"nearOOD={r['cls_nearood']:.3f} far={r['cls_far']:.3f} | "
              f"AUROC ID/nearOOD={r['idtest_v_nearood']:.3f} ID/far={r['idtest_v_farood']:.3f}",
              flush=True)
        shutil.rmtree(db, ignore_errors=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"results/kraken2_parity_{args.v}.csv", index=False)
    print("\nmean:")
    print(df.drop(columns=["subset"]).mean().to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
