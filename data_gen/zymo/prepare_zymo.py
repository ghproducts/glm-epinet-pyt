#!/usr/bin/env python3
"""ZymoBIOMICS Microbial Community Standard (even, D6300), nanopore GridION run ERR3152364
(Nicholls et al. 2019, GigaScience; ENA PRJEB29504), as a real-data test of the clean1500 models.

1. random subsample of reads >= 1.5 kb (seed 1)
2. ground truth: minimap2 -x map-ont against the Zymo reference genomes (ZymoBIOMICS.STD.refseq.v2);
   a read is assigned to a species when its best primary alignment has MAPQ >= 20 and covers >= 50%
   of the read; otherwise 'unassigned'
3. non-overlapping 1.5 kb windows from the start of each read (the clean1500 input length), each window
   carrying its read's label; written twice: as sequenced, and reverse-complemented where the read maps
   to the reverse strand ('oriented', a diagnostic: clean1500 training fragments are forward-strand only)

Labels: NCBI species / family / phylum taxids of the 10 organisms. The two yeasts have no label in the
bacterial training vocabulary (OOD); E. coli and L. fermentum belong to genera held out of training.

    python data_gen/zymo/prepare_zymo.py --n-reads 150000
"""
from __future__ import annotations
import argparse, gzip, random, subprocess
from pathlib import Path
import pandas as pd

W = Path("/scratch/home/glh52/zymo")
MINIMAP2 = "/home/glh52/.claude/jobs/dd626958/tmp/minimap2"
FQ = W / "ERR3152364.fastq.gz"; REF = W / "zymo_ref.fa"
OUT = Path("/scratch/home/glh52/glm-epinet/DATA/zymo")
WIN = 1500
# species, genus, family, phylum NCBI taxids; kingdom-level role in this test
ORG = {
    "Listeria_monocytogenes": (1639, 1637, 186820, 1239, "bacteria"), "Pseudomonas_aeruginosa": (287, 286, 135621, 1224, "bacteria"),
    "Bacillus_subtilis": (1423, 1386, 186817, 1239, "bacteria"), "Escherichia_coli": (562, 561, 543, 1224, "bacteria"),
    "Salmonella_enterica": (28901, 590, 543, 1224, "bacteria"), "Lactobacillus_fermentum": (1613, 2742598, 33958, 1239, "bacteria"),
    "Enterococcus_faecalis": (1351, 1350, 81852, 1239, "bacteria"), "Staphylococcus_aureus": (1280, 1279, 90964, 1239, "bacteria"),
    "Saccharomyces_cerevisiae": (4932, 4930, 4893, 4890, "yeast"), "Cryptococcus_neoformans": (5207, 5206, 1884633, 5204, "yeast"),
}
RC = str.maketrans("ACGTacgtN", "TGCAtgcaN")


def subsample(n, p_keep, out):
    rng = random.Random(1); kept = 0
    with gzip.open(FQ, "rt") as f, open(out, "w") as o:
        while True:
            h = f.readline()
            if not h: break
            s, _, q = f.readline(), f.readline(), f.readline()
            if len(s) - 1 >= WIN and rng.random() < p_keep:
                o.write(f"{h}{s}+\n{q}"); kept += 1
                if kept >= n: break
    return kept


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-reads", type=int, default=150_000)
    ap.add_argument("--p-keep", type=float, default=0.05, help="per-read keep probability (3.49M reads in the run)")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    sub = W / "sub.fq"
    if not sub.exists(): print("subsampled", subsample(a.n_reads, a.p_keep, sub), "reads", flush=True)
    paf = W / "sub.paf"
    if not paf.exists():
        subprocess.run(f"{MINIMAP2} -x map-ont -t 96 --secondary=no {REF} {sub} > {paf}.tmp && mv {paf}.tmp {paf}", shell=True, check=True)
    cols = ["q", "qlen", "qs", "qe", "strand", "t", "tlen", "ts", "te", "match", "alen", "mapq"]
    m = pd.read_csv(paf, sep="\t", header=None, usecols=range(12), names=cols)
    m["qcov"] = (m.qe - m.qs) / m.qlen
    m = m.sort_values("match", ascending=False).drop_duplicates("q")
    m["species"] = m.t.str.split("|").str[0]
    ok = (m.mapq >= 20) & (m.qcov >= 0.5)
    lab = dict(zip(m.q[ok], m.species[ok])); strand = dict(zip(m.q[ok], m.strand[ok]))
    rows = []
    with open(sub) as f:
        while True:
            h = f.readline()
            if not h: break
            s = f.readline().strip(); f.readline(); f.readline()
            rid = h[1:].split()[0]; sp = lab.get(rid, "unassigned"); st = strand.get(rid, ".")
            for w in range(len(s) // WIN):
                seq = s[w * WIN:(w + 1) * WIN]
                o = seq.translate(RC)[::-1] if st == "-" else seq
                t = ORG.get(sp, (-1, -1, -1, -1, "unassigned"))
                rows.append(dict(read=rid, window=w, read_len=len(s), species=sp, strand=st, taxid=t[0], genus=t[1], family=t[2],
                                 phylum=t[3], group=t[4], sequence=seq, sequence_oriented=o))
    d = pd.DataFrame(rows)
    print(d.groupby("species").agg(reads=("read", "nunique"), windows=("read", "size")).assign(
        read_frac=lambda x: x.reads / x.reads.sum()).round(4).to_string(), flush=True)
    d.drop(columns="sequence_oriented").to_csv(OUT / "zymo_windows.csv", index=False)
    d.drop(columns="sequence").rename(columns={"sequence_oriented": "sequence"}).to_csv(OUT / "zymo_windows_oriented.csv", index=False)
    print("wrote", OUT, len(d), "windows", flush=True)


if __name__ == "__main__":
    main()
