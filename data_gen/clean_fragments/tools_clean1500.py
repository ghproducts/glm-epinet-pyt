#!/usr/bin/env python3
"""Kraken2 and MMseqs2 as classifiers on the clean1500 benchmark (phylum and family), scored like the GLMs.

Reviewer 2 (point 8): alignment / k-mer scores are not probabilities, so a raw-score ECE comparison is
close to a tautology. Each tool is therefore reported (a) on its raw confidence and (b) after an
isotonic calibration fitted on val.csv (held-out genome blocks, the same data the GLM temperature is
fitted on), and is compared on ranking metrics (AURC, selective accuracy, misclassification AUROC)
that need no probabilistic interpretation.

Reference database for both tools = the TRAINING genome regions only (export_train_regions.py,
/scratch/home/glh52/kraken2_db_clean1500/library/train_regions.fna): the same sequence the GLMs
trained on, nothing from the val / id_test blocks.

Kraken2  prediction = the assigned taxon's ancestor at the rank; abstains when unclassified or when the
         LCA sits above the rank. Raw confidence = Kraken2's own confidence score (k-mers in the
         assigned clade / unambiguous k-mers). Detection scores: 1 - confidence, fraction of k-mers
         with no database hit, unclassified.
MMseqs2  nucleotide search (--search-type 3) against the same regions; prediction = the best-bitscore
         hit's species' ancestor at the rank; abstains without a hit. Raw confidence = identity x query
         coverage. Detection scores: 1 - identity x coverage, -bitscore, no hit.

    python data_gen/clean_fragments/tools_clean1500.py [--stage kraken|mmseqs|score]
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_common as X  # noqa: E402

KRAKEN = "/home/glh52/miniconda3/envs/kraken2/bin/kraken2"
KDB = Path("/scratch/home/glh52/kraken2_db_clean1500")
MMSEQS = "/home/glh52/opt/mmseqs/bin/mmseqs"
WORK = Path("/scratch/home/glh52/tools_clean1500"); WORK.mkdir(exist_ok=True)
THREADS = "96"
SETS = ["val"] + X.TIERS + ["darkmatter"]


def fasta(name):
    fa = WORK / f"{name}.fa"
    if not fa.exists():
        d = pd.read_csv(X.D / ("darkmatter_contig_windows.csv" if name == "darkmatter" else f"{name}.csv"), usecols=["sequence"])
        with open(fa, "w") as f:
            for i, s in enumerate(d.sequence): f.write(f">q{i}\n{s}\n")
    return fa


def run_kraken():
    for n in SETS:
        out = WORK / f"{n}.kraken"
        if out.exists(): continue
        subprocess.run([KRAKEN, "--db", str(KDB), "--threads", THREADS, "--output", str(out), str(fasta(n))],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("kraken2 done", n, flush=True)


def run_mmseqs():
    tdb = WORK / "mm_target/train"
    if not (tdb.parent / "done").exists():
        tdb.parent.mkdir(exist_ok=True)
        subprocess.run([MMSEQS, "createdb", str(KDB / "library/train_regions.fna"), str(tdb), "--dbtype", "2"], check=True)
        subprocess.run([MMSEQS, "createindex", str(tdb), str(WORK / "mm_tmp_idx"), "--search-type", "3", "--threads", THREADS], check=True)
        (tdb.parent / "done").touch()
    for n in SETS:
        m8 = WORK / f"{n}.m8"
        if m8.exists(): continue
        q = WORK / f"mm_q_{n}/q"; q.parent.mkdir(exist_ok=True); res = WORK / f"mm_res_{n}/r"; res.parent.mkdir(exist_ok=True)
        subprocess.run([MMSEQS, "createdb", str(fasta(n)), str(q), "--dbtype", "2"], check=True)
        subprocess.run([MMSEQS, "search", str(q), str(tdb), str(res), str(WORK / f"mm_tmp_{n}"), "--search-type", "3",
                        "--threads", THREADS, "--max-seqs", "50"], check=True)
        subprocess.run([MMSEQS, "convertalis", str(q), str(tdb), str(res), str(m8) + ".tmp", "--search-type", "3",
                        "--format-output", "query,target,fident,alnlen,qcov,bits,evalue"], check=True)
        Path(str(m8) + ".tmp").rename(m8)
        print("mmseqs2 done", n, flush=True)


def taxonomy():
    nodes = pd.read_csv(KDB / "taxonomy/nodes.dmp", sep="\t\\|\t", engine="python", header=None, usecols=[0, 1, 2])
    parent = dict(zip(nodes[0].astype(int), nodes[1].astype(int))); rank = dict(zip(nodes[0].astype(int), nodes[2].str.strip()))
    return parent, rank


def ancestor(t, want, parent, rank):
    """ancestor of t at rank `want`, or -1 if t sits above that rank"""
    seen = 0
    while t in parent and seen < 50:
        if rank.get(t) == want: return t
        if t == parent[t]: break
        t = parent[t]; seen += 1
    return -1


def parse_kraken(n, parent):
    k = pd.read_csv(WORK / f"{n}.kraken", sep="\t", header=None, usecols=[0, 2, 4], names=["c", "taxid", "kmers"])
    anc = {}
    def lineage_set(t):
        if t not in anc:
            s, u = set(), t
            while u in parent and u not in s:
                s.add(u)
                if parent[u] == u: break
                u = parent[u]
            anc[t] = s
        return anc[t]
    conf, nohit = np.zeros(len(k)), np.zeros(len(k))
    for i, (a, s) in enumerate(zip(k.taxid.values, k.kmers.values)):
        tot = hit = clade = 0
        for tok in str(s).split():
            t, c = tok.split(":"); c = int(c)
            if t == "A": continue
            tot += c; t = int(t)
            if t != 0:
                hit += c
                if a != 0 and a in lineage_set(t): clade += c      # k-mer's taxon lies in the assigned clade
        conf[i] = clade / tot if tot else 0.0; nohit[i] = 1 - hit / tot if tot else 1.0
    return pd.DataFrame(dict(taxid=k.taxid.values.astype(int), unclassified=(k.c == "U").values.astype(float), conf=conf, nohit=nohit))


def parse_mmseqs(n, N):
    m = pd.read_csv(WORK / f"{n}.m8", sep="\t", header=None, names=["q", "t", "fident", "alnlen", "qcov", "bits", "evalue"])
    m = m.sort_values("bits", ascending=False).drop_duplicates("q")
    m["qi"] = m.q.str[1:].astype(int); m["species"] = m.t.str.split("_").str[0].str[1:].astype(int)
    out = pd.DataFrame(dict(species=-1, fident=0.0, qcov=0.0, bits=0.0), index=range(N))
    out.loc[m.qi.values, ["species", "fident", "qcov", "bits"]] = m[["species", "fident", "qcov", "bits"]].values
    out["species"] = out.species.astype(int); out["nohit"] = (out.species < 0).astype(float)
    return out


def score():
    parent, rank_of = taxonomy()
    lin = X.lineage()
    sp2 = {r: dict(zip(lin.species.astype(int), lin[r].astype(int))) for r in ("phylum", "family")}
    for rank in ("phylum", "family"):
        lab2id = X.vocab(rank); C = len(lab2id)
        sets = X.eval_sets(rank)
        val = pd.read_csv(X.D / "val.csv", usecols=["taxid"]); val["y"] = [lab2id.get(sp2[rank].get(int(t), -1), -1) for t in val.taxid]
        raw = {"kraken2": {}, "mmseqs2": {}}
        for n in ["val"] + list(sets):
            N = len(val) if n == "val" else len(sets[n])
            k = parse_kraken(n, parent)
            kp = np.array([lab2id.get(ancestor(t, rank, parent, rank_of), -2) if t > 0 else -2 for t in k.taxid])
            raw["kraken2"][n] = (kp, k.conf.values, {"nohit fraction": k.nohit.values, "unclassified": k.unclassified.values,
                                                      "1 - confidence": 1 - k.conf.values})
            mm = parse_mmseqs(n, N)
            mp = np.array([lab2id.get(sp2[rank].get(s, -1), -2) if s > 0 else -2 for s in mm.species])
            c = (mm.fident * mm.qcov).values
            raw["mmseqs2"][n] = (mp, c, {"1 - identity x coverage": 1 - c, "-bitscore": -mm.bits.values, "no hit": mm.nohit.values})
            print(f"  {rank} {n}: kraken2 abstain {np.mean(kp < 0):.3f}, mmseqs2 no hit {mm.nohit.mean():.3f}", flush=True)
        for tool, R in raw.items():
            vp, vc, _ = R["val"]; vy = val.y.values; keep = vy >= 0
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(vc[keep], (vp[keep] == vy[keep]).astype(float))
            frames, name = {}, "Kraken2" if tool == "kraken2" else "MMseqs2"
            for n in sets:
                p, c, det = R[n]; y = sets[n].y.values
                a = X.rows_from_label(f"{name} (raw score)", y, p, c, C, det)
                b = X.rows_from_label(f"{name} (isotonic on val)", y, p, iso.predict(c), C, det)
                frames[n] = pd.concat([a, b], ignore_index=True)
            dm, idt = frames.pop("darkmatter"), frames["id_test"]
            dm_meta = sets["darkmatter"][["up", "sample"]].reset_index(drop=True)
            dmr = []
            for meth in dm.method.unique():
                d1 = pd.concat([dm_meta, dm[dm.method == meth].reset_index(drop=True)], axis=1); i1 = idt[idt.method == meth]
                dmr += [dict(r, backbone=f"{meth}") for r in X.darkmatter_rows(name, rank, d1, i1, [c for c in d1 if c.startswith("det_")])]
            X.write(tool, rank, frames, dmr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all", choices=["all", "kraken", "mmseqs", "score"])
    s = ap.parse_args().stage
    if s in ("all", "kraken"): run_kraken()
    if s in ("all", "mmseqs"): run_mmseqs()
    if s in ("all", "score"): score()
