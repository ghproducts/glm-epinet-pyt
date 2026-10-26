#!/usr/bin/env python3
"""Kraken2 and MMseqs2 on the Zymo windows (as sequenced), scored like data_gen/clean_fragments/tools_clean1500.py:
same training-genome database, prediction at family / phylum (abstain when unclassified or above the rank), raw
confidence and the isotonic calibration fitted on clean1500 val.csv. Both tools search both strands, so only the
as-sequenced windows are used.

Writes results/zymo/tools_<rank>.parquet: one row per (window, tool variant) with y, pred, conf and detection scores.

    python data_gen/zymo/tools_zymo.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clean_fragments"))
import tools_clean1500 as T  # noqa: E402
import external_common as X  # noqa: E402

Z = Path("/scratch/home/glh52/glm-epinet/DATA/zymo/zymo_windows.csv")
OUT = Path("/scratch/home/glh52/glm-epinet-pyt/results/zymo"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    d = pd.read_csv(Z)
    fa = T.WORK / "zymo.fa"
    if not fa.exists():
        with open(fa, "w") as f:
            for i, s in enumerate(d.sequence): f.write(f">q{i}\n{s}\n")
    if not (T.WORK / "zymo.kraken").exists():
        subprocess.run([T.KRAKEN, "--db", str(T.KDB), "--threads", T.THREADS, "--output", str(T.WORK / "zymo.kraken"), str(fa)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tdb = T.WORK / "mm_target/train"
    if not (tdb.parent / "done").exists():
        tdb.parent.mkdir(exist_ok=True)
        subprocess.run([T.MMSEQS, "createdb", str(T.KDB / "library/train_regions.fna"), str(tdb), "--dbtype", "2"], check=True)
        subprocess.run([T.MMSEQS, "createindex", str(tdb), str(T.WORK / "mm_tmp_idx"), "--search-type", "3", "--threads", T.THREADS], check=True)
        (tdb.parent / "done").touch()
    if not (T.WORK / "zymo.m8").exists():
        q = T.WORK / "mm_q_zymo/q"; q.parent.mkdir(exist_ok=True); r = T.WORK / "mm_res_zymo/r"; r.parent.mkdir(exist_ok=True)
        subprocess.run([T.MMSEQS, "createdb", str(fa), str(q), "--dbtype", "2"], check=True)
        subprocess.run([T.MMSEQS, "search", str(q), str(tdb), str(r), str(T.WORK / "mm_tmp_zymo"), "--search-type", "3",
                        "--threads", T.THREADS, "--max-seqs", "50"], check=True)
        subprocess.run([T.MMSEQS, "convertalis", str(q), str(tdb), str(r), str(T.WORK / "zymo.m8"), "--search-type", "3",
                        "--format-output", "query,target,fident,alnlen,qcov,bits,evalue"], check=True)
    parent, rank_of = T.taxonomy(); lin = X.lineage()
    k = T.parse_kraken("zymo", parent); mm = T.parse_mmseqs("zymo", len(d))
    kv = T.parse_kraken("val", parent); val = pd.read_csv(X.D / "val.csv", usecols=["taxid"]); mv = T.parse_mmseqs("val", len(val))
    for rank in ("family", "phylum"):
        lab2id = X.vocab(rank); C = len(lab2id); sp2 = dict(zip(lin.species.astype(int), lin[rank].astype(int)))
        y = np.array([lab2id.get(int(t), -1) for t in d[rank]])
        vy = np.array([lab2id.get(sp2.get(int(t), -1), -1) for t in val.taxid])
        kp = lambda kk: np.array([lab2id.get(T.ancestor(t, rank, parent, rank_of), -2) if t > 0 else -2 for t in kk.taxid])
        mp = lambda m: np.array([lab2id.get(sp2.get(s, -1), -2) if s > 0 else -2 for s in m.species])
        frames = []
        for name, pred, conf, vpred, vconf, det in [
                ("Kraken2", kp(k), k.conf.values, kp(kv), kv.conf.values,
                 {"nohit fraction": k.nohit.values, "unclassified": k.unclassified.values}),
                ("MMseqs2", mp(mm), (mm.fident * mm.qcov).values, mp(mv), (mv.fident * mv.qcov).values,
                 {"-bitscore": -mm.bits.values, "no hit": mm.nohit.values})]:
            keep = vy >= 0
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(vconf[keep], (vpred[keep] == vy[keep]).astype(float))
            for var, c in (("raw score", conf), ("isotonic on val", iso.predict(conf))):
                f = X.rows_from_label(f"{name} ({var})", y, pred, c, C, det)
                frames.append(pd.concat([d[["read", "window", "species", "strand"]].reset_index(drop=True), f], axis=1))
            print(rank, name, "abstain", round(float(np.mean(pred < 0)), 3), "acc (labeled)", round(float(np.mean(pred[y >= 0] == y[y >= 0])), 3), flush=True)
        pd.concat(frames, ignore_index=True).to_parquet(OUT / f"tools_{rank}.parquet", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
