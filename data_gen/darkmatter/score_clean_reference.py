#!/usr/bin/env python3
"""Rescore dark matter against an ERROR-FREE reference.

clean_vs_noisy.py showed the noisy-read reference (PBSIM3 QSHMM-RSII, ~85%
accurate, indel-dominated) makes error-free sequence look novel on its own:
clean windows of the TRAINING species score AUROC 0.63 against noisy id_test at
6kb, about as far as a genuinely novel family. Dark matter reads (ART Illumina)
and contigs (consensus) are effectively error-free, so every earlier dark matter
score carried that technology shift.

Fix without retraining: build the kNN reference AND the ID null from clean
windows cut directly from the training species' genome FASTAs, so the query and
the reference share an error profile. Species are drawn in proportion to genome
size, mirroring the depth-1 read reference. Reference and null windows are
independent draws.
"""
from __future__ import annotations
import argparse, gzip, re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_contigs import read_contigs, load_encoder, embed, ASM   # noqa: E402

G = Path("data_gen/pbsim/raw_sequences"); S = Path("data_gen/pbsim/splits")
TMP = Path("/scratch/home/glh52/kraken2_tmp_reads")
FPRS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def clean_windows(n, W, rng):
    """n error-free W-bp windows from training-species genomes, species weighted by genome size."""
    sp = [x.strip() for x in open(S / "train_species.txt") if x.strip()]
    files = {s: list((G / s).glob("*.fna*")) for s in sp}
    files = {s: f for s, f in files.items() if f}
    size = np.array([sum(p.stat().st_size for p in files[s]) for s in files], float)
    keys = list(files)
    picks = pd.Series(rng.choice(len(keys), size=n, p=size / size.sum())).value_counts()
    out = []
    for i, k in picks.items():
        seqs, cur = [], []
        for f in files[keys[i]]:
            with (gzip.open if f.suffix == ".gz" else open)(f, "rt") as h:
                for l in h:
                    if l.startswith(">"):
                        if cur: seqs.append("".join(cur)); cur = []
                    else: cur.append(l.strip().upper())
        if cur: seqs.append("".join(cur))
        seqs = [s for s in seqs if len(s) >= W]
        if not seqs: continue
        L = np.array([len(s) for s in seqs], float)
        for _ in range(k):
            s = seqs[rng.choice(len(seqs), p=L / L.sum())]
            st = int(rng.integers(0, len(s) - W + 1)); w = s[st:st + W]
            if w.count("N") < 0.01 * W: out.append(w)
    rng.shuffle(out)
    return out


def trend(label, rep):
    lo, hi = rep[rep.up == rep.up.min()].v, rep[rep.up == rep.up.max()].v
    sl, _, r, p, _ = stats.linregress(rep.up, rep.v)
    d = (hi.mean() - lo.mean()) / np.sqrt((lo.var(ddof=1) + hi.var(ddof=1)) / 2)
    print(f"  {label:30s} slope {sl:+.4f}  R2 {r*r:.3f}  d {d:+6.2f}  p {p:.2e}  means "
          + " ".join(f"{x:.4f}" for x in rep.groupby('up').v.mean()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["nt-phylum", "metagene1"])
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--target", required=True, choices=["contigs", "reads"])
    ap.add_argument("--n-ref", type=int, default=40000)
    ap.add_argument("--n-null", type=int, default=5000)
    a = ap.parse_args(); W = a.window
    rng = np.random.default_rng(7)

    if a.target == "contigs":
        rows = []
        for d in sorted(ASM.glob("up*_s*")):
            up = float(re.search(r"up([0-9.]+)_", d.name).group(1)); s = int(d.name.split("_s")[1])
            seqs, cov = read_contigs(d / "asm" / "final.contigs.fa")
            for x, c in zip(seqs, cov):
                if len(x) >= W:
                    st = int(rng.integers(0, len(x) - W + 1)); rows.append((up, s, x[st:st + W], c, len(x)))
        q = pd.DataFrame(rows, columns=["up", "sample", "seq", "cov", "len"])
    else:
        q = pd.read_csv("data_gen/darkmatter/csv_data/darkmatter_reads_all.csv")
        q = q[q.up != 0.5].reset_index(drop=True)[["up", "sample", "sequence"]].rename(columns={"sequence": "seq"})
    print(f"{a.encoder} {W}bp {a.target}: {q.groupby('up').size().to_dict()}", flush=True)

    win = clean_windows(a.n_ref + a.n_null, W, rng)
    ref, null = win[:a.n_ref], win[a.n_ref:]
    print(f"clean reference {len(ref)} windows, clean ID null {len(null)}", flush=True)

    m, tok, ml, is_nt = load_encoder(a.encoder, W)
    if is_nt: ml = W // 6 + 8
    bs = (max(4, int(64 * 2000 / W)) if W > 300 else 384) if is_nt else (32 if W > 300 else 128)
    R = embed(m, tok, ref, ml, bs, is_nt, "ref").cuda()
    knn = lambda E: np.concatenate([(-(E[i:i+2048].cuda() @ R.T).max(1).values).cpu().numpy()
                                    for i in range(0, len(E), 2048)])
    si = knn(embed(m, tok, null, ml, bs, is_nt, "null"))
    q["v"] = knn(embed(m, tok, list(q.seq), ml, bs, is_nt, a.target))
    tag = f"{a.encoder}_W{W}_{a.target}"
    q.drop(columns="seq").to_csv(f"results/cleanref_{tag}.csv", index=False)
    np.save(f"results/cleanref_{tag}_null.npy", si)

    print(f"\n=== CLEAN reference | {a.encoder} | {W}bp | {a.target} ===")
    print("  AUROC clean-ID null vs dark matter: " + "  ".join(
        f"up={u}: {roc_auc_score(np.r_[np.zeros(len(si)), np.ones((q.up==u).sum())], np.r_[si, q.v[q.up==u]]):.4f}"
        for u in sorted(q.up.unique())))
    trend("mean novelty", q.groupby(["up", "sample"]).v.mean().reset_index())
    for f in FPRS:
        T = np.quantile(si, 1 - f)
        trend(f"flagged @ {f:.1%} FPR", q.assign(v=(q.v > T).astype(float)).groupby(["up", "sample"]).v.mean().reset_index())

    if a.target == "contigs":     # do Kraken2-matched (training-species) contigs now look familiar?
        ks = []
        for f in sorted(TMP.glob("contigs_up*_s*.kraken")):
            k = pd.read_csv(f, sep="\t", header=None, usecols=[0, 3], names=["c", "len"]); k = k[k["len"] >= W]
            mm = re.search(r"up([0-9.]+)_s(\d+)", f.name)
            ks.append(pd.DataFrame({"up": float(mm[1]), "sample": int(mm[2]), "kc": (k.c == "C").values}))
        k = pd.concat(ks).sort_values(["up", "sample"], kind="stable").reset_index(drop=True)
        qq = q.sort_values(["up", "sample"], kind="stable").reset_index(drop=True)
        if len(k) == len(qq):
            kc = k.kc.values
            print(f"  Kraken2-matched contigs: AUROC(unmatched more novel) = "
                  f"{roc_auc_score((~kc).astype(int), qq.v):.3f}   (>0.5 = matched look familiar = correct)")


if __name__ == "__main__":
    main()
