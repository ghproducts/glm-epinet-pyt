#!/usr/bin/env python3
"""Kraken2 baseline on the clean1500 benchmark and the dark matter contig windows.

Database = the TRAINING genome regions only (export_train_regions.py), i.e. the
same sequence the GLMs trained on, nothing from val/id_test blocks.
Two novelty scores per fragment:
  unclassified   binary (Kraken2's native decision)
  1 - hit_frac   fraction of k-mers WITHOUT a database hit (continuous)
"""
import subprocess, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
K = "/home/glh52/miniconda3/envs/kraken2/bin/kraken2"; DB = "/scratch/home/glh52/kraken2_db_clean1500"
D = Path("/scratch/home/glh52/glm-epinet/DATA/clean1500"); TMP = Path("/scratch/home/glh52/kraken2_tmp_clean1500"); TMP.mkdir(exist_ok=True)
OUT = Path("results/clean1500/kraken2"); OUT.mkdir(parents=True, exist_ok=True)
TIERS = ["id_test", "novel_genus", "novel_family", "nonbacterial"]

def classify(name, seqs):
    fa = TMP / f"{name}.fa"; fa.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(seqs)))
    out = TMP / f"{name}.kraken"
    subprocess.run([K, "--db", DB, "--threads", "32", "--output", str(out), str(fa)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    k = pd.read_csv(out, sep="\t", header=None, usecols=[0, 2, 4], names=["c", "taxid", "lca"])
    def nohit(s):
        tot = hit = 0
        for tok in str(s).split():
            t, n = tok.split(":"); n = int(n); tot += n
            if t not in ("0", "A"): hit += n
        return 1 - hit / tot if tot else 1.0
    return pd.DataFrame({"unclassified": (k.c == "U").astype(float).values,
                         "nohit_frac": k.lca.map(nohit).values, "kraken_taxid": k.taxid.values})

S = {}
for t in TIERS:
    for suf in ["", "_err0.5", "_err5"]:
        d = pd.read_csv(D / f"{t}{suf}.csv"); S[t + suf] = classify(t + suf, d.sequence)
        S[t + suf].assign(taxid=d.taxid.values).to_csv(OUT / f"scores_{t}{suf}.csv", index=False)
        print(f"{t+suf:20s} unclassified {S[t+suf].unclassified.mean():.4f}", flush=True)
rows = []
for suf in ["", "_err0.5", "_err5"]:
    for t in TIERS[1:]:
        for s in ["unclassified", "nohit_frac"]:
            a, b = S["id_test" + suf][s].values, S[t + suf][s].values
            rows.append(dict(error=suf or "0", tier=t, score=s,
                             auroc=roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])))
pd.DataFrame(rows).to_csv(OUT / "detection.csv", index=False)
print(pd.DataFrame(rows).pivot_table(index=["error", "tier"], columns="score", values="auroc").round(4).to_string())

dm = pd.read_csv(D / "darkmatter_contig_windows.csv")
r = classify("darkmatter", dm.sequence); dm = pd.concat([dm.drop(columns="sequence"), r], axis=1)
dm.to_csv(OUT / "scores_darkmatter.csv", index=False)
print("\n=== dark matter (1.5kb contig windows) ===")
for s in ["unclassified", "nohit_frac"]:
    rep = dm.groupby(["up", "sample"])[s].mean().reset_index()
    sl, _, rr, p, _ = stats.linregress(rep.up, rep[s])
    lo, hi = rep[rep.up == 0.25][s], rep[rep.up == 1.0][s]
    d_ = (hi.mean() - lo.mean()) / np.sqrt((lo.var(ddof=1) + hi.var(ddof=1)) / 2)
    print(f"  {s:13s} means {rep.groupby('up')[s].mean().round(4).to_dict()}  slope {sl:+.4f} R2 {rr*rr:.3f} d {d_:+.2f} p {p:.1e}")
