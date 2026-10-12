#!/usr/bin/env python3
"""Model-free detection baselines for gene_taxa, saved so the reported numbers
are reproducible rather than computed inline.

Three statistics that use no learned representation at all:
  length      sequence length
  gc          GC fraction
  kmer6       cosine similarity to the nearest training read in 6-mer space

If the GLM's detection AUROC is not clearly above these, its representation is
adding nothing over surface composition. Query sets are the capped tier copies
written by scripts/eval_gene_taxa.sh, so these line up with the model numbers.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

SRC = Path("/scratch/home/glh52/glm-epinet/DATA/gene_taxa")
CAP = Path("temp/gene_taxa_eval")
N_REF, N_Q, SEED = 30_000, 2_000, 1
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gc(s):
    s = s.upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def main():
    tr = pd.read_csv(SRC / "train.csv", usecols=["sequence"]).sample(N_REF, random_state=SEED).sequence
    vec = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    X = torch.tensor(vec.fit_transform(tr).astype(np.float32).toarray())
    X = torch.nn.functional.normalize(X, dim=1).to(DEV)

    S = {}
    for t in ["test", "taxa_out", "gene_out"]:
        q = pd.read_csv(CAP / f"{t}.csv", usecols=["sequence"]).sample(N_Q, random_state=SEED).sequence
        Q = torch.tensor(vec.transform(q).astype(np.float32).toarray())
        Q = torch.nn.functional.normalize(Q, dim=1).to(DEV)
        sim = torch.cat([(Q[i:i + 256] @ X.T).max(1).values.cpu() for i in range(0, len(Q), 256)]).numpy()
        S[t] = {"length": q.str.len().values.astype(float),
                "gc": q.map(gc).values,
                "kmer6": -sim}          # higher = more novel

    def auc(a, b):
        s = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
        return max(s, 1 - s)            # direction-free: these are baselines, not detectors

    rows = [{"method": m,
             "test_v_taxa_out": auc(S["test"][m], S["taxa_out"][m]),
             "test_v_gene_out": auc(S["test"][m], S["gene_out"][m])}
            for m in ["length", "gc", "kmer6"]]
    df = pd.DataFrame(rows)
    df.to_csv("results/gene_taxa_baselines.csv", index=False)
    np.savez_compressed("results/gene_taxa_baselines_scores.npz",
                        **{f"{t}__{m}": v for t, d in S.items() for m, v in d.items()})
    print(f"reference {N_REF:,} train reads | {N_Q:,} queries per tier")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nraw 6-mer similarity to nearest training read (redundancy check):")
    for t in S:
        s = -S[t]["kmer6"]
        print(f"  {t:9s} mean {s.mean():.4f}  median {np.median(s):.4f}  frac>0.95 {(s>0.95).mean():.4f}")


if __name__ == "__main__":
    main()
