#!/usr/bin/env python3
"""Non-pretrained baselines for the leave-phyla-out OOD test: k-mer + Random
Forest, and a from-scratch 1D-CNN.

Every method tried so far (softmax, energy, Mahalanobis, kNN, outlier exposure,
deep ensembles) sits at ~0.5 AUROC, but all of them share one fine-tuned
pretrained GLM backbone. These two share none of it, which separates two very
different conclusions:

  * if these also fail  -> the task is intrinsically hard at 150bp, and the
    result is about read-level taxonomic novelty rather than about GLMs;
  * if these succeed    -> the GLM representation is specifically the problem.

Both give per-sample K x C probability stacks for free -- the RF from its
per-tree votes (a Random Forest is already an ensemble), the CNN from MC
dropout -- so the same uncertainty decompositions apply without modification.
Reuses SmallCNN/one_hot_encode from data_gen/label_noise/cnn_scratch.py.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "label_noise"))
import cnn_scratch
from cnn_scratch import SmallCNN

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def bald(P):
    """P: [K, N, C] -> (total, aleatoric, epistemic) entropies."""
    pbar = P.mean(0)
    ent = lambda q: -(q * np.log(q + 1e-12)).sum(-1)
    tot = ent(pbar); alea = ent(P).mean(0)
    return tot, alea, tot - alea


def run_rf(tr, ev_seqs, y, k=6, n_trees=200, n_train=20000, seed=1):
    tr = tr.sample(min(len(tr), n_train), random_state=seed)
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
    Xtr = vec.fit_transform(tr.sequence.values)
    Xev = vec.transform(ev_seqs)
    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, tr.labels.values)
    P = np.stack([t.predict_proba(Xev.toarray() if hasattr(Xev, "toarray") else Xev)
                  for t in rf.estimators_])          # [n_trees, N, C]
    tot, alea, epi = bald(P)
    pbar = P.mean(0)
    return {"rf_msp": roc_auc_score(y, -pbar.max(1)),
            "rf_entropy": roc_auc_score(y, tot),
            "rf_epistemic": roc_auc_score(y, epi)}


def run_cnn(tr, ev_seqs, y, n_classes=5, epochs=12, k_mc=10, seed=1, bs=256):
    cnn_scratch.SEQ_LEN = 150
    Xtr = cnn_scratch.one_hot_encode(tr.sequence.values)
    ytr = torch.tensor(tr.labels.values, dtype=torch.long)
    Xev = cnn_scratch.one_hot_encode(ev_seqs)
    torch.manual_seed(seed)
    m = SmallCNN(n_classes=n_classes, dropout_p=0.3).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + ep))
        m.train()
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            F.cross_entropy(m(Xtr[idx].to(DEV)), ytr[idx].to(DEV)).backward()
            opt.step()

    @torch.no_grad()
    def fwd(train_mode):
        m.train(train_mode)          # train_mode=True keeps dropout on -> MC dropout
        out = []
        for s in range(0, len(Xev), 512):
            out.append(torch.softmax(m(Xev[s:s + 512].to(DEV)), -1).cpu().numpy())
        return np.concatenate(out)

    det = fwd(False)
    P = np.stack([fwd(True) for _ in range(k_mc)])
    tot, alea, epi = bald(P)
    return {"cnn_msp": roc_auc_score(y, -det.max(1)),
            "cnn_mcdo_entropy": roc_auc_score(y, tot),
            "cnn_mcdo_epistemic": roc_auc_score(y, epi)}


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--subsets", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--out", type=Path,
                    default=here.parent.parent / "results" / "leave_phyla_out" / "nonpretrained.csv")
    args = ap.parse_args()

    design = pd.read_csv(here / "subsets" / "design.csv")
    ev = pd.read_csv(here / "eval_pbsim_150bp.csv"); ev["true_phylum"] = ev.true_phylum.astype(str)
    rows = []
    for i in args.subsets:
        tr = pd.read_csv(here / "subsets" / f"subset_{i}.csv")
        phyla = design.loc[design.subset == i, "phyla"].iloc[0].split("|")
        y = (~ev.true_phylum.isin(phyla)).astype(int).values
        r = {"subset": i}
        r.update(run_rf(tr, ev.sequence.values, y))
        r.update(run_cnn(tr, ev.sequence.values, y))
        rows.append(r)
        print(f"subset_{i}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "subset"),
              flush=True)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nmean AUROC:")
    print(df.drop(columns=["subset"]).mean().sort_values(ascending=False)
            .to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
