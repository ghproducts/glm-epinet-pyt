#!/usr/bin/env python3
"""Non-pretrained baselines on clean1500 (phylum, family) and the Scorpio gene task (reviewer 2, point 7).

Same two families as the promoter sweep (data_gen/uq_sweep/run_baselines.py), scaled to many classes:
  k-mer RF   200-tree random forest on 6-mer frequencies (min_samples_leaf 20); every tree's
             predict_proba is one of K = 200 samples, so it has a total / epistemic split like the
             sampled GLM methods
  CNN        from-scratch DeepBind-style 1D CNN on one-hot DNA (conv 15 -> pool 4 -> conv 9 ->
             global max pool -> MLP, dropout 0.3), Adam lr 1e-3, batch 128, early stopping on
             validation loss (patience 3, cap 30 epochs). Reported as: CNN (seed 1, softmax),
             CNN + MC dropout (seed 1, K = 10, as the GLMs) and CNN ensemble (seeds 1-5).
Training data = the GLMs' train.csv; validation for early stopping = val.csv (clean1500: held-out
genome blocks) or a stratified 10% of train.csv (gene_taxa). Gene sequences are cut to their first
3,000 bp, the longest input any GLM saw.

    python data_gen/clean_fragments/baselines_clean1500.py --rank family --model rf
    python data_gen/clean_fragments/baselines_clean1500.py --rank family --model cnn --gpu 3
"""
from __future__ import annotations
import argparse, itertools, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_common as X  # noqa: E402

KMER = 6
ALPH = {"A": 0, "C": 1, "G": 2, "T": 3}
T0 = time.time()
def log(m): print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)


def train_data(rank):
    lab2id = X.vocab(rank)
    if rank == "gene_taxa":
        tr = pd.read_csv(X.GENE_D / "train.csv", usecols=["sequence", "labels"])
        tr["y"] = [lab2id.get(int(l), -1) for l in tr["labels"]]
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(tr, test_size=0.1, random_state=1, stratify=tr["y"])
    else:
        lin = X.lineage(); s2r = dict(zip(lin.species.astype(int), lin[rank].astype(int)))
        tr = pd.read_csv(X.D / "train.csv", usecols=["sequence", "taxid"]); va = pd.read_csv(X.D / "val.csv", usecols=["sequence", "taxid"])
        for d in (tr, va): d["y"] = [lab2id.get(s2r.get(int(t), -1), -1) for t in d.taxid]
    tr, va = tr[tr.y >= 0].reset_index(drop=True), va[va.y >= 0].reset_index(drop=True)
    if SMOKE: tr, va = tr.sample(3000, random_state=0).reset_index(drop=True), va.head(500)
    return tr, va, len(lab2id)


SMOKE = bool(os.environ.get("SMOKE"))
def eval_sets(rank):
    s = X.eval_sets(rank, darkmatter=rank != "gene_taxa")
    return {k: v.sample(min(400, len(v)), random_state=0).reset_index(drop=True) for k, v in s.items()} if SMOKE else s


MAXLEN = {"gene_taxa": 3000}


def cut(seqs, rank):
    m = MAXLEN.get(rank); return [s[:m] for s in seqs] if m else list(seqs)


# ---------------------------------------------------------------- k-mer RF
def kmer_freq(seqs):
    """[N, 4^k] float32 6-mer frequencies over ACGT k-mers (k-mers with other characters skipped)"""
    lut = np.full(256, 255, np.uint8)
    for c, v in ALPH.items(): lut[ord(c)] = v; lut[ord(c.lower())] = v
    X_ = np.zeros((len(seqs), 4 ** KMER), np.float32); w = 4 ** np.arange(KMER - 1, -1, -1)
    for i, s in enumerate(seqs):
        a = lut[np.frombuffer(s.encode(), np.uint8)]
        if len(a) < KMER: continue
        win = np.lib.stride_tricks.sliding_window_view(a, KMER)
        ok = (win != 255).all(1); idx = (win[ok].astype(np.int64) * w).sum(1)
        c = np.bincount(idx, minlength=4 ** KMER).astype(np.float32); X_[i] = c / max(c.sum(), 1)
    return X_


def run_rf(rank):
    from sklearn.ensemble import RandomForestClassifier
    tr, _, C = train_data(rank); log(f"RF {rank}: {len(tr)} training sequences, {C} classes")
    Xtr = kmer_freq(cut(tr.sequence, rank)); log("features done")
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=int(os.environ.get("RF_JOBS", 96)), random_state=1)
    rf.fit(Xtr, tr.y.values); del Xtr; log("fit done")
    sets = eval_sets(rank); frames = {}
    for n, d in sets.items():
        Xe = kmer_freq(cut(d.sequence, rank)); N = len(d)
        pm = np.zeros((N, C)); hk = np.zeros(N)
        for t in rf.estimators_:                                   # stream over trees: mean p and mean per-tree entropy
            p = np.zeros((N, C)); p[:, rf.classes_] = t.predict_proba(Xe)
            pm += p; hk += -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
        pm /= len(rf.estimators_); hk /= len(rf.estimators_)
        H = -(pm * np.log(np.clip(pm, 1e-12, None))).sum(1)
        frames[n] = X.rows_from_probs("k-mer RF", d.y.values, pm, H / np.log(C), np.clip(H - hk, 0, None) / np.log(C))
        log(f"  {n}: acc {(frames[n].pred.values == d.y.values)[d.y.values >= 0].mean() if (d.y.values >= 0).any() else float('nan'):.4f}")
    finish("kmer_rf", rank, sets, frames)


# ---------------------------------------------------------------- CNN
def onehot_u8(seqs, L):
    lut = np.full(256, 4, np.uint8)
    for c, v in ALPH.items(): lut[ord(c)] = v; lut[ord(c.lower())] = v
    A = np.full((len(seqs), L), 4, np.uint8)
    for i, s in enumerate(seqs):
        a = lut[np.frombuffer(s[:L].encode(), np.uint8)]; A[i, :len(a)] = a
    return A                                                      # index 4 = padding / N -> all-zero one-hot


def make_cnn(C, p=0.3):
    import torch.nn as nn, torch.nn.functional as F
    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.c1 = nn.Conv1d(4, 64, 15, padding=7); s.c2 = nn.Conv1d(64, 128, 9, padding=4)
            s.d1 = nn.Dropout(p); s.f1 = nn.Linear(128, 64); s.d2 = nn.Dropout(p); s.f2 = nn.Linear(64, C)
        def forward(s, x):
            x = F.max_pool1d(F.relu(s.c1(x)), 4); x = F.adaptive_max_pool1d(F.relu(s.c2(x)), 1).squeeze(-1)
            return s.f2(s.d2(F.relu(s.f1(s.d1(x)))))
    return CNN()


def run_cnn(rank, seeds):
    import torch, torch.nn.functional as F
    dev = "cuda"
    tr, va, C = train_data(rank); L = MAXLEN.get(rank, 1500)
    Atr, Ava = onehot_u8(tr.sequence.tolist(), L), onehot_u8(va.sequence.tolist(), L)
    ytr, yva = torch.from_numpy(tr.y.values).long(), torch.from_numpy(va.y.values).long()
    eye = torch.cat([torch.eye(4), torch.zeros(1, 4)]).to(dev)
    oh = lambda a: eye[torch.from_numpy(a).to(dev).long()].transpose(1, 2)          # [B, 4, L]
    log(f"CNN {rank}: {len(tr)} train / {len(va)} val, {C} classes, length {L}")

    @torch.no_grad()
    def logits(m, A, bs=512):
        return torch.cat([m(oh(A[i:i + bs])).float().cpu() for i in range(0, len(A), bs)])

    models = {}
    for sd in seeds:
        ck = X.OUT / f"{'smoke_' if SMOKE else ''}cnn_{rank}" / f"cnn_seed{sd}.pt"
        if ck.exists():
            m = make_cnn(C).to(dev); m.load_state_dict(torch.load(ck)); models[sd] = m.eval(); log(f"seed {sd}: loaded"); continue
        torch.manual_seed(sd); m = make_cnn(C).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
        best, bad, state = 1e9, 0, None; g = torch.Generator().manual_seed(sd)
        for ep in range(30):
            m.train(); perm = torch.randperm(len(Atr), generator=g)
            for i in range(0, len(perm), 128):
                idx = perm[i:i + 128].numpy(); opt.zero_grad()
                F.cross_entropy(m(oh(Atr[idx])), ytr[idx].to(dev)).backward(); opt.step()
            m.eval(); vl = F.cross_entropy(logits(m, Ava), yva).item()
            log(f"seed {sd} epoch {ep + 1}: val loss {vl:.4f}")
            if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 3: break
        m.load_state_dict(state); ck.parent.mkdir(parents=True, exist_ok=True); torch.save(state, ck); models[sd] = m.eval()
    sets = eval_sets(rank); frames = {}
    for n, d in sets.items():
        A = onehot_u8(cut(d.sequence.tolist(), rank), L); y = d.y.values
        P = {sd: logits(m, A).softmax(-1).numpy()[None] for sd, m in models.items()}
        parts = [X.rows_from_probs("CNN", y, *X.summarize_probs(P[seeds[0]]))]
        m = models[seeds[0]]; m.train()                                              # dropout live: MC samples
        with torch.no_grad(): mc = np.stack([logits(m, A).softmax(-1).numpy() for _ in range(10)])
        m.eval(); parts.append(X.rows_from_probs("CNN + MC dropout", y, *X.summarize_probs(mc)))
        if len(models) > 1:
            parts.append(X.rows_from_probs(f"CNN ensemble ({len(models)} seeds)", y, *X.summarize_probs(np.concatenate(list(P.values())))))
        frames[n] = pd.concat(parts, ignore_index=True)
        iv = y >= 0
        log(f"  {n}: " + ", ".join(f"{p.method.iloc[0]} acc {(p.pred.values[iv] == y[iv]).mean():.4f}" for p in parts) if iv.any() else f"  {n}: scored")
    finish("cnn", rank, sets, frames)


def finish(model, rank, sets, frames):
    if SMOKE: model = "smoke_" + model
    dmr = []
    if "darkmatter" in frames:
        dm = frames.pop("darkmatter"); meta = sets["darkmatter"][["up", "sample"]].reset_index(drop=True); idt = frames["id_test"]
        for meth in dm.method.unique():
            d1 = pd.concat([meta, dm[dm.method == meth].reset_index(drop=True)], axis=1); i1 = idt[idt.method == meth]
            dmr += [dict(r, backbone=meth) for r in X.darkmatter_rows(meth, rank, d1, i1, [c for c in d1 if c.startswith("det_") and d1[c].notna().all()])]
    X.write(model, rank, frames, dmr)
    log("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True, choices=["phylum", "family", "gene_taxa"])
    ap.add_argument("--model", required=True, choices=["rf", "cnn"])
    ap.add_argument("--seeds", default="1,2,3,4,5")
    a = ap.parse_args()
    if a.model == "rf": run_rf(a.rank)
    else: run_cnn(a.rank, [int(s) for s in a.seeds.split(",")])
