#!/usr/bin/env python3
"""METAGENE-1 embeddings for dark matter ID/OOD, with NO fine-tuning.

Motivation: the length sweep (data_gen/pbsim/length_sweep.py) showed the NT
encoder has NO detection signal below ~600bp -- AUROC 0.51-0.53 on tiers where
the ground truth is known -- and only recovers above ~1200bp. Dark matter is
150bp, so every NT result sits below that threshold. METAGENE-1 is pretrained
on 1.5T bp of SHORT metagenomic reads and is explicitly built for anomaly
detection at 100-300bp, so it is the natural test of whether the failure is the
read length itself or just this model at this read length.

Configuration is copied verbatim from darkmatter_all_methods.py so the numbers
sit beside the NT ones: all 180k dark matter reads (up=0.5 dropped), 150bp
windows for the reference and id_test, 80k reference windows at natural
abundance, thresholds from all 13,220 id_test reads, seed 1. Only the encoder
changes.

Scores are representation-distance only (k-NN, Mahalanobis) -- no classifier
head is trained, so there is no output-space score to report.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from scipy import stats
from sklearn.metrics import roc_auc_score

MODEL = "metagene-ai/METAGENE-1"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FPRS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


@torch.no_grad()
def embed(m, tok, seqs, L, bs, tag=""):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L)
        b = {k: v.to(DEV) for k, v in b.items()}
        o = m(**b)
        h = o[0] if isinstance(o, (tuple, list)) else o.last_hidden_state
        mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        if tag and i % (bs * 100) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out)


def whitener(P):
    w, V = np.linalg.eigh(P)
    return V * np.sqrt(np.clip(w, 0, None))


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=512)   # model max_position_embeddings
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--n-ref", type=int, default=80000)
    args = ap.parse_args()
    rng = np.random.default_rng(1); L = args.read_len
    cut = lambda s: s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]

    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    dm = dm[dm.up != 0.5].reset_index(drop=True)
    ref = pd.concat([c for c in pd.read_csv(D / "train.csv", usecols=["sequence", "taxid"],
                                            chunksize=50_000)], ignore_index=True)
    ref = ref.sample(args.n_ref, random_state=1)
    lin = pd.read_csv(D / "full_basic_lineage.csv", index_col=0)
    for rank in ("phylum", "family"):
        s2t = dict(zip(lin["species"].astype("Int64"), lin[rank].astype("Int64")))
        ref[rank] = ref.taxid.map(lambda t: int(s2t.get(t, -1)))
    ref = ref[(ref.phylum != -1) & (ref.family != -1)]
    ref_w = [cut(s) for s in ref.sequence]
    id_w = [cut(s) for s in pd.read_csv(D / "id_test.csv", usecols=["sequence"]).sequence]
    print(f"METAGENE-1 (no fine-tuning)\n  reference {len(ref_w):,} | id_test {len(id_w):,} "
          f"| dark matter {len(dm):,}", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = transformers.AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                               torch_dtype=torch.float16).to(DEV).eval()
    R = embed(m, tok, ref_w, args.max_len, args.bs, "ref")
    Ei = embed(m, tok, id_w, args.max_len, args.bs, "id")
    Ed = embed(m, tok, dm.sequence.values, args.max_len, args.bs, "dm")
    del m; torch.cuda.empty_cache()
    print(f"  embeddings: ref {tuple(R.shape)} id {tuple(Ei.shape)} dm {tuple(Ed.shape)}", flush=True)

    S = {}
    Rn = R.numpy().astype(np.float64)
    mu = Rn.mean(0, keepdims=True)
    Wg = whitener(np.linalg.pinv(np.cov((Rn - mu).T) + 1e-6 * np.eye(Rn.shape[1])))
    mg = lambda E: (((E.numpy().astype(np.float64) - mu) @ Wg) ** 2).sum(1)
    S["maha_global"] = (mg(Ei), mg(Ed))
    for rank in ("phylum", "family"):
        lab = ref[rank].values
        cls = np.unique(lab)
        MU = np.stack([Rn[lab == c].mean(0) for c in cls])
        resid = Rn - MU[np.searchsorted(cls, lab)]
        Wc = whitener(np.linalg.pinv(np.cov(resid.T) + 1e-6 * np.eye(Rn.shape[1])))
        MUw = torch.from_numpy(MU @ Wc).to(DEV).float()
        def mc(E):
            Xw = torch.from_numpy(E.numpy().astype(np.float64) @ Wc).to(DEV).float()
            o = torch.empty(len(Xw), device=DEV)
            for i in range(0, len(Xw), 1024):
                o[i:i + 1024] = torch.cdist(Xw[i:i + 1024], MUw).pow(2).min(1).values
            return o.cpu().numpy()
        S[f"maha_class_{rank}"] = (mc(Ei), mc(Ed))
        print(f"  class-conditional fit over {len(cls)} {rank} classes", flush=True)
    Rd = R.to(DEV)
    def knn(E, k):
        s = np.empty(len(E))
        for i in range(0, len(E), 512):
            sim = E[i:i + 512].to(DEV) @ Rd.T
            s[i:i + 512] = (-sim.topk(k, 1).values[:, -1]).cpu().numpy()
        return s
    for k in (1, 10, 50):
        S[f"knn_k{k}"] = (knn(Ei, k), knn(Ed, k))
    del Rd; torch.cuda.empty_cache()

    ups = sorted(dm.up.unique()); rows = []
    for name, (si, sd) in S.items():
        d2 = dm.assign(v=sd)
        rep = d2.groupby(["up", "sample"]).v.mean().reset_index()
        sl, _, _, pv, _ = stats.linregress(rep.up, rep.v)
        r = {"method": name, "mean_slope": sl, "mean_p": pv,
             "auroc_id_vs_all": roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(sd))], np.r_[si, sd])}
        for u in ups:
            b = d2[d2.up == u].v.values
            r[f"auroc_id_vs_{u}"] = roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(b))], np.r_[si, b])
        for f in FPRS:
            T = np.quantile(si, 1 - f)
            fl = d2.assign(fl=(d2.v > T).astype(float))
            rr = fl.groupby(["up", "sample"]).fl.mean().reset_index()
            s2, _, _, p2, _ = stats.linregress(rr.up, rr.fl)
            r[f"slope_fpr{f}"] = s2; r[f"p_fpr{f}"] = p2; r[f"flag_fpr{f}"] = (d2.v > T).mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv("results/darkmatter_metagene1.csv", index=False)
    np.savez_compressed("results/darkmatter_metagene1_scores.npz", up=dm.up.values,
                        sample=dm["sample"].values,
                        **{f"id__{k}": v[0] for k, v in S.items()},
                        **{f"dm__{k}": v[1] for k, v in S.items()})
    print("\n=== METAGENE-1 (no fine-tuning), dark matter ===")
    print(f"{'method':20s}{'AUROCid/dm':>12}{'id v .25':>10}{'id v 1.0':>10}"
          f"{'flag@1%':>10}{'slope@1%':>11}{'p':>8}")
    for _, r in out.iterrows():
        print(f"{r.method:20s}{r.auroc_id_vs_all:>12.4f}{r['auroc_id_vs_0.25']:>10.4f}"
              f"{r['auroc_id_vs_1.0']:>10.4f}{r['flag_fpr0.01']:>10.4f}"
              f"{r['slope_fpr0.01']:>+11.4f}{r['p_fpr0.01']:>8.4f}")


if __name__ == "__main__":
    main()
