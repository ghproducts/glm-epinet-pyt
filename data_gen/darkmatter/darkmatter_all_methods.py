#!/usr/bin/env python3
"""Every OOD-detection method on dark matter, for one encoder rank.

darkmatter_canonical.py covers retrieval (kNN) only. This runs the full set the
rest of the project uses, so the dark matter question gets the same treatment as
the pbsim ID/OOD ladder:

  output space    msp, energy, entropy                      (base model)
  sampling        entropy + epistemic, K=10                 (MC-dropout, p=0.1)
  sampling        entropy + epistemic, K=16                 (Epinet, conv prior)
  feature space   Mahalanobis global / class-conditional,
                  kNN k=1/10/50                             (retrieval)

phylum and family are SEPARATELY fine-tuned checkpoints (36-way vs 429-way) with
their own Epinet heads, so --rank selects a genuinely different encoder, not a
different read-out of one model.

What is being measured is the TREND across `up`, not any absolute separation:
the composition of the dark matter set is unknown, so a single AUROC against
id_test means little, while "does this score rise as the unknown fraction
rises" is answerable. Each method therefore reports
  - slope of the mean score vs up (replicate means, threshold-free)
  - AUROC up=0.25 vs up=1.00 (threshold-free, 0.5 = no trend)
  - slope of the flagged fraction vs up at id_test-calibrated FPRs

Configuration is copied from darkmatter_canonical.py so the two are comparable:
all 180k reads (up=0.5 dropped, it has no replicates), 150bp windows, 48 tokens,
80k reference windows at natural abundance, thresholds from all 13,220 id_test
reads, seed 1. Scores are reduced to scalars per batch -- a [16, 180000, 429]
logit stack would not fit.

Caveat inherited from the whole dark matter analysis: these reads are ~150bp
while the encoders were fine-tuned on 6kb windows, a distribution shift layered
underneath whatever `up` represents. It applies equally to every method here.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score

from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.models.epinet import (EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier,
                                   MLPEpinetWithConvPrior)
from nn_proj.models.epinet.feature_fns import NT_feature_fn

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
CKPT = {"phylum": "checkpoints/pbsim_clean/phylum/checkpoint-57734",
        "family": "checkpoints/pbsim_clean/family/checkpoint-57734"}
EPI = {"phylum": "checkpoints/pbsim_clean_epinet/phylum/checkpoint-57734",
       "family": "checkpoints/pbsim_clean_epinet/family/checkpoint-57734"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FPRS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def build_base(rank, half=True):
    cfg = transformers.AutoConfig.from_pretrained(CKPT[rank], trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB, config=cfg, trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CKPT[rank]) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval()
    return (m.half() if half else m), cfg


@torch.no_grad()
def pass_base(m, tok, seqs, L, bs, tag=""):
    """One forward pass -> (normalized mean-pooled embeddings, logits)."""
    F, G = [], []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        o = m(**b, output_hidden_states=True)
        h = o.hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        F.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        G.append(o.logits.float().cpu())
        if tag and i % (bs * 200) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(F), torch.cat(G)


@torch.no_grad()
def pass_sampled(m, tok, seqs, L, bs, K, mode, tag=""):
    """K-sample pass -> per-read (entropy, epistemic, max_confidence).

    Reduced per batch: holding [K, N, C] for N=180,000 is not feasible.
    """
    ent, epi, conf = [], [], []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        if mode == "epinet":
            # plain dict, not the tokenizer's BatchEncoding: the conv prior tests
            # `isinstance(inputs, dict)` to find input_ids, and BatchEncoding
            # extends UserDict rather than dict, so it silently fails that test.
            stack = m.wrapper({k: v for k, v in b.items()},
                              n_index_samples=K, return_all=True)         # [K,B,C]
        else:
            stack = torch.stack([m(**b).logits for _ in range(K)], 0)     # [K,B,C]
        u = compute_uncertainty(stack.float())
        ent.append(u["normalized_total_uncertainty"].cpu())
        epi.append(u["normalized_epistemic_uncertainty"].cpu())
        conf.append(u["max_confidence"].cpu())
        if tag and i % (bs * 100) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return (torch.cat(ent).numpy(), torch.cat(epi).numpy(), torch.cat(conf).numpy())


def whitener(P):
    w, V = np.linalg.eigh(P)
    return V * np.sqrt(np.clip(w, 0, None))


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--rank", required=True, choices=["phylum", "family"])
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=384)
    ap.add_argument("--n-ref", type=int, default=80000)
    ap.add_argument("--k-mc", type=int, default=10)
    ap.add_argument("--k-epi", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: cap dark matter / id_test / reference sizes")
    args = ap.parse_args()
    rng = np.random.default_rng(1); L = args.read_len
    cut = lambda s: s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]

    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    dm = dm[dm.up != 0.5].reset_index(drop=True)
    ref = pd.concat([c for c in pd.read_csv(D / "train.csv", usecols=["sequence", "taxid"],
                                            chunksize=50_000)], ignore_index=True)
    ref = ref.sample(args.n_ref, random_state=1)
    lin = pd.read_csv(D / "full_basic_lineage.csv", index_col=0)
    s2t = dict(zip(lin["species"].astype("Int64"), lin[args.rank].astype("Int64")))
    ref["ranklab"] = ref.taxid.map(lambda t: int(s2t.get(t, -1)))
    ref = ref[ref.ranklab != -1]
    if args.limit:   # smoke test only; keeps every `up` level represented
        dm = dm.groupby("up", group_keys=False).apply(
            lambda g: g.sample(min(len(g), args.limit), random_state=1)).reset_index(drop=True)
        ref = ref.head(args.limit)
    ref_w = [cut(s) for s in ref.sequence]
    id_w = [cut(s) for s in pd.read_csv(D / "id_test.csv", usecols=["sequence"]).sequence]
    if args.limit:
        id_w = id_w[:args.limit]
    dm_s = list(dm.sequence.values)
    print(f"encoder {CKPT[args.rank]}\ndark matter ALL {len(dm):,} | reference {len(ref_w):,} "
          f"| id_test ALL {len(id_w):,}", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token

    S = {}   # name -> (id_test scores, darkmatter scores); higher = more novel
    m, cfg = build_base(args.rank)
    Rf, _ = pass_base(m, tok, ref_w, args.max_len, args.bs, "ref")
    Fi, Gi = pass_base(m, tok, id_w, args.max_len, args.bs, "id")
    Fd, Gd = pass_base(m, tok, dm_s, args.max_len, args.bs, "dm")
    S["msp"] = (-torch.softmax(Gi, -1).max(1).values.numpy(),
                -torch.softmax(Gd, -1).max(1).values.numpy())
    S["energy"] = (-torch.logsumexp(Gi, 1).numpy(), -torch.logsumexp(Gd, 1).numpy())
    S["entropy"] = (compute_uncertainty(Gi)["normalized_total_uncertainty"].numpy(),
                    compute_uncertainty(Gd)["normalized_total_uncertainty"].numpy())
    del m; torch.cuda.empty_cache()

    # ---- feature space -------------------------------------------------
    Rn = Rf.numpy().astype(np.float64)
    mu_g = Rn.mean(0, keepdims=True)
    W_g = whitener(np.linalg.pinv(np.cov((Rn - mu_g).T) + 1e-6 * np.eye(Rn.shape[1])))
    mg = lambda F: (((F.numpy().astype(np.float64) - mu_g) @ W_g) ** 2).sum(1)
    S["maha_global"] = (mg(Fi), mg(Fd))

    classes = np.unique(ref.ranklab.values)
    MUc = np.stack([Rn[ref.ranklab.values == c].mean(0) for c in classes])
    resid = Rn - MUc[np.searchsorted(classes, ref.ranklab.values)]
    W_c = whitener(np.linalg.pinv(np.cov(resid.T) + 1e-6 * np.eye(Rn.shape[1])))
    MUw = torch.from_numpy(MUc @ W_c).to(DEV).float()
    def mc_(F):
        Xw = torch.from_numpy(F.numpy().astype(np.float64) @ W_c).to(DEV).float()
        o = torch.empty(len(Xw), device=DEV)
        for i in range(0, len(Xw), 1024):
            o[i:i + 1024] = torch.cdist(Xw[i:i + 1024], MUw).pow(2).min(1).values
        return o.cpu().numpy()
    S["maha_class"] = (mc_(Fi), mc_(Fd))
    print(f"  class-conditional fit over {len(classes)} {args.rank} classes", flush=True)

    Rd = Rf.to(DEV)
    def knn(E, k):
        s = np.empty(len(E))
        for i in range(0, len(E), 512):
            sim = E[i:i + 512].to(DEV) @ Rd.T
            s[i:i + 512] = (-sim.topk(k, 1).values[:, -1]).cpu().numpy()
        return s
    for k in (1, 10, 50):
        S[f"knn_k{k}"] = (knn(Fi, k), knn(Fd, k))
    del Rd; torch.cuda.empty_cache()

    # ---- MC-dropout ----------------------------------------------------
    m, _ = build_base(args.rank, half=False)
    enable_mc_dropout(m, p=0.1)
    ei, pi, ci = pass_sampled(m, tok, id_w, args.max_len, args.bs, args.k_mc, "mc", "mc-id")
    ed, pd_, cd = pass_sampled(m, tok, dm_s, args.max_len, args.bs, args.k_mc, "mc", "mc-dm")
    S["mc_entropy"] = (ei, ed); S["mc_epistemic"] = (pi, pd_); S["mc_msp"] = (-ci, -cd)
    del m; torch.cuda.empty_cache()

    # ---- Epinet --------------------------------------------------------
    try:
        base, cfg = build_base(args.rank, half=False)
        ecfg = EpinetConfig(num_classes=cfg.num_labels, include_inputs=True,
                            vocab_size=cfg.vocab_size)
        wrap = EpinetWrapper(base, NT_feature_fn, ecfg, epinet=MLPEpinetWithConvPrior)
        em = HFEpinetSeqClassifier(wrap, k_train=8, k_eval=8).to(DEV)
        with torch.no_grad():
            d0 = tok("ACGT", truncation=True, padding="max_length",
                     max_length=args.max_len, return_tensors="pt")
            em(**{k: v.to(DEV) for k, v in d0.items()})
        em.load_state_dict(load_file(str(Path(EPI[args.rank]) / "model.safetensors")), strict=True)
        em.eval()
        ei, pi, ci = pass_sampled(em, tok, id_w, args.max_len, args.bs, args.k_epi, "epinet", "epi-id")
        ed, pd_, cd = pass_sampled(em, tok, dm_s, args.max_len, args.bs, args.k_epi, "epinet", "epi-dm")
        S["epi_entropy"] = (ei, ed); S["epi_epistemic"] = (pi, pd_); S["epi_msp"] = (-ci, -cd)
        del em; torch.cuda.empty_cache()
    except Exception as e:                      # epinet was trained at 1000 tokens
        print(f"  [epinet SKIPPED] {type(e).__name__}: {e}", flush=True)

    # ---- analysis ------------------------------------------------------
    ups = sorted(dm.up.unique())
    rows = []
    for name, (si, sd) in S.items():
        d2 = dm.assign(v=sd)
        rep = d2.groupby(["up", "sample"]).v.mean().reset_index()
        sl, _, _, pv, _ = stats.linregress(rep.up, rep.v)
        a, b = d2[d2.up == ups[0]].v.values, d2[d2.up == ups[-1]].v.values
        auc = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
        r = dict(method=name, mean_slope=sl, mean_p=pv, auroc_lo_hi=auc,
                 auroc_id_vs_hi=roc_auc_score(
                     np.r_[np.zeros(len(si)), np.ones(len(b))], np.r_[si, b]))
        for f in FPRS:
            T = np.quantile(si, 1 - f)
            fl = d2.assign(flag=(d2.v > T).astype(float))
            rr = fl.groupby(["up", "sample"]).flag.mean().reset_index()
            s2, _, _, p2, _ = stats.linregress(rr.up, rr.flag)
            r[f"slope_fpr{f}"] = s2; r[f"p_fpr{f}"] = p2
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(f"results/darkmatter_all_methods_{args.rank}.csv", index=False)
    np.savez_compressed(f"results/darkmatter_all_methods_{args.rank}_scores.npz",
                        up=dm.up.values, sample=dm["sample"].values,
                        **{f"id__{k}": v[0] for k, v in S.items()},
                        **{f"dm__{k}": v[1] for k, v in S.items()})

    print(f"\n=== {args.rank}: trend across up (threshold-free) ===")
    print(f"{'method':16s}{'mean slope':>13}{'p':>9}{'AUROC .25v1':>13}{'AUROC id v 1':>14}")
    for _, r in out.iterrows():
        print(f"{r.method:16s}{r.mean_slope:>+13.5f}{r.mean_p:>9.4f}"
              f"{r.auroc_lo_hi:>13.4f}{r.auroc_id_vs_hi:>14.4f}")
    print(f"\n=== {args.rank}: flagged-fraction slope vs up (FPR from id_test) ===")
    print(f"{'method':16s}" + "".join(f"{f:>10.1%}" for f in FPRS))
    for _, r in out.iterrows():
        print(f"{r.method:16s}" + "".join(
            f"{r[f'slope_fpr{f}']:>+9.4f}" + ("*" if r[f"p_fpr{f}"] < 0.05 else " ")
            for f in FPRS))


if __name__ == "__main__":
    main()
