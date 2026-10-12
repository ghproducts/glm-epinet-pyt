#!/usr/bin/env python3
"""Dark matter scored against FULL-LENGTH in-distribution data -- no windowing.

Everywhere else in this project id_test is cut to a 150bp window so it sits in
the same length range as dark matter. This does the opposite: id_test and the
retrieval reference are left at the length the model was fine-tuned on (~6kb,
i.e. model_max_length=1000 tokens), while dark matter is fed at its native
75-150bp. It is therefore the maximally UNMATCHED comparison -- every separation
it reports is confounded with read length -- but it is also the model's native
operating point, and it bounds how large the length artifact can get.

Single-pass scores only (msp, energy, entropy, Mahalanobis, k-NN). MC-dropout
and Epinet are omitted: K forward passes over 13k reads at 1000 tokens costs
hours and adds nothing to the length question.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score

from nn_proj.common.utils import compute_uncertainty

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
CKPT = {"phylum": "checkpoints/pbsim_clean/phylum/checkpoint-57734",
        "family": "checkpoints/pbsim_clean/family/checkpoint-57734"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FPRS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


@torch.no_grad()
def fwd(m, tok, seqs, L, bs, tag=""):
    F, G = [], []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        o = m(**b, output_hidden_states=True)
        h = o.hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        F.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        G.append(o.logits.float().cpu())
        if tag and i % (bs * 100) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(F), torch.cat(G)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--rank", required=True, choices=["phylum", "family"])
    ap.add_argument("--max-len", type=int, default=1000)     # ~6kb, the training length
    ap.add_argument("--n-ref", type=int, default=40000)
    ap.add_argument("--bs-long", type=int, default=12)
    ap.add_argument("--bs-short", type=int, default=384)
    args = ap.parse_args()

    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    dm = dm[dm.up != 0.5].reset_index(drop=True)
    ref = pd.concat([c for c in pd.read_csv(D / "train.csv", usecols=["sequence", "taxid"],
                                            chunksize=50_000)], ignore_index=True)
    ref = ref.sample(args.n_ref, random_state=1)
    lin = pd.read_csv(D / "full_basic_lineage.csv", index_col=0)
    s2t = dict(zip(lin["species"].astype("Int64"), lin[args.rank].astype("Int64")))
    ref["lab"] = ref.taxid.map(lambda t: int(s2t.get(t, -1))); ref = ref[ref.lab != -1]
    idt = pd.read_csv(D / "id_test.csv", usecols=["sequence"])
    print(f"encoder {CKPT[args.rank]}\n  reference {len(ref):,} train reads at full length"
          f"\n  id_test {len(idt):,} at full length\n  dark matter {len(dm):,} at native "
          f"{dm.sequence.str.len().mean():.0f}bp mean", flush=True)

    cfg = transformers.AutoConfig.from_pretrained(CKPT[args.rank], trust_remote_code=True)
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    m = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB, config=cfg, trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CKPT[args.rank]) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()

    R, _ = fwd(m, tok, ref.sequence.values, args.max_len, args.bs_long, "ref")
    Fi, Gi = fwd(m, tok, idt.sequence.values, args.max_len, args.bs_long, "id")
    Fd, Gd = fwd(m, tok, dm.sequence.values, args.max_len, args.bs_short, "dm")
    del m; torch.cuda.empty_cache()

    S = {}
    S["msp"] = (-torch.softmax(Gi, -1).max(1).values.numpy(),
                -torch.softmax(Gd, -1).max(1).values.numpy())
    S["energy"] = (-torch.logsumexp(Gi, 1).numpy(), -torch.logsumexp(Gd, 1).numpy())
    S["entropy"] = (compute_uncertainty(Gi)["normalized_total_uncertainty"].numpy(),
                    compute_uncertainty(Gd)["normalized_total_uncertainty"].numpy())
    Rn = R.numpy().astype(np.float64)
    def whitener(P):
        w, V = np.linalg.eigh(P); return V * np.sqrt(np.clip(w, 0, None))
    classes = np.unique(ref.lab.values)
    MU = np.stack([Rn[ref.lab.values == c].mean(0) for c in classes])
    resid = Rn - MU[np.searchsorted(classes, ref.lab.values)]
    W = whitener(np.linalg.pinv(np.cov(resid.T) + 1e-6 * np.eye(Rn.shape[1])))
    MUw = torch.from_numpy(MU @ W).to(DEV).float()
    def maha(F):
        Xw = torch.from_numpy(F.numpy().astype(np.float64) @ W).to(DEV).float()
        o = torch.empty(len(Xw), device=DEV)
        for i in range(0, len(Xw), 1024):
            o[i:i + 1024] = torch.cdist(Xw[i:i + 1024], MUw).pow(2).min(1).values
        return o.cpu().numpy()
    S["maha_class"] = (maha(Fi), maha(Fd))
    Rd_ = R.to(DEV)
    def knn(E, k):
        s = np.empty(len(E))
        for i in range(0, len(E), 512):
            sim = E[i:i + 512].to(DEV) @ Rd_.T
            s[i:i + 512] = (-sim.topk(k, 1).values[:, -1]).cpu().numpy()
        return s
    for k in (1, 10, 50):
        S[f"knn_k{k}"] = (knn(Fi, k), knn(Fd, k))

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
            rr = d2.assign(fl=(d2.v > T).astype(float)).groupby(["up", "sample"]).fl.mean().reset_index()
            s2, _, _, p2, _ = stats.linregress(rr.up, rr.fl)
            r[f"slope_fpr{f}"] = s2; r[f"p_fpr{f}"] = p2
            r[f"flag_fpr{f}"] = (d2.v > T).mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(f"results/darkmatter_fulllen_{args.rank}.csv", index=False)
    np.savez_compressed(f"results/darkmatter_fulllen_{args.rank}_scores.npz", up=dm.up.values,
                        sample=dm["sample"].values,
                        **{f"id__{k}": v[0] for k, v in S.items()},
                        **{f"dm__{k}": v[1] for k, v in S.items()})
    print(f"\n=== {args.rank}: id_test at FULL LENGTH vs dark matter at native ~120bp ===")
    print(f"{'method':14s}{'AUROC id v dm':>15}{'flag@1%':>10}{'slope@1%':>11}{'p':>9}")
    for _, r in out.iterrows():
        print(f"{r.method:14s}{r.auroc_id_vs_all:>15.4f}{r['flag_fpr0.01']:>10.4f}"
              f"{r['slope_fpr0.01']:>+11.4f}{r['p_fpr0.01']:>9.4f}")


if __name__ == "__main__":
    main()
