#!/usr/bin/env python3
"""Every uncertainty score on the dark matter set, scored the same way.

Scores computed in one pass from the clean phylum checkpoint:
  msp, energy               output space (classifier head)
  mahalanobis, knn          feature space (distance to the training reference)
  mcdo_entropy, mcdo_epi    MC dropout, K=10

For each: AUROC separating dark matter from id_test, and the slope of
flagged-fraction vs `up`. Thresholds are calibrated on id_test at a fixed false
positive rate -- "more novel than (1-fpr) of known reads" -- rather than as a
quantile of the dark matter scores themselves, which would be calibrated on the
test data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nn_proj.common.utils import enable_mc_dropout

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
CLEAN = "checkpoints/pbsim_clean/phylum/checkpoint-57734"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run(m, tok, seqs, L, bs, mc=0, tag=""):
    """returns (features, logits) or (None, [K,N,C] logits) when mc>0"""
    F, G = [], []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        if mc:
            G.append(torch.stack([m(**b).logits.float().cpu() for _ in range(mc)]))
        else:
            o = m(**b, output_hidden_states=True)
            h = o.hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
            e = (h * mk).sum(1) / mk.sum(1)
            F.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
            G.append(o.logits.float().cpu())
        if tag and i % (bs * 200) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return (None, torch.cat(G, 1)) if mc else (torch.cat(F), torch.cat(G))


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--mc-k", type=int, default=10)
    ap.add_argument("--n-ref", type=int, default=40000)
    ap.add_argument("--n-id", type=int, default=6000)
    ap.add_argument("--n-dm", type=int, default=16000)
    args = ap.parse_args()
    rng = np.random.default_rng(1); L = args.read_len

    cut = lambda ss: [s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]
                      for s in ss]
    ref = pd.concat([c for c in pd.read_csv(D / "train.csv", usecols=["sequence"],
                                            chunksize=50_000)], ignore_index=True)
    ref_w = cut(ref.sample(args.n_ref, random_state=1).sequence.values) * 2
    idt = cut(pd.read_csv(D / "id_test.csv", usecols=["sequence"])
                .sample(args.n_id, random_state=1).sequence.values)
    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    dm = dm[dm.up != 0.5]                                   # single file, no replicates
    per = args.n_dm // dm.up.nunique()
    dm = (dm.groupby("up", group_keys=False)
            .apply(lambda g: g.sample(n=min(per, len(g)), random_state=1)).reset_index(drop=True))
    print(f"ref {len(ref_w):,} | id {len(idt):,} | dm {len(dm):,}")

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    cfg = transformers.AutoConfig.from_pretrained(CLEAN, trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                        trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CLEAN) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()

    Fr, _ = run(m, tok, ref_w, args.max_len, args.bs, tag="ref")
    Fi, Gi = run(m, tok, idt, args.max_len, args.bs, tag="id")
    Fd, Gd = run(m, tok, dm.sequence.values, args.max_len, args.bs, tag="dm")
    enable_mc_dropout(m, p=0.1)
    _, Mi = run(m, tok, idt, args.max_len, args.bs, mc=args.mc_k, tag="mc-id")
    _, Md = run(m, tok, dm.sequence.values, args.max_len, args.bs, mc=args.mc_k, tag="mc-dm")
    del m; torch.cuda.empty_cache()

    mu = Fr.numpy().mean(0, keepdims=True)
    P = np.linalg.pinv(np.cov((Fr.numpy() - mu).T) + 1e-6 * np.eye(Fr.shape[1]))
    maha = lambda F: np.einsum("ij,jk,ik->i", F.numpy() - mu, P, F.numpy() - mu)
    Rd = Fr.to(DEV)
    def knn(F):
        s = np.empty(len(F))
        for i in range(0, len(F), 512):
            sim = F[i:i + 512].to(DEV) @ Rd.T
            s[i:i + 512] = (-sim.topk(args.k, 1).values[:, -1]).cpu().numpy()
        return s
    def bald(G):
        p = torch.softmax(G, -1); pb = p.mean(0)
        ent = lambda q: -(q * torch.log(q + 1e-12)).sum(-1)
        return ent(pb).numpy(), (ent(pb) - ent(p).mean(0)).numpy()
    ei, ii = bald(Mi); ed, idd = bald(Md)

    S = {"msp": (-torch.softmax(Gi, -1).max(1).values.numpy(), -torch.softmax(Gd, -1).max(1).values.numpy()),
         "energy": (-torch.logsumexp(Gi, 1).numpy(), -torch.logsumexp(Gd, 1).numpy()),
         "mahalanobis": (maha(Fi), maha(Fd)),
         f"knn_k{args.k}": (knn(Fi), knn(Fd)),
         "mcdo_entropy": (ei, ed),
         "mcdo_epistemic": (ii, idd)}

    rows = []
    for name, (si, sd) in S.items():
        auc = roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(sd))], np.r_[si, sd])
        r = {"method": name, "auroc_dm_vs_id": auc}
        d2 = dm.assign(score=sd)
        for fpr in (0.01, 0.05, 0.10, 0.20):
            thr = np.quantile(si, 1 - fpr)
            rep = (d2.assign(flag=(d2.score > thr).astype(float))
                     .groupby(["up", "sample"]).flag.mean().reset_index())
            sl, _, rr, pp, se = stats.linregress(rep.up, rep.flag)
            r[f"slope_{int(fpr*100)}"] = sl; r[f"p_{int(fpr*100)}"] = pp
        rows.append(r)
        print(f"  {name:16s} AUROC={auc:.3f}  " +
              "  ".join(f"{f}%:{r[f'slope_{f}']:+.4f}(p={r[f'p_{f}']:.3f})" for f in (1,5,10,20)),
              flush=True)
        np.save(f"results/dm_scores_{name}_id.npy", si)
        np.save(f"results/dm_scores_{name}_dm.npy", sd)
    pd.DataFrame(rows).to_csv("results/darkmatter_all_methods.csv", index=False)
    dm.to_csv("results/darkmatter_all_methods_reads.csv", index=False)


if __name__ == "__main__":
    main()
