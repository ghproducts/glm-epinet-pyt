#!/usr/bin/env python3
"""k-sweep for the kNN novelty score on dark matter.

kNN is the only score showing any positive `up` trend, and it appears only in
the extreme tail -- which is precisely what k controls. k=1 asks "is there any
single close match in the reference?", k=50 asks "how dense is the reference
around this point?". Fixing k=50 by convention (Sun et al.) is defensible for a
general benchmark but is the wrong thing to hold constant when the question is
whether a tail effect exists.

Embeds once, scores at every k, then sweeps the id_test-calibrated FPR
threshold for each.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
CLEAN = "checkpoints/pbsim_clean/phylum/checkpoint-57734"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed(m, tok, seqs, L, bs, tag=""):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        o = m(**b, output_hidden_states=True)
        h = o.hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        if tag and i % (bs * 200) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 5, 10, 20, 50, 100])
    ap.add_argument("--n-ref", type=int, default=40000)
    args = ap.parse_args()
    rng = np.random.default_rng(1); L = args.read_len
    cut = lambda ss: [s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]
                      for s in ss]

    ref = pd.concat([c for c in pd.read_csv(D / "train.csv", usecols=["sequence"],
                                            chunksize=50_000)], ignore_index=True)
    ref_w = cut(ref.sample(args.n_ref, random_state=1).sequence.values) * 2
    idt = cut(pd.read_csv(D / "id_test.csv", usecols=["sequence"])
                .sample(6000, random_state=1).sequence.values)
    dm = pd.read_csv("results/darkmatter_all_methods_reads.csv")

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    cfg = transformers.AutoConfig.from_pretrained(CLEAN, trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                        trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CLEAN) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()
    R = embed(m, tok, ref_w, args.max_len, args.bs, "ref").to(DEV)
    Ei = embed(m, tok, idt, args.max_len, args.bs, "id")
    Ed = embed(m, tok, dm.sequence.values, args.max_len, args.bs, "dm")
    del m; torch.cuda.empty_cache()

    kmax = max(args.ks)
    def topk(E):
        out = np.empty((len(E), kmax))
        for i in range(0, len(E), 256):
            sim = E[i:i + 256].to(DEV) @ R.T
            out[i:i + 256] = sim.topk(kmax, 1).values.cpu().numpy()
        return out
    Ti, Td = topk(Ei), topk(Ed)

    fprs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    print(f"\n{'k':>5}{'AUROC id/dm':>13}{'AUROC up.25v1':>15}" +
          "".join(f"{f:>9.1%}" for f in fprs))
    rows = []
    for k in args.ks:
        si, sd = -Ti[:, k - 1], -Td[:, k - 1]
        d = dm.assign(score=sd)
        auc = roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(sd))], np.r_[si, sd])
        a, b = d[d.up == 0.25].score.values, d[d.up == 1.0].score.values
        auc_up = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
        line = f"{k:>5}{auc:>13.3f}{auc_up:>15.3f}"
        rec = {"k": k, "auroc_id_dm": auc, "auroc_up25_v_up100": auc_up}
        for f in fprs:
            T = np.quantile(si, 1 - f)
            rep = (d.assign(flag=(d.score > T).astype(float))
                     .groupby(["up", "sample"]).flag.mean().reset_index())
            sl, _, _, p, _ = stats.linregress(rep.up, rep.flag)
            rec[f"slope_{f}"] = sl; rec[f"p_{f}"] = p
            line += f"{sl:>+8.4f}{'*' if p < 0.05 else ' '}"
        rows.append(rec); print(line, flush=True)
    pd.DataFrame(rows).to_csv("results/darkmatter_knn_k_sweep.csv", index=False)
    print("\n* p<0.05.  AUROC up.25v1 is threshold-free: >0.5 means up=1.00 reads score more novel.")


if __name__ == "__main__":
    main()
