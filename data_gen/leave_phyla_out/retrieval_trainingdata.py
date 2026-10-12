#!/usr/bin/env python3
"""Retrieval against the project's own training data -- no sampling, no new sets.

This is the like-for-like comparison against Kraken2's parity run. Both index
EXACTLY the same sequences: all 20,000 6kb training windows of a given subset.
Kraken2 matches them by exact k-mers; here they are embedded and matched by
nearest-neighbour distance. Any difference is the matching method.

Earlier versions sampled 6,000 of the 20,000 windows for the kNN/Mahalanobis
reference. That was a compute shortcut, not a design choice, and it handed
Kraken2 a 3x larger index. It is also unnecessary: build_v3 draws every
subset's windows from one shared per-phylum pool, so across all 10 subsets
there are only 40,000 distinct windows. Embedding that pool once with a single
shared encoder and selecting each subset's 5 phyla from it costs 40,000
forward passes instead of 200,000, and uses every window.

The encoder is the 36-way phylum checkpoint for all subsets. Retrieval does not
use a label space at all, so there is no reason to tie the encoder to a
subset's 5 classes -- and holding it fixed means the only thing varying across
subsets is which phyla count as known.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
FINETUNED = "old_models/trained_models_1/NT_transformer/pbsim_phylum/checkpoint-60708"
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
        if tag and i % (bs * 100) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()

    root = here.parent / "leave_phyla_out_v3"
    design = pd.read_csv(root / "design.csv")
    tiers = {t: pd.read_csv(root / f"eval_{t}_{args.read_len}bp.csv")
             for t in ["id_test", "near", "far"]}
    for d in tiers.values():
        d["true_phylum"] = d.true_phylum.astype(str)

    # the shared window pool: every distinct training window, kept whole
    pool = {}
    for i in range(10):
        d = pd.read_csv(root / "subsets" / f"subset_{i}.csv")
        for p, g in d.groupby("phylum"):
            pool.setdefault(str(p), set()).update(g.sequence)
    pool = {p: sorted(v) for p, v in pool.items()}
    print(f"reference pool: {sum(len(v) for v in pool.values()):,} windows across {len(pool)} phyla")

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    cfg = transformers.AutoConfig.from_pretrained(FINETUNED, trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                       trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(FINETUNED) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()

    print("embedding reference pool (once, reused by every subset) ...")
    emb = {p: embed(m, tok, v, args.max_len, args.bs, tag=f"ref[{p}]") for p, v in pool.items()}
    print("embedding evaluation tiers ...")
    qemb = {t: embed(m, tok, d.sequence.values, args.max_len, args.bs, tag=t)
            for t, d in tiers.items()}
    del m; torch.cuda.empty_cache()

    def nn_score(R, Q):
        R = R.to(DEV); s = np.empty(len(Q))
        for i in range(0, len(Q), 128):
            sim = Q[i:i + 128].to(DEV) @ R.T
            s[i:i + 128] = (-sim.topk(args.k, 1).values[:, -1]).cpu().numpy()
        return s

    def auc(a, b):
        return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])

    rows = []
    for i in range(10):
        phyla = design.loc[design.subset == i, "phyla"].iloc[0].split("|")
        R = torch.cat([emb[p] for p in phyla])          # ALL windows of those 5 phyla
        idt = tiers["id_test"].true_phylum.isin(phyla).values
        near = tiers["near"]; in_lab = near.true_phylum.isin(phyla).values
        s_id = nn_score(R, qemb["id_test"][idt])
        s_near = nn_score(R, qemb["near"])
        s_far = nn_score(R, qemb["far"])
        r = dict(subset=i, n_ref=len(R),
                 idtest_v_nearid=auc(s_id, s_near[in_lab]),
                 idtest_v_nearood=auc(s_id, s_near[~in_lab]),
                 idtest_v_farood=auc(s_id, s_far),
                 nearid_v_nearood=auc(s_near[in_lab], s_near[~in_lab]))
        rows.append(r)
        print(f"subset_{i}: n_ref={len(R)}  ID/nearOOD={r['idtest_v_nearood']:.3f}  "
              f"ID/far={r['idtest_v_farood']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv("results/retrieval_trainingdata.csv", index=False)

    df = pd.DataFrame(rows)
    print("\nmean:")
    print(df.drop(columns=["subset"]).mean().to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nKraken2 on the identical windows: ID/nearOOD ~0.55, ID/far ~0.55")


if __name__ == "__main__":
    main()
