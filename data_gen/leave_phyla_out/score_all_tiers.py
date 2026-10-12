#!/usr/bin/env python3
"""Every GLM-based OOD score across the full novelty ladder.

Tiers (all 150bp, all scored by the same model):
  true_id    held-out windows from the TRAINING species            (in-distribution)
  near_id    id_novel_genus reads from the model's own phyla       (novel genus)
  near_ood   id_novel_genus reads from other bacterial phyla       (novel phylum)
  far_ood    non-bacterial reads                                    (novel superkingdom)

Reporting true_id as the reference as well as near_id separates two things that
were conflated before: whether a method detects novelty at all, and whether it
detects the specific step from "novel genus" to "novel phylum".
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_ckpt(d, device):
    cfg = transformers.AutoConfig.from_pretrained(str(d), trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                       trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(d) / "model.safetensors")), strict=True)
    return m.to(device).eval().half()


@torch.no_grad()
def fl(model, tok, seqs, L, bs=512):
    fs, ls = [], []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        o = model(**b, output_hidden_states=True)
        h = o.hidden_states[-1]; m = b["attention_mask"].unsqueeze(-1).float()
        fs.append(((h * m).sum(1) / m.sum(1)).float().cpu()); ls.append(o.logits.float().cpu())
    return torch.cat(fs).numpy(), torch.cat(ls).numpy()


def maha_fit(tr_f, tr_y):
    cls = np.unique(tr_y)
    mus = np.stack([tr_f[tr_y == c].mean(0) for c in cls])
    cen = np.concatenate([tr_f[tr_y == c] - mus[k] for k, c in enumerate(cls)])
    return mus, np.linalg.pinv(np.cov(cen.T) + 1e-6 * np.eye(tr_f.shape[1]))


def maha(mus, P, f):
    return np.stack([np.einsum("ij,jk,ik->i", f - m, P, f - m) for m in mus]).min(0)


def knn(tr_f, f, k=50):
    a = tr_f / (np.linalg.norm(tr_f, axis=1, keepdims=True) + 1e-9)
    b = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    o = np.empty(len(b))
    for i in range(0, len(b), 2048):
        o[i:i + 2048] = -np.partition(b[i:i + 2048] @ a.T, -k, axis=1)[:, -k]
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--n-train", type=int, default=6000)
    args = ap.parse_args()

    design = pd.read_csv(here / "subsets" / "design.csv")
    oe_des = pd.read_csv(here / "subsets_oe" / "design_oe.csv")
    near = pd.read_csv(here / "eval_pbsim_150bp.csv"); near["true_phylum"] = near.true_phylum.astype(str)
    far = pd.read_csv(here / "eval_farood_150bp.csv"); far = far[far.tier == "nonbact"]
    tid = pd.read_csv(here / "eval_trueid_150bp.csv"); tid["true_phylum"] = tid.true_phylum.astype(str)
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token

    def auc(a, b):
        return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])

    rows = []
    for i in range(10):
        phyla = design.loc[design.subset == i, "phyla"].iloc[0].split("|")
        tid_i = tid[tid.true_phylum.isin(phyla)]          # true-ID restricted to this label space
        ck = sorted((Path("checkpoints/leave_phyla_out") / f"subset_{i}").glob("checkpoint-*"))[-1]
        m = load_ckpt(ck, DEV)
        tr = pd.read_csv(here / "subsets" / f"subset_{i}.csv").sample(args.n_train, random_state=1)
        trf, _ = fl(m, tok, tr.sequence.values, args.max_len)
        mus, P = maha_fit(trf, tr.labels.values)
        packs = {}
        for nm, seqs in [("true_id", tid_i.sequence.values), ("near", near.sequence.values),
                         ("far", far.sequence.values)]:
            packs[nm] = fl(m, tok, seqs, args.max_len)
        is_id = near.true_phylum.isin(phyla).values
        del m; torch.cuda.empty_cache()

        S = {}
        for nm, (f_, l_) in packs.items():
            S[(nm, "msp")] = -l_.max(1)
            S[(nm, "energy")] = -torch.logsumexp(torch.tensor(l_), 1).numpy()
            S[(nm, "mahalanobis")] = maha(mus, P, f_)
            S[(nm, "knn")] = knn(trf, f_)

        r = {"subset": i}
        for sc in ["msp", "energy", "mahalanobis", "knn"]:
            nid, nood = S[("near", sc)][is_id], S[("near", sc)][~is_id]
            t, fo = S[("true_id", sc)], S[("far", sc)]
            r[f"{sc}__nearID_v_nearOOD"] = auc(nid, nood)
            r[f"{sc}__trueID_v_nearOOD"] = auc(t, nood)
            r[f"{sc}__trueID_v_farOOD"] = auc(t, fo)
            r[f"{sc}__trueID_v_nearID"] = auc(t, nid)
        rows.append(r)
        print(f"subset_{i} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/leave_phyla_out/all_tiers_glm.csv", index=False)
    mean = df.drop(columns=["subset"]).mean()
    print("\n{:<14}{:>12}{:>12}{:>12}{:>12}".format(
        "score", "trueID/nearID", "trueID/nearOOD", "trueID/farOOD", "nearID/nearOOD"))
    for sc in ["msp", "energy", "mahalanobis", "knn"]:
        print("{:<14}{:>12.3f}{:>12.3f}{:>12.3f}{:>12.3f}".format(
            sc, mean[f"{sc}__trueID_v_nearID"], mean[f"{sc}__trueID_v_nearOOD"],
            mean[f"{sc}__trueID_v_farOOD"], mean[f"{sc}__nearID_v_nearOOD"]))


if __name__ == "__main__":
    main()
