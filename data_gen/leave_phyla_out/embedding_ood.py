#!/usr/bin/env python3
"""Feature-space OOD scores for the leave-phyla-out models.

Softmax confidence failed at this task (mean AUROC 0.514 across ten label
spaces). The standard explanation is that the softmax discards the magnitude of
evidence and keeps only the relative ordering of logits, so the informative part
of the network -- its representation -- is never consulted. Fort, Ren &
Lakshminarayanan (2021) found that with large pretrained transformers,
distance in feature space dramatically outperforms softmax-based scores.

Three scores computed from the penultimate (pre-classifier) embedding:
  mahalanobis  min over classes of the Mahalanobis distance to the class-
               conditional Gaussian, shared covariance (Lee et al. 2018)
  knn          distance to the k-th nearest training embedding, cosine-
               normalised (Sun et al. 2022)
  energy       -logsumexp(logits); keeps logit magnitude, unlike softmax
               (Liu et al. 2020)

Evaluated against the same ground truth as everything else: for each model,
eval reads from its 5 phyla are ID, all other phyla genuinely OOD.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"


def load_ckpt(ckpt_dir, device):
    """Same pattern as nn_proj/models/NT_transformer/inference.py: build from
    the hub id using the checkpoint's config, then load the saved weights. The
    checkpoints do not carry the remote-code module, so from_pretrained cannot
    be pointed at them directly."""
    cfg = transformers.AutoConfig.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB, config=cfg, trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(ckpt_dir) / "model.safetensors")), strict=True)
    return m.to(device).eval()

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed(model, tok, seqs, max_len, bs=512):
    """Penultimate features + logits. NT's classifier head is an MLP, so we take
    the encoder's mean-pooled hidden state as the representation."""
    feats, logits = [], []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=max_len).to(DEV)
        out = model(**b, output_hidden_states=True)
        h = out.hidden_states[-1]
        mask = b["attention_mask"].unsqueeze(-1).float()
        feats.append(((h * mask).sum(1) / mask.sum(1)).float().cpu())
        logits.append(out.logits.float().cpu())
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def mahalanobis(tr_f, tr_y, te_f):
    cls = np.unique(tr_y)
    mus = np.stack([tr_f[tr_y == c].mean(0) for c in cls])
    cen = np.concatenate([tr_f[tr_y == c] - mus[k] for k, c in enumerate(cls)])
    cov = np.cov(cen.T) + 1e-6 * np.eye(tr_f.shape[1])
    P = np.linalg.pinv(cov)
    d = np.stack([np.einsum("ij,jk,ik->i", te_f - m, P, te_f - m) for m in mus])
    return d.min(0)


def knn_score(tr_f, te_f, k=50):
    a = tr_f / (np.linalg.norm(tr_f, axis=1, keepdims=True) + 1e-9)
    b = te_f / (np.linalg.norm(te_f, axis=1, keepdims=True) + 1e-9)
    out = np.empty(len(b))
    for i in range(0, len(b), 2048):
        sim = b[i:i + 2048] @ a.T
        out[i:i + 2048] = -np.partition(sim, -k, axis=1)[:, -k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--n-train", type=int, default=6000, help="training embeddings to fit on")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", type=Path,
                    default=here.parent.parent / "results" / "leave_phyla_out" / "embedding_ood.csv")
    args = ap.parse_args()

    design = pd.read_csv(here / "subsets" / "design.csv")
    ev = pd.read_csv(here / "eval_pbsim_150bp.csv"); ev["true_phylum"] = ev.true_phylum.astype(str)
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    rows = []
    for i in range(10):
        ck = sorted((Path("checkpoints/leave_phyla_out") / f"subset_{i}").glob("checkpoint-*"))[-1]
        model = load_ckpt(ck, DEV).half()
        tr = pd.read_csv(here / "subsets" / f"subset_{i}.csv").sample(args.n_train, random_state=1)
        tr_f, _ = embed(model, tok, tr.sequence.values, args.max_len)
        te_f, te_lg = embed(model, tok, ev.sequence.values, args.max_len)

        phyla = design.loc[design.subset == i, "phyla"].iloc[0].split("|")
        y = (~ev.true_phylum.isin(phyla)).astype(int).values
        scores = {
            "mahalanobis": mahalanobis(tr_f, tr.labels.values, te_f),
            "knn": knn_score(tr_f, te_f, args.k),
            "energy": -(torch.logsumexp(torch.tensor(te_lg), dim=1).numpy()),
            "msp": -te_lg.max(1),
        }
        r = {"subset": i}
        for nm, s in scores.items():
            r[nm] = roc_auc_score(y, s)
        rows.append(r)
        print(f"subset_{i}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "subset"))
        del model; torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\nmean AUROC:")
    print(df.drop(columns=["subset"]).mean().sort_values(ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
