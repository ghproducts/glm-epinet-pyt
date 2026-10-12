#!/usr/bin/env python3
"""Three-tier novelty ladder for the leave-phyla-out models.

The pre-existing pbsim splits hold out families and genera, never phyla, so
`ood_novel_family` is ~98% in-distribution at phylum rank and cannot test
phylum-level novelty. `ood_nonbacterial` can (97.8% of its reads come from
phyla absent from training), and is included here as a far-OOD tier.

Tiers, scored per model against the same in-distribution reference:
  tier 0  ID          id_novel_genus reads from the model's own 5 phyla
  tier 1  near-OOD    id_novel_genus reads from other bacterial phyla
  tier 2  far-OOD     non-bacterial reads (archaea / eukaryota / viruses)

If the methods separate tier 2 but not tier 1, the negative result is bounded
to near-novelty. If they fail on tier 2 as well, they detect no novelty at all
at this read length.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_ckpt(ckpt_dir, device):
    cfg = transformers.AutoConfig.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB, config=cfg, trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(ckpt_dir) / "model.safetensors")), strict=True)
    return m.to(device).eval().half()


@torch.no_grad()
def feats_logits(model, tok, seqs, max_len, bs=512):
    fs, ls = [], []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=max_len).to(DEV)
        o = model(**b, output_hidden_states=True)
        h = o.hidden_states[-1]; m = b["attention_mask"].unsqueeze(-1).float()
        fs.append(((h * m).sum(1) / m.sum(1)).float().cpu()); ls.append(o.logits.float().cpu())
    return torch.cat(fs).numpy(), torch.cat(ls).numpy()


def mahalanobis(tr_f, tr_y, te_f):
    cls = np.unique(tr_y)
    mus = np.stack([tr_f[tr_y == c].mean(0) for c in cls])
    cen = np.concatenate([tr_f[tr_y == c] - mus[k] for k, c in enumerate(cls)])
    P = np.linalg.pinv(np.cov(cen.T) + 1e-6 * np.eye(tr_f.shape[1]))
    return np.stack([np.einsum("ij,jk,ik->i", te_f - m, P, te_f - m) for m in mus]).min(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--n-train", type=int, default=6000)
    args = ap.parse_args()

    design = pd.read_csv(here / "subsets" / "design.csv")
    near = pd.read_csv(here / "eval_pbsim_150bp.csv"); near["true_phylum"] = near.true_phylum.astype(str)
    far = pd.read_csv(here / "eval_farood_150bp.csv"); far["true_phylum"] = far.true_phylum.astype(str)
    far = far[far.tier == "nonbact"]
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    rows = []
    for i in range(10):
        ck = sorted((Path("checkpoints/leave_phyla_out") / f"subset_{i}").glob("checkpoint-*"))[-1]
        m = load_ckpt(ck, DEV)
        tr = pd.read_csv(here / "subsets" / f"subset_{i}.csv").sample(args.n_train, random_state=1)
        tr_f, _ = feats_logits(m, tok, tr.sequence.values, args.max_len)
        nf, nl = feats_logits(m, tok, near.sequence.values, args.max_len)
        ff, fl = feats_logits(m, tok, far.sequence.values, args.max_len)
        phyla = design.loc[design.subset == i, "phyla"].iloc[0].split("|")
        is_id = near.true_phylum.isin(phyla).values

        def auc(id_s, ood_s):
            y = np.r_[np.zeros(len(id_s)), np.ones(len(ood_s))]
            return roc_auc_score(y, np.r_[id_s, ood_s])

        sc = {}
        for nm, nsc, fsc in [
            ("msp", -nl.max(1), -fl.max(1)),
            ("energy", -torch.logsumexp(torch.tensor(nl), 1).numpy(),
                       -torch.logsumexp(torch.tensor(fl), 1).numpy()),
            ("mahalanobis", mahalanobis(tr_f, tr.labels.values, nf),
                            mahalanobis(tr_f, tr.labels.values, ff)),
        ]:
            sc[f"{nm}_near"] = auc(nsc[is_id], nsc[~is_id])   # ID vs other bacterial phyla
            sc[f"{nm}_far"] = auc(nsc[is_id], fsc)            # ID vs non-bacterial
        rows.append({"subset": i, **sc})
        print(f"subset_{i}: " + "  ".join(f"{k}={v:.3f}" for k, v in sc.items()), flush=True)
        del m; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv("results/leave_phyla_out/farood_ladder.csv", index=False)
    print("\nmean AUROC:")
    print(df.drop(columns=["subset"]).mean().to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
