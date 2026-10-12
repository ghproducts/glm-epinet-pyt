#!/usr/bin/env python3
"""Score the outlier-exposure and deep-ensemble models for OOD detection.

Both are trained fixes rather than post-hoc scores, and are the two remaining
candidates after softmax, energy, Mahalanobis and kNN all landed at ~0.5.

  outlier exposure  novelty = P(background class). Scoring EXCLUDES the two
                    exposure phyla each model saw as background, so this
                    measures generalisation to unseen novelty rather than
                    recall of the exposure set.
  deep ensemble     5 seeds, identical label space (subset_0). Scores: mean
                    max-prob, BALD mutual information, and the variance-based
                    epistemic term from nn_proj.common.variance_decomp.
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
def probs(model, tok, seqs, max_len, bs=512):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=max_len).to(DEV)
        out.append(torch.softmax(model(**b).logits.float(), -1).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--max-len", type=int, default=48)
    args = ap.parse_args()

    ev = pd.read_csv(here / "eval_pbsim_150bp.csv"); ev["true_phylum"] = ev.true_phylum.astype(str)
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token

    # ---------- outlier exposure ----------
    oe = pd.read_csv(here / "subsets_oe" / "design_oe.csv")
    rows = []
    for _, r in oe.iterrows():
        i = int(r.subset)
        ck = sorted((Path("checkpoints/oe") / f"subset_{i}").glob("checkpoint-*"))[-1]
        m = load_ckpt(ck, DEV)
        p = probs(m, tok, ev.sequence.values, args.max_len)
        inlab = r.in_label.split("|"); expo = set(r.exposure.split("|"))
        keep = ~ev.true_phylum.isin(expo)          # drop the exposure phyla
        y = (~ev.true_phylum.isin(inlab))[keep].astype(int).values
        bg = p[keep.values, len(inlab)]            # P(background)
        msp = -p[keep.values, :len(inlab)].max(1)  # in-label max prob, inverted
        rows.append(dict(subset=i, auroc_background=roc_auc_score(y, bg),
                         auroc_msp=roc_auc_score(y, msp),
                         n_id=int((y == 0).sum()), n_ood=int((y == 1).sum())))
        print(f"OE subset_{i}: P(background) AUROC={rows[-1]['auroc_background']:.3f}  "
              f"msp AUROC={rows[-1]['auroc_msp']:.3f}")
        del m; torch.cuda.empty_cache()
    oe_df = pd.DataFrame(rows)
    oe_df.to_csv("results/leave_phyla_out/outlier_exposure.csv", index=False)
    print(f"\nOE mean: P(background) AUROC = {oe_df.auroc_background.mean():.3f}, "
          f"msp = {oe_df.auroc_msp.mean():.3f}\n")

    # ---------- deep ensemble ----------
    design = pd.read_csv(here / "subsets" / "design.csv")
    phyla = design.loc[design.subset == 0, "phyla"].iloc[0].split("|")
    y = (~ev.true_phylum.isin(phyla)).astype(int).values
    members = []
    ck0 = sorted((Path("checkpoints/leave_phyla_out") / "subset_0").glob("checkpoint-*"))[-1]
    paths = [ck0] + [sorted((Path("checkpoints/ensemble") / f"seed_{s}").glob("checkpoint-*"))[-1]
                     for s in (2, 3, 4, 5, 6)]
    for pth in paths:
        m = load_ckpt(pth, DEV)
        members.append(probs(m, tok, ev.sequence.values, args.max_len))
        del m; torch.cuda.empty_cache()
    P = np.stack(members)                       # [K, N, C]
    pbar = P.mean(0)
    ent = lambda q: -(q * np.log(q + 1e-12)).sum(-1)
    total = ent(pbar); alea = ent(P).mean(0); epi = total - alea          # BALD
    var_epi = P.var(0).sum(-1)                                            # variance-based
    res = {"ensemble_msp": roc_auc_score(y, -pbar.max(1)),
           "ensemble_total_entropy": roc_auc_score(y, total),
           "ensemble_BALD_epistemic": roc_auc_score(y, epi),
           "ensemble_variance_epistemic": roc_auc_score(y, var_epi)}
    print(f"DEEP ENSEMBLE ({len(paths)} members, label space = subset_0):")
    for k, v in res.items():
        print(f"  {k:30s} AUROC = {v:.3f}")
    pd.DataFrame([res]).to_csv("results/leave_phyla_out/ensemble.csv", index=False)


if __name__ == "__main__":
    main()
