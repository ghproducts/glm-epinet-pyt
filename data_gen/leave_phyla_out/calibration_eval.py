#!/usr/bin/env python3
"""Calibration of the leave-phyla-out models, following the paper's protocol.

ECE uses 50 equal-mass bins (Guo et al. 2017), matching
.docs/latex/UQ_plos_paper/main.tex. Reliability curves use 20 equal-width bins
for plotting only, as the paper specifies, and bins with <5 examples are
dropped.

Tiers where calibration is defined:
  id_test in-label   training species, phylum in the model's 5   (ID)
  near    in-label   novel genus, phylum in the model's 5        (near-ID)

near-OOD and far-OOD are deliberately excluded: those reads come from phyla
outside the label space, so no correct class exists and accuracy -- hence ECE
-- is undefined. Their uncertainty is an OOD-detection question (AUROC), which
is scored separately.

Methods: softmax baseline, temperature scaling, MC dropout. Temperature is fit
on one half of the ID tier and applied to the held-out half and to near-ID,
mirroring the paper's "representative validation set" and its finding that the
benefit can extend to mild shift.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nn_proj.common.utils import enable_mc_dropout

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ece_equal_mass(conf, correct, n_bins=50):
    """ECE with equal-mass bins, as the paper specifies."""
    n = len(conf)
    if n == 0:
        return np.nan
    order = np.argsort(conf)
    c, a = conf[order], correct[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        e += (hi - lo) / n * abs(a[lo:hi].mean() - c[lo:hi].mean())
    return e


def reliability(conf, correct, n_bins=20, min_count=5):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= min_count:
            out.append((0.5 * (edges[b] + edges[b + 1]), conf[m].mean(), correct[m].mean(), int(m.sum())))
    return pd.DataFrame(out, columns=["bin_mid", "mean_conf", "accuracy", "n"])


def load_ckpt(d):
    cfg = transformers.AutoConfig.from_pretrained(str(d), trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                       trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(d) / "model.safetensors")), strict=True)
    return m.to(DEV).eval().half()


@torch.no_grad()
def logits_of(m, tok, seqs, L, bs, mc=False):
    if mc:
        # NT ships hidden_dropout_prob=0.0, so flipping dropout modules to
        # train() samples nothing and MC-dropout comes out byte-identical to
        # the deterministic pass. The repo helper sets p explicitly (0.1, the
        # rate the manuscript reports) for exactly this reason.
        enable_mc_dropout(m, p=0.1)
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        out.append(m(**b).logits.float().cpu())
    return torch.cat(out)


def fit_temperature(logits, labels):
    T = torch.nn.Parameter(torch.ones(1) * 1.0)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)
    lg, lb = logits.clone(), torch.tensor(labels)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / T.clamp(min=1e-3), lb)
        loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1e-3))


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--root", type=Path, default=here.parent / "leave_phyla_out_v3")
    ap.add_argument("--ckpt-root", type=Path, default=Path("checkpoints/lpo_v3_full"))
    ap.add_argument("--tier-len", default="6000bp")
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--mc-k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results/calibration_v3.csv"))
    args = ap.parse_args()

    design = pd.read_csv(args.root / "design.csv")
    idt = pd.read_csv(args.root / f"eval_id_test_{args.tier_len}.csv")
    near = pd.read_csv(args.root / f"eval_near_{args.tier_len}.csv")
    for d in (idt, near):
        d["true_phylum"] = d.true_phylum.astype(str)
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token

    rows, rel_rows = [], []
    for i in range(10):
        phyla = design.loc[design.subset == i, "phyla"].iloc[0].split("|")
        lab = {p: k for k, p in enumerate(phyla)}
        m = load_ckpt(sorted((args.ckpt_root / f"subset_{i}").glob("checkpoint-*"))[-1])

        sets = {}
        for name, d in [("ID", idt), ("nearID", near)]:
            sub = d[d.true_phylum.isin(lab)]
            if len(sub) < 50:
                continue
            y = sub.true_phylum.map(lab).values
            sets[name] = (sub.sequence.values, y)

        lg = {k: logits_of(m, tok, v[0], args.max_len, args.bs) for k, v in sets.items()}
        mc_lg = {}
        for k, v in sets.items():
            stack = torch.stack([logits_of(m, tok, v[0], args.max_len, args.bs, mc=True)
                                 for _ in range(args.mc_k)])
            mc_lg[k] = stack
        m.eval(); del m; torch.cuda.empty_cache()

        # temperature fit on half the ID tier
        yid = sets["ID"][1]
        half = len(yid) // 2
        T = fit_temperature(lg["ID"][:half], yid[:half])

        for tier, (seqs, y) in sets.items():
            base_p = torch.softmax(lg[tier], -1).numpy()
            temp_p = torch.softmax(lg[tier] / T, -1).numpy()
            mcd_p = torch.softmax(mc_lg[tier], -1).mean(0).numpy()
            sl = slice(half, None) if tier == "ID" else slice(None)   # held-out half for ID
            yy = y[sl]
            for meth, P in [("base", base_p[sl]), ("temp_scaled", temp_p[sl]), ("mc_dropout", mcd_p[sl])]:
                conf, pred = P.max(1), P.argmax(1)
                corr = (pred == yy).astype(float)
                rows.append(dict(subset=i, tier=tier, method=meth, n=len(yy),
                                 accuracy=corr.mean(), mean_conf=conf.mean(),
                                 ece=ece_equal_mass(conf, corr), temperature=T if meth == "temp_scaled" else np.nan))
                r = reliability(conf, corr)
                r.insert(0, "method", meth); r.insert(0, "tier", tier); r.insert(0, "subset", i)
                rel_rows.append(r)
        print(f"subset_{i}: T={T:.3f} " + "  ".join(
            f"{r['tier']}/{r['method']} acc={r['accuracy']:.3f} ECE={r['ece']:.4f}"
            for r in rows[-6:]), flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        pd.concat(rel_rows).to_csv(str(args.out).replace(".csv", "_reliability.csv"), index=False)

    df = pd.DataFrame(rows)
    print("\n=== mean over 10 subsets ===")
    print(df.groupby(["tier", "method"])[["accuracy", "mean_conf", "ece"]].mean()
            .to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
