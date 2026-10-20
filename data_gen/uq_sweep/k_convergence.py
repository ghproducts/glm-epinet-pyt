#!/usr/bin/env python3
"""Does K=10 samples suffice? (R2-6's remaining sub-ask.)

The manuscript uses K=10 for MC-dropout and epinet without justification. This
draws K_MAX samples once per cell and then evaluates metrics on nested prefixes
of that draw (first 5, 10, 20, ... samples). For i.i.d. samples that is
equivalent to running each K separately, but costs one pass instead of five and
removes between-run noise, so the curve reflects sample count alone.

    python k_convergence.py --datasets boundary --seeds 1

Writes summaries/k_convergence.csv: one row per (backbone, method, seed, K) with
accuracy, ECE, NLL and the mean decomposition terms.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_sweep as R  # noqa: E402
import sweep_lib as S  # noqa: E402

sys.path.insert(0, S.REPO)
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout  # noqa: E402

K_MAX = 100
K_GRID = [5, 10, 20, 50, 100]
OUT = os.path.join(HERE, "summaries")


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    return float(sum(
        (idx == b).mean() * abs(correct[idx == b].mean() - conf[idx == b].mean())
        for b in range(n_bins) if (idx == b).any()))


@torch.no_grad()
def draw_stack(bb, ds, seed, method) -> torch.Tensor:
    """[K_MAX, N, C] logits from one pass of the requested stochastic method."""
    if method == "mc_dropout":
        model = S.load_finetuned(bb, S.ckpt_path(bb.name, seed, "base"))
        enable_mc_dropout(model, p=R.MC_DROPOUT_P)
        out = []
        for batch in ds.loader(R.BATCH_SIZE):
            batch.pop("labels", None)
            batch.pop("label", None)
            inp = {k: v.to(S.DEVICE) for k, v in batch.items()}
            out.append(torch.stack([model(**inp).logits for _ in range(K_MAX)], 0).float().cpu())
        del model
        return torch.cat(out, dim=1)

    if method == "laplace":
        from nn_proj.models.laplace import (LaplaceConfig, LaplaceSeqClassifier,
                                            fit_diagonal_laplace)
        base = S.load_finetuned(bb, S.ckpt_path(bb.name, seed, "base"))
        attr = S.find_classifier_attr(base)
        cfg = LaplaceConfig(classifier_attr=attr, prior_precision=R.LAPLACE_PRIOR_PRECISION,
                            max_examples=R.LAPLACE_FIT_EXAMPLES)
        fit = S.build_train_subset(bb, ds.tokenizer, max_examples=R.LAPLACE_FIT_EXAMPLES)
        w, b, _ = fit_diagonal_laplace(base, fit.collate_ds, fit.collator, cfg,
                                       batch_size=R.BATCH_SIZE)
        model = LaplaceSeqClassifier(base, w, b, classifier_attr=attr).to(S.DEVICE).eval()
        out = []
        for batch in ds.loader(R.BATCH_SIZE):
            batch.pop("labels", None)
            batch.pop("label", None)
            inp = {k: v.to(S.DEVICE) for k, v in batch.items()}
            model(**inp)
            out.append(model.sample_logits(K_MAX).float().cpu())
        del model, base
        return torch.cat(out, dim=1)

    if method == "conv_epinet":
        from nn_proj.models.epinet.epinet import (EpinetConfig, EpinetWrapper,
                                                 HFEpinetSeqClassifier,
                                                 MLPEpinetWithConvPrior)
        ck = S.ckpt_path(bb.name, seed, "epinet_zfix")
        base_ck = S.ckpt_path(bb.name, seed, "base")
        config = S.load_config(bb, ckpt_dir=base_ck)
        base = S.load_finetuned(bb, base_ck, config=config)
        for p in base.parameters():
            p.requires_grad = False
        saved = S.load_epinet_saved_config(ck)
        cfg = (EpinetConfig(**{k: v for k, v in saved.items()
                               if k in EpinetConfig.__dataclass_fields__}) if saved
               else EpinetConfig(num_classes=2, include_inputs=True,
                                 vocab_size=config.vocab_size))
        model = HFEpinetSeqClassifier(
            EpinetWrapper(base, S.get_feature_fn(bb), cfg, epinet=MLPEpinetWithConvPrior),
            k_train=8, k_eval=K_MAX).to(S.DEVICE)
        dummy = ds.collator([ds.collate_ds[i] for i in range(2)])
        dummy = {k: v.to(S.DEVICE) for k, v in dummy.items() if k not in ("labels", "label")}
        model.wrapper(dummy, n_index_samples=1)
        miss, unexp = model.load_state_dict(S.load_checkpoint_weights(ck), strict=False)
        if miss or unexp:
            raise RuntimeError(f"epinet state_dict mismatch: {list(miss)[:4]} {list(unexp)[:4]}")
        model.eval()
        out = []
        for batch in ds.loader(R.BATCH_SIZE):
            batch.pop("labels", None)
            batch.pop("label", None)
            inp = {k: v.to(S.DEVICE) for k, v in batch.items()}
            out.append(model.wrapper(inp, n_index_samples=K_MAX, return_all=True).float().cpu())
        del model, base
        return torch.cat(out, dim=1)

    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default=",".join(S.BACKBONES))
    ap.add_argument("--methods", default="mc_dropout,laplace,conv_epinet")
    ap.add_argument("--datasets", default="boundary")
    ap.add_argument("--seeds", default="1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for dsname in args.datasets.split(","):
        for bname in args.backbones.split(","):
            bb = S.BACKBONES[bname.strip()]
            tok = S.load_tokenizer(bb)
            ds = S.DATASET_BUILDERS[dsname.strip()](bb, tok)
            labels = np.asarray(ds.labels, dtype=int)
            for method in [m.strip() for m in args.methods.split(",")]:
                for seed in [s.strip() for s in args.seeds.split(",")]:
                    tag = f"{dsname}/{bname}/{method}/seed{seed}"
                    try:
                        cs = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
                        torch.manual_seed(cs)
                        if S.DEVICE == "cuda":
                            torch.cuda.manual_seed_all(cs)
                        stack = draw_stack(bb, ds, seed, method)
                    except Exception as exc:
                        print(f"[FAILED] {tag}: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    for K in K_GRID:
                        unc = compute_uncertainty(stack[:K])
                        P = unc["mean_probs"].numpy()
                        pred = P.argmax(1)
                        correct = (pred == labels).astype(float)
                        conf = P.max(1)
                        p_true = P[np.arange(len(labels)), labels]
                        rows.append(dict(
                            dataset=dsname, backbone=bname, method=method, seed=seed, K=K,
                            n=len(labels), accuracy=float(correct.mean()),
                            ece_15bin=ece(conf, correct),
                            nll=float(-np.log(np.clip(p_true, 1e-12, None)).mean()),
                            u_total=float(unc["normalized_total_uncertainty"].mean()),
                            u_aleatoric=float(unc["normalized_aleatoric_uncertainty"].mean()),
                            u_epistemic=float(unc["normalized_epistemic_uncertainty"].mean()),
                        ))
                    print(f"  {tag}: done", flush=True)
                    del stack
                    if S.DEVICE == "cuda":
                        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no cells completed")
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "k_convergence.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path} ({len(df)} rows)")

    # how much does each metric still move between K=10 and K=100?
    piv = df.pivot_table(index=["dataset", "backbone", "method", "seed"], columns="K",
                         values=["ece_15bin", "u_epistemic", "accuracy"])
    print("\nchange from K=10 to K=100 (mean over cells):")
    for m in ["accuracy", "ece_15bin", "u_epistemic"]:
        d = (piv[(m, 100)] - piv[(m, 10)])
        print(f"  {m:12s} mean {d.mean():+.4f}   max |delta| {d.abs().max():.4f}")


if __name__ == "__main__":
    main()
