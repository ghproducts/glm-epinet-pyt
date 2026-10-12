#!/usr/bin/env python3
"""Is the reported temperature-scaling result an artifact of the fitting split?

nn_proj/models/NT_transformer/scaling.py fits T on a 10% slice of the file it is
given, and scripts/eval_clean_all.sh gives it train.csv -- so the fitted T comes
from IN-SAMPLE data. Measured consequences on id_test:

  phylum  T=1.477  ECE 0.0465 -> 0.0261   helped   (model was +0.046 over-confident)
  family  T=0.892  ECE 0.0227 -> 0.0451   HURT     (model was -0.012 under-confident)

Phylum was over-confident, so a softening T helped; family was already close to
calibrated, so a sharpening T overshot. That makes "temperature scaling is best"
a property of which rank happened to be over-confident, not of the method.

This refits T the textbook way (Guo et al. 2017: a held-out split) by cutting
id_test in half at the read level -- fit on half A, report on half B -- and
compares three temperatures on the SAME held-out half B, so the only thing
varying is where T came from.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch

TRAIN_FIT_T = {"phylum": 1.477049, "family": 0.892238}   # what the driver produced


def ece_equal_mass(conf, correct, n_bins=50):
    n = len(conf)
    if n < n_bins:
        return np.nan
    o = np.argsort(conf); c, a = conf[o], correct[o]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    return sum((hi - lo) / n * abs(a[lo:hi].mean() - c[lo:hi].mean())
               for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo)


def fit_T(logits, labels, max_iter=50):
    logits = torch.as_tensor(logits).float(); labels = torch.as_tensor(labels).long()
    log_T = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
    lf = torch.nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad(); loss = lf(logits / torch.exp(log_T), labels); loss.backward(); return loss
    opt.step(closure)
    return float(torch.exp(log_T).item())


def stats(logits, labels, T):
    p = torch.softmax(torch.as_tensor(logits).float() / T, -1)
    conf, pred = p.max(-1)
    conf = conf.numpy(); corr = (pred.numpy() == labels).astype(float)
    nll = torch.nn.functional.cross_entropy(
        torch.as_tensor(logits).float() / T, torch.as_tensor(labels).long()).item()
    return dict(acc=corr.mean(), conf=conf.mean(), overconf=conf.mean() - corr.mean(),
                ece=ece_equal_mass(conf, corr), nll=nll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", nargs="*", default=["phylum", "family"])
    ap.add_argument("--logits", default="results/logits")
    args = ap.parse_args()
    rng = np.random.default_rng(1)

    for rank in args.ranks:
        f = Path(args.logits) / f"{rank}_id_test.npz"
        if not f.exists():
            print(f"[skip] {f} not present yet"); continue
        z = np.load(f); L, Y = z["logits"], z["labels"]
        idx = rng.permutation(len(Y)); half = len(Y) // 2
        A, B = idx[:half], idx[half:]          # A = fit, B = report
        T_held = fit_T(L[A], Y[A])
        T_train = TRAIN_FIT_T[rank]

        print(f"\n=== {rank}: n={len(Y)}, fit on {len(A)} held-out reads, "
              f"report on the disjoint {len(B)} ===")
        print(f"  T fitted in-sample on train.csv (what the driver used): {T_train:.4f}")
        print(f"  T fitted on held-out half A (Guo et al. protocol):      {T_held:.4f}")
        print(f"\n  {'temperature':34s}{'acc':>8}{'conf':>8}{'overconf':>10}{'ECE':>9}{'NLL':>8}")
        for lab, T in [("T = 1  (no scaling)", 1.0),
                       (f"T = {T_train:.3f}  (in-sample fit)", T_train),
                       (f"T = {T_held:.3f}  (held-out fit)", T_held)]:
            s = stats(L[B], Y[B], T)
            print(f"  {lab:34s}{s['acc']:>8.4f}{s['conf']:>8.4f}"
                  f"{s['overconf']:>+10.4f}{s['ece']:>9.4f}{s['nll']:>8.4f}")

        # Where does ECE actually bottom out? Distinguishes "the fit was bad" from
        # "temperature scaling cannot help this model".
        grid = np.linspace(0.6, 2.2, 81)
        eces = [stats(L[B], Y[B], t)["ece"] for t in grid]
        j = int(np.argmin(eces))
        print(f"\n  best achievable on half B: T={grid[j]:.3f} -> ECE {eces[j]:.4f} "
              f"(vs {eces[np.argmin(abs(grid-1))]:.4f} unscaled)")


if __name__ == "__main__":
    main()
