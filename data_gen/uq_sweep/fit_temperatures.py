#!/usr/bin/env python3
"""Fit one temperature per (backbone, seed) and record it in temperatures.json.

`base_scaled` in the sweep refuses to run without an entry here, on purpose:
an earlier pass in this project reused a temperature fitted against a
checkpoint that had since been deleted, which silently mis-scales every
probability it touches.

T is fitted by minimizing NLL with LBFGS on the held-out 10% of the *training*
split (the manuscript's D_val) -- never on an evaluation set, since fitting a
calibrator on the data you then score makes the resulting ECE meaningless.

    python fit_temperatures.py --backbones DNABERT2 --seeds 1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_lib as S  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temperatures.json")
FIT_EXAMPLES = int(os.environ.get("TEMP_FIT_EXAMPLES", "3000"))


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    """Same recipe as nn_proj/models/*/scaling.py: scalar T minimizing NLL,
    optimized in log-space so T stays positive."""
    logits, labels = logits.float(), labels.long()
    log_T = torch.zeros((), requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn(logits / torch.exp(log_T), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).detach().cpu())


@torch.no_grad()
def collect_logits(model, ds) -> tuple[torch.Tensor, torch.Tensor]:
    logits_all, labels_all = [], []
    for batch in ds.loader(32):
        key = "labels" if "labels" in batch else "label"
        labels_all.append(batch.pop(key))
        inputs = {k: v.to(S.DEVICE) for k, v in batch.items()}
        logits_all.append(model(**inputs).logits.float().cpu())
    return torch.cat(logits_all), torch.cat(labels_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default=",".join(S.BACKBONES))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    temps = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {}
    for bname in [b.strip() for b in args.backbones.split(",")]:
        bb = S.BACKBONES[bname]
        tok = None
        for seed in [s.strip() for s in args.seeds.split(",")]:
            key = f"{bname}__seed{seed}"
            if key in temps and not args.force:
                print(f"[skip] {key}: already fitted T={temps[key]:.4f}")
                continue
            ckpt = S.ckpt_path(bname, seed, "base")
            if not os.path.isdir(ckpt):
                print(f"[skip] {key}: no base checkpoint")
                continue
            try:
                if tok is None:
                    tok = S.load_tokenizer(bb)
                val = S.build_train_subset(bb, tok, max_examples=FIT_EXAMPLES)
                model = S.load_finetuned(bb, ckpt)
                logits, labels = collect_logits(model, val)
                T = fit_temperature(logits, labels)
                nll_before = nn.functional.cross_entropy(logits, labels.long()).item()
                nll_after = nn.functional.cross_entropy(logits / T, labels.long()).item()
                temps[key] = T
                print(f"{key}: T={T:.4f}  val NLL {nll_before:.4f} -> {nll_after:.4f} "
                      f"(n={len(labels)})", flush=True)
                del model
                if S.DEVICE == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                import traceback
                print(f"[FAILED] {key}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

    with open(OUT_PATH, "w") as f:
        json.dump(temps, f, indent=2, sort_keys=True)
    print(f"\nwrote {OUT_PATH} ({len(temps)} entries)")


if __name__ == "__main__":
    main()
