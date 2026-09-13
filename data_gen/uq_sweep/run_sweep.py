#!/usr/bin/env python3
"""UQ sweep driver: every backbone x every method x every eval dataset.

Replaces the per-run hardcoded scripts with one resumable grid, so the set of
runs is specified in code rather than by hand-editing constants between runs.

    # one cell
    python run_sweep.py --backbones DNABERT2 --methods base --datasets boundary
    # everything that has the checkpoints it needs
    python run_sweep.py --all
    # what would run, and what is missing, without running it
    python run_sweep.py --all --dry-run

Each cell writes results/<dataset>/<backbone>/<method>__seed<N>.csv with one
row per example, including the full mean predictive distribution (prob_0,
prob_1, ...) so ECE/NLL/Brier are computable directly.

Cells whose checkpoints don't exist are reported as skipped, never silently
dropped or substituted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sweep_lib as S  # noqa: E402

sys.path.insert(0, S.REPO)
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout  # noqa: E402

BATCH_SIZE = int(os.environ.get("SWEEP_BATCH_SIZE", "32"))
K_MC = 10
K_EPINET = 10
K_LAPLACE = 16
LAPLACE_FIT_EXAMPLES = 2000
LAPLACE_PRIOR_PRECISION = 1.0
MC_DROPOUT_P = 0.1
ENSEMBLE_SEEDS = {"ensemble_k3": [1, 2, 3], "ensemble_k5": [1, 2, 3, 4, 42]}
CONFORMAL_ALPHA = 0.1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rows_from_uncertainty(ds: S.EvalDataset, unc: dict, labels: torch.Tensor,
                           method: str, start: int, extra: dict | None = None) -> list[dict]:
    """Common per-example row builder. `unc` is compute_uncertainty's output."""
    metas = ds.metadata_rows(start, start + labels.shape[0])
    rows = []
    for i in range(labels.shape[0]):
        pred = int(unc["predicted_class"][i])
        label = int(labels[i])
        row = {
            **metas[i], "method": method, "row_idx": start + i,
            "U_total": float(unc["normalized_total_uncertainty"][i]),
            "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
            "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
            "pred": pred, "labels": label, "correct": int(pred == label),
        }
        row.update(S.probs_to_row_fields(unc["mean_probs"][i]))
        if extra:
            row.update({k: (v[i] if isinstance(v, (list, torch.Tensor)) else v)
                        for k, v in extra.items()})
        rows.append(row)
    return rows


@torch.no_grad()
def _stacked_logits_pass(model, ds, k: int, resample_each: bool) -> tuple[list, list]:
    """Run the loader once, returning (per-batch [K,B,C] logit stacks, labels)."""
    loader = ds.loader(BATCH_SIZE)
    stacks, all_labels = [], []
    for batch in loader:
        key = "labels" if "labels" in batch else "label"
        labels = batch.pop(key)
        inputs = {kk: v.to(S.DEVICE) for kk, v in batch.items()}
        if resample_each:
            logits = torch.stack([model(**inputs).logits for _ in range(k)], dim=0)
        else:
            logits = model(**inputs).logits.unsqueeze(0)
        stacks.append(logits.float().cpu())
        all_labels.append(labels)
    return stacks, all_labels


def _emit(ds, stacks, all_labels, method, extra_per_batch=None) -> list[dict]:
    rows, start = [], 0
    for bi, (logits, labels) in enumerate(zip(stacks, all_labels)):
        unc = compute_uncertainty(logits)
        extra = extra_per_batch[bi] if extra_per_batch else None
        rows.extend(_rows_from_uncertainty(ds, unc, labels, method, start, extra))
        start += labels.shape[0]
    return rows


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

def m_base(bb, ds, seed, args):
    model = S.load_finetuned(bb, S.ckpt_path(bb.name, seed, "base"))
    stacks, labels = _stacked_logits_pass(model, ds, 1, resample_each=False)
    del model
    return _emit(ds, stacks, labels, "base")


def m_base_scaled(bb, ds, seed, args):
    """Temperature fitted on the held-out 10% of the training split (the same
    D_val the manuscript selects checkpoints on), never on the eval set."""
    t_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temperatures.json")
    temps = json.load(open(t_path)) if os.path.exists(t_path) else {}
    key = f"{bb.name}__seed{seed}"
    if key not in temps:
        raise RuntimeError(
            f"no fitted temperature for {key}; run fit_temperatures.py first "
            f"(refusing to reuse another checkpoint's T)"
        )
    T = float(temps[key])
    model = S.load_finetuned(bb, S.ckpt_path(bb.name, seed, "base"))
    stacks, labels = _stacked_logits_pass(model, ds, 1, resample_each=False)
    del model
    stacks = [s / T for s in stacks]
    rows = _emit(ds, stacks, labels, "base_scaled")
    for r in rows:
        r["temperature"] = T
    return rows


def m_mc_dropout(bb, ds, seed, args):
    model = S.load_finetuned(bb, S.ckpt_path(bb.name, seed, "base"))
    enable_mc_dropout(model, p=MC_DROPOUT_P)
    stacks, labels = _stacked_logits_pass(model, ds, K_MC, resample_each=True)
    del model
    return _emit(ds, stacks, labels, "mc_dropout")


def m_conv_epinet(bb, ds, seed, args):
    """Requires an epinet trained under the per-example-z fix. The pre-fix
    checkpoints (`epinet`) are refused rather than silently used: they were
    trained under batch-shared z, so evaluating them with fixed code measures
    neither the old nor the new pipeline."""
    from nn_proj.models.epinet.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier
    kind = args.epinet_kind
    ckpt = S.ckpt_path(bb.name, seed, kind)
    if kind == "epinet":
        raise RuntimeError(
            "refusing pre-z-fix 'epinet' checkpoints; retrain as 'epinet_zfix' "
            "or pass --epinet-kind explicitly if you really mean the old weights"
        )
    base = S.load_finetuned(bb, S.ckpt_path(bb.name, seed, "base"))
    for p in base.parameters():
        p.requires_grad = False
    cfg_path = os.path.join(ckpt, "epinet_config.json")
    cfg_kwargs = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    wrapper = EpinetWrapper(base, S.get_feature_fn(bb), EpinetConfig(**cfg_kwargs))
    model = HFEpinetSeqClassifier(wrapper).to(S.DEVICE)
    # epinet internals are lazily built on first forward
    loader = ds.loader(BATCH_SIZE)
    first = next(iter(loader))
    first.pop("labels" if "labels" in first else "label")
    with torch.no_grad():
        model(**{k: v.to(S.DEVICE) for k, v in first.items()})
    missing, unexpected = model.load_state_dict(
        S.load_checkpoint_weights(ckpt), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"epinet state_dict mismatch: missing={list(missing)[:5]} "
                           f"unexpected={list(unexpected)[:5]}")
    model.eval()
    stacks, labels = _stacked_logits_pass(model, ds, K_EPINET, resample_each=True)
    del model, base
    return _emit(ds, stacks, labels, "conv_epinet")


def m_laplace(bb, ds, seed, args):
    from nn_proj.models.laplace import LaplaceConfig, LaplaceSeqClassifier, fit_diagonal_laplace
    base = S.load_finetuned(bb, S.ckpt_path(bb.name, seed, "base"))
    attr = S.find_classifier_attr(base)
    cfg = LaplaceConfig(classifier_attr=attr, prior_precision=LAPLACE_PRIOR_PRECISION,
                        max_examples=LAPLACE_FIT_EXAMPLES)
    fit_ds = S.build_train_subset(bb, ds.tokenizer, max_examples=LAPLACE_FIT_EXAMPLES)
    w_var, b_var, n_seen = fit_diagonal_laplace(base, fit_ds.collate_ds, fit_ds.collator,
                                                cfg, batch_size=BATCH_SIZE)
    model = LaplaceSeqClassifier(base, w_var, b_var, classifier_attr=attr).to(S.DEVICE).eval()

    loader = ds.loader(BATCH_SIZE)
    stacks, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            key = "labels" if "labels" in batch else "label"
            labels = batch.pop(key)
            inputs = {k: v.to(S.DEVICE) for k, v in batch.items()}
            model(**inputs)                       # populates the feature hook
            stacks.append(model.sample_logits(K_LAPLACE).float().cpu())
            all_labels.append(labels)
    del model, base
    rows = _emit(ds, stacks, all_labels, "laplace")
    for r in rows:
        r["laplace_fit_examples"] = n_seen
        r["laplace_classifier_attr"] = attr
    return rows


def m_evidential(bb, ds, seed, args):
    from nn_proj.models.evidential import EvidentialConfig, EvidentialWrapper, HFEvidentialSeqClassifier
    ckpt = S.ckpt_path(bb.name, seed, "evidential")
    saved = {}
    cfg_path = os.path.join(ckpt, "evidential_config.json")
    if os.path.exists(cfg_path):
        saved = json.load(open(cfg_path))
    evid_cfg = EvidentialConfig(
        num_classes=2,
        evidence_activation=saved.get("evidence_activation", "softplus"),
        loss_type=saved.get("loss_type", "mse"),
        annealing_step=saved.get("annealing_step", 10),
    )
    base = S.build_base_arch(bb, ckpt_dir=ckpt if bb.custom_class else None)
    model = HFEvidentialSeqClassifier(EvidentialWrapper(base, evid_cfg)).to(S.DEVICE)
    missing, unexpected = model.load_state_dict(S.load_checkpoint_weights(ckpt), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"evidential state_dict mismatch: missing={list(missing)[:5]} "
                           f"unexpected={list(unexpected)[:5]}")
    model.eval()

    loader = ds.loader(BATCH_SIZE)
    rows, start = [], 0
    with torch.no_grad():
        for batch in loader:
            key = "labels" if "labels" in batch else "label"
            labels = batch.pop(key)
            inputs = {k: v.to(S.DEVICE) for k, v in batch.items()}
            evidence = model(**inputs)["logits"].detach().cpu().float()
            alpha = evidence + 1.0
            Ssum = alpha.sum(-1, keepdim=True)
            probs = alpha / Ssum
            unc = compute_uncertainty(torch.log(probs.clamp_min(1e-12)).unsqueeze(0))
            # single-Dirichlet: U_epistemic is 0 by construction; vacuity C/S is
            # the epistemic-axis analogue and is reported in its own column.
            vac = (evid_cfg.num_classes / Ssum.squeeze(-1)).tolist()
            rows.extend(_rows_from_uncertainty(ds, unc, labels, "evidential", start,
                                               extra={"vacuity": vac,
                                                      "dirichlet_strength": Ssum.squeeze(-1).tolist()}))
            start += labels.shape[0]
    del model, base
    return rows


def m_ensemble(bb, ds, seed, args, method="ensemble_k3"):
    """Independently fine-tuned checkpoints as the K samples. `seed` is ignored
    (the ensemble *is* the seed set); output is filed under seed='ens'."""
    seeds = ENSEMBLE_SEEDS[method]
    per_member = []
    for sd in seeds:
        model = S.load_finetuned(bb, S.ckpt_path(bb.name, sd, "base"))
        stacks, labels = _stacked_logits_pass(model, ds, 1, resample_each=False)
        del model
        if S.DEVICE == "cuda":
            torch.cuda.empty_cache()
        per_member.append(stacks)
        last_labels = labels
    combined = [torch.cat([per_member[m][b] for m in range(len(seeds))], dim=0)
                for b in range(len(per_member[0]))]
    rows = _emit(ds, combined, last_labels, method)
    for r in rows:
        r["ensemble_seeds"] = ",".join(str(s) for s in seeds)
    return rows


def m_conformal(bb, ds, seed, args):
    """Split conformal prediction sets, calibrated on the held-out 10% of the
    training split so no evaluation data is used to calibrate.

    Note on coverage: conformal's 1-alpha guarantee needs calibration and
    evaluation data to be exchangeable. Calibrating on train-val and scoring the
    *test* split slightly violates that (observed ~0.87 vs nominal 0.90 for
    DNABERT2 on the boundary set), and the AliSim set violates it on purpose.
    The shortfall is a result to report, not a bug -- `conformal_cal_source` and
    `conformal_qhat` are recorded per row so it stays auditable.

    `--conformal-score aps` is available but non-randomized APS degenerates on a
    2-class task (every set becomes all classes: coverage 1.0, size 2.0), so the
    default is the standard 1-p_true score, which stays adaptive.
    """
    from nn_proj.common.conformal import (
        aps_nonconformity, fit_conformal_threshold, predict_sets, score_nonconformity,
    )
    base = S.load_finetuned(bb, S.ckpt_path(bb.name, seed, "base"))
    cal = S.build_train_subset(bb, ds.tokenizer)
    with torch.no_grad():
        cal_stacks, cal_labels = _stacked_logits_pass(base, cal, 1, resample_each=False)
    cal_probs = torch.cat([s.squeeze(0).softmax(-1) for s in cal_stacks], 0)
    cal_y = torch.cat(cal_labels, 0)
    score_fn = aps_nonconformity if args.conformal_score == "aps" else score_nonconformity
    qhat = fit_conformal_threshold(score_fn(cal_probs, cal_y), alpha=CONFORMAL_ALPHA)

    stacks, labels = _stacked_logits_pass(base, ds, 1, resample_each=False)
    del base
    rows, start = [], 0
    for logits, lab in zip(stacks, labels):
        probs = logits.squeeze(0).softmax(-1)
        sets = predict_sets(probs, qhat, method=args.conformal_score)
        unc = compute_uncertainty(logits)
        covered = [int(bool(sets[i, int(lab[i])])) for i in range(lab.shape[0])]
        rows.extend(_rows_from_uncertainty(
            ds, unc, lab, "conformal", start,
            extra={"set_size": sets.sum(-1).tolist(), "covered": covered,
                   "conformal_qhat": float(qhat),
                   "conformal_score": args.conformal_score,
                   "conformal_cal_source": "train_val_10pct",
                   "conformal_alpha": CONFORMAL_ALPHA}))
        start += lab.shape[0]
    return rows


METHODS = {
    "base": m_base,
    "base_scaled": m_base_scaled,
    "mc_dropout": m_mc_dropout,
    "conv_epinet": m_conv_epinet,
    "laplace": m_laplace,
    "evidential": m_evidential,
    "ensemble_k3": lambda bb, ds, sd, a: m_ensemble(bb, ds, sd, a, "ensemble_k3"),
    "ensemble_k5": lambda bb, ds, sd, a: m_ensemble(bb, ds, sd, a, "ensemble_k5"),
    "conformal": m_conformal,
}
ENSEMBLE_METHODS = {"ensemble_k3", "ensemble_k5"}
# checkpoint flavour each method needs, for the dry-run availability report
NEEDS = {"base": "base", "base_scaled": "base", "mc_dropout": "base", "laplace": "base",
         "conformal": "base", "conv_epinet": "epinet_zfix", "evidential": "evidential",
         "ensemble_k3": "base", "ensemble_k5": "base"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="DNABERT2")
    ap.add_argument("--methods", default="base")
    ap.add_argument("--datasets", default="boundary")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--all", action="store_true",
                    help="all backbones x all methods x both datasets x seeds 1,2,3")
    ap.add_argument("--epinet-kind", default="epinet_zfix")
    ap.add_argument("--conformal-score", default="score", choices=["score", "aps"],
                    help="APS is non-randomized here and degenerates to all-classes "
                         "sets on a 2-class task (coverage 1.0, size 2.0); the "
                         "standard 1-p_true score stays adaptive")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute cells that already exist")
    args = ap.parse_args()

    if args.all:
        backbones = list(S.BACKBONES)
        methods = list(METHODS)
        datasets = ["boundary", "alisim"]
        seeds = ["1", "2", "3"]
    else:
        backbones = [b.strip() for b in args.backbones.split(",")]
        methods = [m.strip() for m in args.methods.split(",")]
        datasets = [d.strip() for d in args.datasets.split(",")]
        seeds = [s.strip() for s in args.seeds.split(",")]

    plan, skipped = [], []
    for dsname in datasets:
        for bname in backbones:
            for mname in methods:
                for sd in seeds:
                    eff_seed = "ens" if mname in ENSEMBLE_METHODS else sd
                    if mname in ENSEMBLE_METHODS and sd != seeds[0]:
                        continue  # ensemble is seed-set-defined; run once
                    need_seeds = ENSEMBLE_SEEDS[mname] if mname in ENSEMBLE_METHODS else [sd]
                    missing = [s2 for s2 in need_seeds
                               if not os.path.isdir(S.ckpt_path(bname, s2, NEEDS[mname]))]
                    if missing:
                        skipped.append((dsname, bname, mname, eff_seed,
                                        f"missing {NEEDS[mname]} for seed(s) {missing}"))
                        continue
                    out = S.cell_path(dsname, bname, mname, eff_seed)
                    if os.path.exists(out) and not args.force:
                        skipped.append((dsname, bname, mname, eff_seed, "already done"))
                        continue
                    plan.append((dsname, bname, mname, eff_seed, sd))

    print(f"planned cells: {len(plan)}   skipped: {len(skipped)}")
    for s in skipped:
        print(f"  [skip] {s[0]}/{s[1]}/{s[2]} seed {s[3]}: {s[4]}")
    for p in plan:
        print(f"  [run ] {p[0]}/{p[1]}/{p[2]} seed {p[3]}")
    if args.dry_run:
        return

    # datasets are tokenizer-specific, so cache per (backbone, dataset)
    cache: dict[tuple[str, str], S.EvalDataset] = {}
    failures = []
    for dsname, bname, mname, eff_seed, raw_seed in plan:
        bb = S.BACKBONES[bname]
        try:
            if (bname, dsname) not in cache:
                tok = S.load_tokenizer(bb)
                cache[(bname, dsname)] = S.DATASET_BUILDERS[dsname](bb, tok)
            ds = cache[(bname, dsname)]
            print(f"=== {dsname} / {bname} / {mname} / seed {eff_seed} ===", flush=True)
            rows = METHODS[mname](bb, ds, raw_seed, args)
            path = S.write_cell(rows, dsname, bname, mname, eff_seed)
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"    wrote {path}  n={len(rows)}  acc={acc:.4f}", flush=True)
        except Exception as exc:
            import traceback
            print(f"[FAILED] {dsname}/{bname}/{mname} seed {eff_seed}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            failures.append((dsname, bname, mname, eff_seed, f"{type(exc).__name__}: {exc}"))
        finally:
            if S.DEVICE == "cuda":
                torch.cuda.empty_cache()

    print(f"\ndone. {len(plan) - len(failures)}/{len(plan)} cells succeeded")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  {f[0]}/{f[1]}/{f[2]} seed {f[3]}: {f[4]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
