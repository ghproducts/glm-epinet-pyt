#!/usr/bin/env python3
"""Extends the boundary-sampling aleatoric-uncertainty test
(`make_boundary_eval.py`) from 3 methods (conv_epinet, mc_dropout,
evidential) to the full method roster used elsewhere in this project's
UQ comparisons: `laplace`, DNABERT2 deep ensemble (K=5, K=3), a
from-scratch CNN (mc_dropout and ensemble sampling variants), and a
k-mer + Random Forest baseline -- 6 new method-configurations, run on the
exact same margin-quartile partition of the `promoter_all` test set
(`csv_data/margin_scores.csv`, produced by `make_boundary_eval.py` and
reused verbatim here, not recomputed).

Templates adapted from `data_gen/label_noise/` (built for the label-noise
axis; the label_noise-r00 train/test split there is byte-identical to this
project's standard clean `promoter_all` split -- see README.md's "Row-order
verification" note): `decomp_compare_label_noise_laplace.py`,
`decomp_compare_label_noise_ensemble.py`, `decomp_compare_label_noise_cnn.py`
(+ `cnn_scratch.py`), `decomp_compare_label_noise_rf.py`.

Row-order note
--------------
`data_gen/label_noise/csv_data_r00/test.csv` was verified (see README.md) to
be in *exactly* the same row order as
`load_NT_tasks(task="promoter_all", split="test")` (the source
`make_boundary_eval.py` used for `margin_scores.csv`'s `idx`), so `idx == i`
for the CNN/RF methods that read that csv directly with `pandas`, and `idx`
increments in DataLoader order (shuffle=False) for the two DNABERT2-based
methods (laplace, ensemble) that go through the same
`load_NT_tasks`/`prep_for_trainer` path `make_boundary_eval.py` used. All six
methods therefore share the exact same per-example `idx` <-> sequence
mapping as `margin_scores.csv` and the original 3-method
`uncertainty_by_method.csv`.

Both the BALD/entropy decomposition (`nn_proj.common.utils.compute_uncertainty`,
matching the original 3 methods' `U_epistemic`/`U_aleatoric` columns exactly)
and the variance-based decomposition
(`nn_proj.common.variance_decomp.compute_uncertainty_variance`, this
project's other standard convention, used throughout `data_gen/label_noise/`)
are computed for every new method and stored as extra `var_U_epistemic`/
`var_U_aleatoric` columns (NaN for the original 3 methods' rows, which never
had this decomposition computed for them).

Environment: same `aleatoric_boundary_venv` as `make_boundary_eval.py`
(Python 3.10, transformers==4.30.2, triton uninstalled) -- see README.md.
Run from the repo root:

    /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \\
        data_gen/aleatoric_boundary/make_boundary_eval_extra_methods.py
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from safetensors.torch import load_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.common.variance_decomp import compute_uncertainty_variance
from nn_proj.models.laplace import LaplaceConfig, LaplaceSeqClassifier, fit_diagonal_laplace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = "/scratch/home/glh52/glm-epinet-pyt"  # see make_boundary_eval.py: worktree has no checkpoints/ dir
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "data_gen", "label_noise"))

from cnn_scratch import batched_logits, one_hot_encode, train_cnn  # noqa: E402
from make_boundary_eval import sanity_check_margin_quartiles, summarize, write_results_summary_md  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
MAX_LENGTH = 75
SEED = 1
BATCH_SIZE = 64

CKPT_ROOT_SEED1 = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00")
BASE_CKPT_SEED1 = os.path.join(CKPT_ROOT_SEED1, "base")
ENSEMBLE_SEEDS_K5 = [1, 2, 3, 4, 42]  # this project's standard 5-seed convention
ENSEMBLE_SEEDS_K3 = [1, 2, 3]         # first 3 of the standard set

# The label_noise_r00 csv snapshot of the clean promoter_all train/test
# split -- row-order-verified identical to load_NT_tasks(..., split="test")
# (see README.md), so it is reused directly for CNN/RF training+eval instead
# of regenerating an identical csv from scratch.
LABEL_NOISE_TRAIN_CSV = os.path.join(REPO_ROOT, "data_gen/label_noise/csv_data_r00/train.csv")
LABEL_NOISE_TEST_CSV = os.path.join(REPO_ROOT, "data_gen/label_noise/csv_data_r00/test.csv")

OUT_DIR = os.path.join(SCRIPT_DIR)
CSV_DIR = os.path.join(OUT_DIR, "csv_data")
MARGIN_CSV = os.path.join(CSV_DIR, "margin_scores.csv")
UNCERTAINTY_CSV = os.path.join(CSV_DIR, "uncertainty_by_method.csv")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Shared DNABERT2 loading (same 3-way version-conflict workaround as
# make_boundary_eval.py -- see its docstring / README.md for why).
# ---------------------------------------------------------------------------

def load_hub_config():
    return transformers.AutoConfig.from_pretrained(HUB_ID, num_labels=2, trust_remote_code=True)


def load_tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=MAX_LENGTH, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    tok.eos_token = tok.pad_token
    return tok


def load_test_set(tokenizer):
    test_ds = load_NT_tasks(task="promoter_all", split="test")
    labels = list(test_ds["labels"])
    tokenized, collator = prep_for_trainer(test_ds, tokenizer, max_length=MAX_LENGTH, metadata_cols=())
    tokenized = tokenized.remove_columns(["sequence"])
    return labels, tokenized, collator


def make_loader(tokenized, collator, batch_size=BATCH_SIZE):
    return torch.utils.data.DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator)


def load_dnabert_base(ckpt_dir, config):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(HUB_ID, config=config, trust_remote_code=True)
    model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
    return model.to(DEVICE).eval()


def rows_from_logits_stack(logits_all: torch.Tensor, method_name: str) -> list[dict]:
    """logits_all: [K,B,C] cpu tensor -> list of per-example dict rows with
    both BALD (U_*) and variance (var_U_*) decompositions. `idx` is NOT set
    here; callers assign it (batch-loop callers keep a running counter,
    whole-tensor callers use range(B))."""
    unc = compute_uncertainty(logits_all)
    unc_var = compute_uncertainty_variance(logits_all)
    B = logits_all.shape[1]
    rows = []
    for i in range(B):
        rows.append({
            "method": method_name,
            "pred": int(unc["predicted_class"][i]),
            "U_total": float(unc["normalized_total_uncertainty"][i]),
            "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
            "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
            "vote_pct": float(unc["vote_percentage"][i]),
            "var_U_epistemic": float(unc_var["normalized_epistemic_uncertainty"][i]),
            "var_U_aleatoric": float(unc_var["normalized_aleatoric_uncertainty"][i]),
        })
    return rows


# ---------------------------------------------------------------------------
# 1. laplace -- last-layer diagonal Laplace, post-hoc on the base checkpoint,
#    fit on the checkpoint's own (clean, r00) training labels.
# ---------------------------------------------------------------------------

def run_laplace(tokenizer, tokenized, collator) -> pd.DataFrame:
    print("\n[1/6] Running laplace (K=16 posterior samples, post-hoc on seed_1/base)...")
    config = load_hub_config()
    model = load_dnabert_base(BASE_CKPT_SEED1, config)

    fit_ds_raw = load_local_dataset(LABEL_NOISE_TRAIN_CSV)
    fit_tok, fit_collator = prep_for_trainer(fit_ds_raw, tokenizer, max_length=MAX_LENGTH, metadata_cols=())
    fit_tok = fit_tok.remove_columns(["sequence"])

    lap_cfg = LaplaceConfig(classifier_attr="classifier", prior_precision=1.0, max_examples=2000)
    weight_var, bias_var, n_used = fit_diagonal_laplace(model, fit_tok, fit_collator, lap_cfg, batch_size=64)
    lap_model = LaplaceSeqClassifier(model, weight_var.to(DEVICE), bias_var.to(DEVICE),
                                      classifier_attr="classifier").to(DEVICE)
    lap_model.eval()
    print(f"[laplace] GGN fit on {n_used} training examples (prior_precision=1.0)")

    loader = make_loader(tokenized, collator)
    rows = []
    idx = 0
    K = 16
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            lap_model(**inputs, labels=None)  # populate cached pre-classifier features
            logits_all = lap_model.sample_logits(n_samples=K).float().cpu()  # [K,B,C]
            for r in rows_from_logits_stack(logits_all, "laplace"):
                r["idx"] = idx
                rows.append(r)
                idx += 1

    del model, lap_model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2/3. DNABERT2 deep ensemble, K=5 and K=3 -- softmax-averaging independently
#    fine-tuned seeded `base` checkpoints, no new training.
# ---------------------------------------------------------------------------

def run_ensemble(tokenized, collator, seeds: list[int], method_name: str) -> pd.DataFrame:
    print(f"\nRunning {method_name} (K={len(seeds)}, seeds={seeds})...")
    config = load_hub_config()
    members = [load_dnabert_base(os.path.join(REPO_ROOT, f"checkpoints/seed_{s}/DNABERT2/label_noise_r00/base"), config)
               for s in seeds]

    loader = make_loader(tokenized, collator)
    rows = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            logits_all = torch.stack([m(**inputs).logits for m in members], dim=0).float().cpu()  # [K,B,C]
            for r in rows_from_logits_stack(logits_all, method_name):
                r["idx"] = idx
                rows.append(r)
                idx += 1

    for m in members:
        del m
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4/5. From-scratch CNN (DeepBind/DeepSEA-lineage), mc_dropout and ensemble
#    sampling variants. Trains fresh (no saved checkpoint format defined by
#    cnn_scratch.py) on the standard clean promoter_all/label_noise_r00
#    training data, evaluated on the row-order-verified-identical test split.
# ---------------------------------------------------------------------------

def _load_cnn_data():
    train_df = pd.read_csv(LABEL_NOISE_TRAIN_CSV)
    test_df = pd.read_csv(LABEL_NOISE_TEST_CSV)
    X_train = one_hot_encode(train_df["sequence"].tolist())
    y_train = torch.tensor(train_df["label"].values, dtype=torch.long)
    X_test = one_hot_encode(test_df["sequence"].tolist())
    return X_train, y_train, X_test


def run_cnn_mc_dropout(X_train, y_train, X_test) -> pd.DataFrame:
    print("\nRunning cnn_mc_dropout (fresh SmallCNN, seed=1, dropout p=0.3, K=16)...")
    model = train_cnn(X_train, y_train, seed=1, device=DEVICE)
    enable_mc_dropout(model, p=0.3)
    with torch.no_grad():
        logits_all = torch.stack([batched_logits(model, X_test, DEVICE) for _ in range(16)], dim=0)
    rows = rows_from_logits_stack(logits_all, "cnn_mc_dropout")
    for i, r in enumerate(rows):
        r["idx"] = i
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def run_cnn_ensemble(X_train, y_train, X_test) -> pd.DataFrame:
    seeds = [1, 2, 3, 4, 42]
    print(f"\nRunning cnn_ensemble (K={len(seeds)} independently-initialized-and-trained SmallCNNs, seeds={seeds})...")
    members = [train_cnn(X_train, y_train, seed=s, device=DEVICE) for s in seeds]
    with torch.no_grad():
        logits_all = torch.stack([batched_logits(m, X_test, DEVICE) for m in members], dim=0)
    rows = rows_from_logits_stack(logits_all, "cnn_ensemble")
    for i, r in enumerate(rows):
        r["idx"] = i
    for m in members:
        del m
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. k-mer + Random Forest -- classical, non-neural baseline. Per-tree
#    predict_proba stack is the K "samples" (same convention as
#    decomp_compare_label_noise_rf.py).
# ---------------------------------------------------------------------------

def run_rf() -> pd.DataFrame:
    print("\nRunning rf (6-mer counts + RandomForest, 200 trees)...")
    train_df = pd.read_csv(LABEL_NOISE_TRAIN_CSV)
    test_df = pd.read_csv(LABEL_NOISE_TEST_CSV)

    vec = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
    X_train = vec.fit_transform(train_df["sequence"])
    X_test = vec.transform(test_df["sequence"])

    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1, min_samples_leaf=20)
    rf.fit(X_train, train_df["label"].values)

    probs = np.stack([tree.predict_proba(X_test) for tree in rf.estimators_], axis=0)  # [K,B,C]
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    logits_all = torch.from_numpy(np.log(probs)).float()

    rows = rows_from_logits_stack(logits_all, "rf")
    for i, r in enumerate(rows):
        r["idx"] = i
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CSV_DIR, exist_ok=True)

    margin_df = pd.read_csv(MARGIN_CSV)
    quartile_map = margin_df.set_index("idx")["quartile"]
    margin_map = margin_df.set_index("idx")["margin"]
    label_map = margin_df.set_index("idx")["label"]

    set_seed(SEED)
    tokenizer = load_tokenizer()
    labels, tokenized, collator = load_test_set(tokenizer)
    print(f"promoter_all test set size: {len(labels)}")
    assert labels == list(label_map.reindex(range(len(labels)))), "label order mismatch vs margin_scores.csv idx"

    new_dfs = []

    set_seed(SEED)
    new_dfs.append(run_laplace(tokenizer, tokenized, collator))

    set_seed(SEED)
    new_dfs.append(run_ensemble(tokenized, collator, ENSEMBLE_SEEDS_K5, "ensemble_k5"))

    set_seed(SEED)
    new_dfs.append(run_ensemble(tokenized, collator, ENSEMBLE_SEEDS_K3, "ensemble_k3"))

    set_seed(SEED)
    X_train, y_train, X_test = _load_cnn_data()
    new_dfs.append(run_cnn_mc_dropout(X_train, y_train, X_test))

    set_seed(SEED)
    new_dfs.append(run_cnn_ensemble(X_train, y_train, X_test))

    set_seed(SEED)
    new_dfs.append(run_rf())

    new_uncertainty_df = pd.concat(new_dfs, ignore_index=True)
    new_uncertainty_df["label"] = new_uncertainty_df["idx"].map(label_map)
    new_uncertainty_df["quartile"] = new_uncertainty_df["idx"].map(quartile_map)
    new_uncertainty_df["margin"] = new_uncertainty_df["idx"].map(margin_map)

    # --- Merge with the existing 3-method csv (conv_epinet/mc_dropout/evidential) ---
    existing_df = pd.read_csv(UNCERTAINTY_CSV)
    combined_df = pd.concat([existing_df, new_uncertainty_df], ignore_index=True)
    combined_df.to_csv(UNCERTAINTY_CSV, index=False)
    print(f"\nwrote {UNCERTAINTY_CSV} ({len(combined_df)} rows, {combined_df['method'].nunique()} methods: "
          f"{sorted(combined_df['method'].unique())})")

    # --- Combined summary across all methods (reuses make_boundary_eval.py's summarize()) ---
    acc_table = sanity_check_margin_quartiles(margin_df)
    summary_df = summarize(combined_df)
    summary_df.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)
    write_results_summary_md(acc_table, summary_df, os.path.join(OUT_DIR, "results_summary.md"))

    # --- Supplementary: variance-decomposition summary, new methods only ---
    var_rows = []
    for method, mdf in new_uncertainty_df.groupby("method"):
        for score_col in ["var_U_aleatoric", "var_U_epistemic"]:
            per_q = mdf.groupby("quartile", observed=True)[score_col].mean().reindex(["Q1", "Q2", "Q3", "Q4"])
            from scipy.stats import mannwhitneyu
            q1 = mdf.loc[mdf["quartile"] == "Q1", score_col]
            q4 = mdf.loc[mdf["quartile"] == "Q4", score_col]
            u_stat, p_val = mannwhitneyu(q1, q4, alternative="two-sided")
            var_rows.append({
                "method": method, "score": score_col,
                "Q1_mean": per_q["Q1"], "Q2_mean": per_q["Q2"], "Q3_mean": per_q["Q3"], "Q4_mean": per_q["Q4"],
                "mwu_U": u_stat, "mwu_p": p_val,
            })
    var_summary_df = pd.DataFrame(var_rows)
    var_summary_df.to_csv(os.path.join(OUT_DIR, "results_summary_variance_supplement.csv"), index=False)
    print(f"wrote {os.path.join(OUT_DIR, 'results_summary_variance_supplement.csv')}")

    print("\n=== Final combined summary (BALD decomposition, all methods) ===")
    print(summary_df.to_string(index=False))
    print("\n=== Variance-decomposition supplement (new methods only) ===")
    print(var_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
