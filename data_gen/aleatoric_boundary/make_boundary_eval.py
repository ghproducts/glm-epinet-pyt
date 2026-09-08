#!/usr/bin/env python3
"""Boundary-sampling redesign of the aleatoric-uncertainty test.

Question: does U_aleatoric actually rise on genuinely hard/ambiguous
examples? `data_gen/label_noise/` already answered a *different* question --
does U_aleatoric rise when uniform label-flip noise is injected into
training? This script instead asks whether a UQ method's own U_aleatoric
tracks a static, non-circular measure of how hard an example *already is* at
test time, with no synthetic corruption involved.

Design
------
1. Run the plain `base` DNABERT2 checkpoint once over the real `promoter_all`
   test split (1584 sequences). For each example compute a softmax-margin
   score

       margin = |p1 - 0.5| * 2    (p1 = softmax probability of class 1)

   which is 0 exactly at the decision boundary and 1 for a maximally
   confident prediction. Split the test set into quartiles by margin: Q1
   (boundary, hardest) through Q4 (canonical, easiest). This score is a
   function of the model's own margin, so a first *independent* check is
   whether accuracy against the ground-truth label rises monotonically
   Q1 -> Q4 (accuracy is not a function of the model's confidence, so this
   is not circular).
2. Run three genuinely stochastic/decomposable UQ methods over the exact
   same 1584 examples, in the exact same order (same tokenizer, same
   `prep_for_trainer` call, same un-shuffled DataLoader, so row i always
   refers to the same sequence across every method and the margin-score
   pass):
     - mc_dropout   (base checkpoint, dropout enabled at eval, K=10)
     - conv_epinet  (label_noise_r00/epinet checkpoint, K=10 index samples)
     - evidential   (label_noise_r00/evidential checkpoint, one forward pass,
                     Dirichlet vacuity as the epistemic-axis analogue)
   and decompose each into U_total/U_epistemic/U_aleatoric via the same
   BALD/entropy identity used everywhere else in this repo
   (`nn_proj.common.utils.compute_uncertainty`).
3. Join every method's per-example uncertainty to the margin quartile and
   report, per method, per quartile: mean U_aleatoric, mean U_epistemic, and
   a Mann-Whitney U test comparing Q1 vs Q4 for each.

Hypothesis under test (report honestly either way -- see README):
  - U_aleatoric should be markedly higher in Q1 (boundary) than Q4
    (canonical): these are the examples that are genuinely hard to call.
  - U_epistemic should NOT track the margin quartile much: promoter_all's
    test split is still in-distribution data the checkpoint was fine-tuned
    on -- just examples the *label* is harder to call, not examples the
    *input* is novel with respect to. A method that conflates the two would
    show epistemic uncertainty rising in Q1 as well; a method that
    correctly separates them should not.

Checkpoint substitution
------------------------
The original manuscript's `trained_models_*/DNABERT2/promoter_all` checkpoint
no longer exists on disk. This script uses
`checkpoints/seed_1/DNABERT2/label_noise_r00/{base,epinet,evidential}`
instead -- DNABERT2 fine-tuned/epinet-trained/evidential-trained on
byte-identical `promoter_all` data (label_noise_r00 = 0% injected noise),
under this repo's current standard pipeline. See README.md for the full
caveat.

Environment
-----------
Must run under the `aleatoric_boundary_venv` virtualenv documented in
README.md (Python 3.10, transformers==4.30.2, triton uninstalled) -- see
that file for why. Run from the repo root:

    /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \\
        data_gen/aleatoric_boundary/make_boundary_eval.py
"""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from scipy.stats import mannwhitneyu, spearmanr

from nn_proj.common.datasets import load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithConvPrior
from nn_proj.models.epinet.feature_fns import NT_feature_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
SEED = 1
K = 10
BATCH_SIZE = 64
MAX_LENGTH = 75  # DNABERT2, promoter_all: 300bp * 0.25 tokens/base


# Absolute, not repo-relative: `checkpoints/` is gitignored, and this script
# runs from inside a git worktree (data_gen/aleatoric_boundary is a worktree
# branch) that does not itself contain the checkpoints/ directory -- only
# the primary checkout does, on the same shared filesystem. See README.md.
REPO_ROOT = "/scratch/home/glh52/glm-epinet-pyt"
CKPT_ROOT = os.path.join(REPO_ROOT, "checkpoints/seed_1/DNABERT2/label_noise_r00")
BASE_CKPT = os.path.join(CKPT_ROOT, "base")
EPINET_CKPT = os.path.join(CKPT_ROOT, "epinet")
EVIDENTIAL_CKPT = os.path.join(CKPT_ROOT, "evidential")

OUT_DIR = "data_gen/aleatoric_boundary"
CSV_DIR = os.path.join(OUT_DIR, "csv_data")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_test_set(tokenizer):
    """promoter_all test split, tokenized once and reused (unshuffled) for
    every model below, so row order is identical across every pass."""
    test_ds = load_NT_tasks(task="promoter_all", split="test")
    labels = list(test_ds["labels"])
    tokenized, collator = prep_for_trainer(test_ds, tokenizer, max_length=MAX_LENGTH, metadata_cols=())
    tokenized = tokenized.remove_columns(["sequence"])
    return labels, tokenized, collator


def make_loader(tokenized, collator):
    return torch.utils.data.DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)


# ---------------------------------------------------------------------------
# Base config/tokenizer loading (shared "3-way version conflict" workaround)
# ---------------------------------------------------------------------------

def load_hub_config():
    """Config is loaded from HUB_ID (not the checkpoint dir) even though the
    checkpoint dir's config.json is content-identical apart from
    torch/transformers bookkeeping fields. Loading from a *local* directory
    with trust_remote_code=True registers DNABERT2's dynamically-loaded
    BertConfig class under a different module name than loading it from the
    hub does (transformers_modules.<checkpoint-dir-basename>.configuration_bert
    vs. transformers_modules.zhihan1996.DNABERT-2-117M...configuration_bert),
    and on transformers>=~4.31 `AutoModel...from_pretrained` enforces a
    strict `model_class.config_class == config.__class__` check when
    registering the dynamic model class -- which then fails with "config
    class you passed is not consistent" purely because of *which path* the
    identical config content was loaded from. Loading config from HUB_ID
    for every checkpoint (base/epinet/evidential all share the same
    architecture) sidesteps this. See README.md's environment notes.
    """
    return transformers.AutoConfig.from_pretrained(HUB_ID, num_labels=2, trust_remote_code=True)


def load_tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=MAX_LENGTH, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    tok.eos_token = tok.pad_token  # InstaDeepAI-specific fixup used throughout this repo
    return tok


# ---------------------------------------------------------------------------
# Step 1: margin score from the plain `base` checkpoint
# ---------------------------------------------------------------------------

def compute_margin_scores(tokenized, collator, labels) -> pd.DataFrame:
    config = load_hub_config()
    model = transformers.AutoModelForSequenceClassification.from_pretrained(HUB_ID, config=config, trust_remote_code=True)
    model.load_state_dict(load_file(os.path.join(BASE_CKPT, "model.safetensors")), strict=True)
    model = model.to(DEVICE).eval()

    loader = make_loader(tokenized, collator)
    rows = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**inputs).logits.float().cpu()
            probs = F.softmax(logits, dim=-1)
            p1 = probs[:, 1]
            margin = (p1 - 0.5).abs() * 2.0
            pred = probs.argmax(dim=-1)
            for i in range(logits.shape[0]):
                rows.append({
                    "idx": idx,
                    "label": int(labels[idx]),
                    "pred_base": int(pred[i]),
                    "prob_class1": float(p1[i]),
                    "margin": float(margin[i]),
                })
                idx += 1

    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    # Rank-based qcut: guarantees four exactly-equal-sized quartiles (396
    # each for N=1584) with deterministic tie-breaking by original row
    # order, rather than pd.qcut on raw margin values, which can raise on
    # duplicate bin edges when many examples share a margin.
    df["margin_rank"] = df["margin"].rank(method="first")
    df["quartile"] = pd.qcut(df["margin_rank"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    df = df.drop(columns=["margin_rank"])
    return df


def sanity_check_margin_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Independent, non-circular check: accuracy against the *true* label,
    grouped by margin quartile. Not a function of the model's own margin
    score, so a monotonic Q1 -> Q4 rise is real evidence the margin score
    tracks genuine example difficulty (not just an artifact of how it was
    constructed)."""
    df = df.copy()
    df["correct"] = (df["pred_base"] == df["label"]).astype(int)
    table = df.groupby("quartile", observed=True).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_margin=("margin", "mean"),
    ).reindex(["Q1", "Q2", "Q3", "Q4"])
    print("\n=== Sanity check: accuracy by margin quartile (independent of any method's own U) ===")
    print(table.to_string())
    accs = table["accuracy"].to_list()
    monotonic = all(a <= b + 1e-9 for a, b in zip(accs, accs[1:]))
    print(f"Monotonic non-decreasing Q1->Q4: {monotonic}")
    return table


# ---------------------------------------------------------------------------
# Step 2a: mc_dropout (base checkpoint, dropout enabled at eval, K stochastic
# forward passes -- see nn_proj.common.utils.enable_mc_dropout)
# ---------------------------------------------------------------------------

def run_mc_dropout(tokenized, collator) -> pd.DataFrame:
    config = load_hub_config()
    model = transformers.AutoModelForSequenceClassification.from_pretrained(HUB_ID, config=config, trust_remote_code=True)
    model.load_state_dict(load_file(os.path.join(BASE_CKPT, "model.safetensors")), strict=True)
    model = model.to(DEVICE)
    enable_mc_dropout(model, p=0.1)

    loader = make_loader(tokenized, collator)
    rows = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            logits_all = torch.stack([model(**inputs).logits for _ in range(K)], dim=0).float().cpu()  # [K,B,C]
            unc = compute_uncertainty(logits_all)
            B = logits_all.shape[1]
            for i in range(B):
                rows.append({
                    "idx": idx,
                    "method": "mc_dropout",
                    "pred": int(unc["predicted_class"][i]),
                    "U_total": float(unc["normalized_total_uncertainty"][i]),
                    "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                    "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                    "vote_pct": float(unc["vote_percentage"][i]),
                })
                idx += 1

    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2b: conv_epinet (FIXED epinet.py -- per-example z, not batch-shared)
# ---------------------------------------------------------------------------

def run_conv_epinet(tokenized, collator) -> pd.DataFrame:
    config = load_hub_config()
    base = transformers.AutoModelForSequenceClassification.from_pretrained(HUB_ID, config=config, trust_remote_code=True)
    epi_cfg = EpinetConfig(num_classes=2, include_inputs=True, vocab_size=config.vocab_size)
    wrapper = EpinetWrapper(base, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=K).to(DEVICE)

    # Trigger lazy ProjectedMLP.core build before loading the state dict.
    dummy_batch = collator([tokenized[i] for i in range(2)])
    dummy_batch = {k: v.to(DEVICE) for k, v in dummy_batch.items() if k != "labels"}
    with torch.no_grad():
        model.wrapper(dummy_batch, n_index_samples=1)
    model.load_state_dict(load_file(os.path.join(EPINET_CKPT, "model.safetensors")), strict=True)
    model.eval()

    loader = make_loader(tokenized, collator)
    rows = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            logits_all = model.wrapper(inputs, n_index_samples=K, return_all=True).float().cpu()  # [K,B,C]
            unc = compute_uncertainty(logits_all)
            B = logits_all.shape[1]
            for i in range(B):
                rows.append({
                    "idx": idx,
                    "method": "conv_epinet",
                    "pred": int(unc["predicted_class"][i]),
                    "U_total": float(unc["normalized_total_uncertainty"][i]),
                    "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                    "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                    "vote_pct": float(unc["vote_percentage"][i]),
                })
                idx += 1

    del model, base
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2c: evidential (Sensoy et al. 2018 Dirichlet evidential head).
#
# nn_proj/models/evidential/ is not present in this worktree (it is an
# untracked file in the main checkout, and this task's scope is restricted
# to data_gen/aleatoric_boundary/ plus the one explicitly-authorized
# epinet.py overwrite -- see README.md). The handful of formulas needed to
# score an already-trained evidential checkpoint are reimplemented here,
# matching nn_proj/models/evidential/evidential.py exactly (read from the
# main checkout for reference): evidence = softplus(logits), alpha =
# evidence + 1, S = sum(alpha), predictive prob = alpha / S, vacuity =
# num_classes / S, and U_aleatoric/U_epistemic/U_total come from
# compute_uncertainty on log(prob) treated as a K=1 stack (so
# U_epistemic == 0 identically by construction for this method -- vacuity
# is the epistemic-axis analogue to read instead, exactly as documented in
# data_gen/label_noise/decomp_compare_label_noise_evidential.py and used in
# its output csv).
# ---------------------------------------------------------------------------

class _EvidentialWrapper(nn.Module):
    def __init__(self, base_model, activation="softplus"):
        super().__init__()
        self.base = base_model
        self.activation = activation

    def forward(self, **inputs):
        out = self.base(**inputs)
        if self.activation == "softplus":
            return F.softplus(out.logits)
        return F.relu(out.logits)


class _HFEvidentialSeqClassifier(nn.Module):
    """Mirrors nn_proj.models.evidential.evidential.HFEvidentialSeqClassifier's
    attribute layout (`self.wrapper.base....`) so the saved checkpoint's
    `wrapper.base.*`-prefixed state dict keys load with strict=True."""

    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, **inputs):
        return self.wrapper(**inputs)


def run_evidential(tokenized, collator, num_classes: int = 2) -> pd.DataFrame:
    config = load_hub_config()
    base = transformers.AutoModelForSequenceClassification.from_pretrained(HUB_ID, config=config, trust_remote_code=True)
    model = _HFEvidentialSeqClassifier(_EvidentialWrapper(base, "softplus")).to(DEVICE)
    model.load_state_dict(load_file(os.path.join(EVIDENTIAL_CKPT, "model.safetensors")), strict=True)
    model.eval()

    loader = make_loader(tokenized, collator)
    rows = []
    idx = 0
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            evidence = model(**inputs).float().cpu()  # [B,C]
            alpha = evidence + 1.0
            S = alpha.sum(dim=-1, keepdim=True)  # [B,1]
            probs = alpha / S
            vacuity = num_classes / S.squeeze(-1)  # [B]

            unc = compute_uncertainty(torch.log(probs.clamp_min(1e-12)).unsqueeze(0))  # [1,B,C]
            B = evidence.shape[0]
            for i in range(B):
                rows.append({
                    "idx": idx,
                    "method": "evidential",
                    "pred": int(unc["predicted_class"][i]),
                    "U_total": float(unc["normalized_total_uncertainty"][i]),
                    "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                    "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                    "vote_pct": float(unc["vote_percentage"][i]),
                    "vacuity": float(vacuity[i]),
                    "dirichlet_strength": float(S[i, 0]),
                })
                idx += 1

    del model, base
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: merge + per-quartile summary + Mann-Whitney U (Q1 vs Q4)
# ---------------------------------------------------------------------------

def summarize(uncertainty_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for method, mdf in uncertainty_df.groupby("method"):
        score_cols = ["U_aleatoric", "U_epistemic"]
        if method == "evidential":
            score_cols = ["U_aleatoric", "vacuity"]

        per_q = mdf.groupby("quartile", observed=True)[score_cols].mean().reindex(["Q1", "Q2", "Q3", "Q4"])
        n_per_q = mdf.groupby("quartile", observed=True).size().reindex(["Q1", "Q2", "Q3", "Q4"])

        for score_col in score_cols:
            q1_vals = mdf.loc[mdf["quartile"] == "Q1", score_col].dropna()
            q4_vals = mdf.loc[mdf["quartile"] == "Q4", score_col].dropna()
            u_stat, p_val = mannwhitneyu(q1_vals, q4_vals, alternative="two-sided")
            direction = "Q1 > Q4" if q1_vals.mean() > q4_vals.mean() else "Q1 < Q4"
            # Supplementary: Spearman rank correlation between the continuous
            # margin score and this uncertainty score across all 1584
            # examples (not just Q1/Q4), for a sense of how strongly the
            # relationship holds across the *whole* range, not just the
            # extremes the MWU test compares.
            rho, rho_p = spearmanr(mdf["margin"], mdf[score_col])
            summary_rows.append({
                "method": method,
                "score": score_col,
                "Q1_mean": per_q.loc["Q1", score_col],
                "Q2_mean": per_q.loc["Q2", score_col],
                "Q3_mean": per_q.loc["Q3", score_col],
                "Q4_mean": per_q.loc["Q4", score_col],
                "Q1_n": int(n_per_q["Q1"]),
                "Q4_n": int(n_per_q["Q4"]),
                "mwu_U": u_stat,
                "mwu_p": p_val,
                "direction": direction,
                "spearman_rho_vs_margin": rho,
                "spearman_p": rho_p,
            })
    return pd.DataFrame(summary_rows)


def write_results_summary_md(acc_table: pd.DataFrame, summary_df: pd.DataFrame, path: str) -> None:
    fmt_summary = summary_df.copy()
    fmt_summary["mwu_p"] = fmt_summary["mwu_p"].map(lambda p: f"{p:.3e}")
    fmt_summary["spearman_p"] = fmt_summary["spearman_p"].map(lambda p: f"{p:.3e}")
    for c in ["Q1_mean", "Q2_mean", "Q3_mean", "Q4_mean", "spearman_rho_vs_margin"]:
        fmt_summary[c] = fmt_summary[c].map(lambda v: f"{v:.4f}")
    fmt_summary["mwu_U"] = fmt_summary["mwu_U"].map(lambda v: f"{v:.0f}")

    lines = []
    lines.append("# Boundary-sampling aleatoric/epistemic results\n")
    lines.append("## Sanity check: accuracy by margin quartile (independent of any method's own U)\n")
    lines.append(acc_table.reset_index().to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n")
    lines.append("## Per-method, per-quartile mean U_aleatoric / U_epistemic(-analogue), Q1 vs Q4 Mann-Whitney U\n")
    lines.append("`spearman_rho_vs_margin` is the rank correlation between the continuous margin score and the "
                  "uncertainty score across all 1584 examples (negative = score falls as margin rises, i.e. rises "
                  "toward the boundary), supplementary to the Q1-vs-Q4 comparison.\n")
    lines.append(fmt_summary.to_markdown(index=False))
    lines.append("\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {path}")


def main():
    set_seed(SEED)
    os.makedirs(CSV_DIR, exist_ok=True)

    tokenizer = load_tokenizer()
    labels, tokenized, collator = load_test_set(tokenizer)
    print(f"promoter_all test set size: {len(labels)}")

    # --- Step 1: margin score + quartiles from the base checkpoint ---
    print("\n[1/4] Computing margin scores from the base checkpoint...")
    margin_df = compute_margin_scores(tokenized, collator, labels)
    margin_df.to_csv(os.path.join(CSV_DIR, "margin_scores.csv"), index=False)
    print(f"wrote {os.path.join(CSV_DIR, 'margin_scores.csv')}")
    acc_table = sanity_check_margin_quartiles(margin_df)

    quartile_map = margin_df.set_index("idx")["quartile"]
    margin_map = margin_df.set_index("idx")["margin"]

    # --- Step 2: stochastic-K inference, all sharing the margin partition ---
    set_seed(SEED)
    print("\n[2/4] Running mc_dropout (K=10)...")
    mc_df = run_mc_dropout(tokenized, collator)

    set_seed(SEED)
    print("\n[3/4] Running conv_epinet (K=10, fixed per-example z)...")
    epinet_df = run_conv_epinet(tokenized, collator)

    set_seed(SEED)
    print("\n[4/4] Running evidential...")
    evid_df = run_evidential(tokenized, collator)

    uncertainty_df = pd.concat([mc_df, epinet_df, evid_df], ignore_index=True)
    uncertainty_df["label"] = uncertainty_df["idx"].map(lambda i: labels[i])
    uncertainty_df["quartile"] = uncertainty_df["idx"].map(quartile_map)
    uncertainty_df["margin"] = uncertainty_df["idx"].map(margin_map)
    uncertainty_df.to_csv(os.path.join(CSV_DIR, "uncertainty_by_method.csv"), index=False)
    print(f"\nwrote {os.path.join(CSV_DIR, 'uncertainty_by_method.csv')}")

    # --- Step 3: merge + summarize ---
    summary_df = summarize(uncertainty_df)
    summary_df.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)
    print(f"wrote {os.path.join(OUT_DIR, 'results_summary.csv')}")
    write_results_summary_md(acc_table, summary_df, os.path.join(OUT_DIR, "results_summary.md"))

    print("\n=== Final summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
