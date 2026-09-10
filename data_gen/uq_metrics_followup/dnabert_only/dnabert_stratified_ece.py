"""
Pooled vs. stratified ECE/NLL/Brier, DNABERT2 methods only, both new
datasets. Stratifies by margin quartile (aleatoric_boundary) and by
branch_length (promoter_alisim_dense) instead of pooling across the whole
axis of variation -- see chat: pooling risks exactly the R2-3 masking
pathology (a wide accuracy range can average out in a way that misrepresents
calibration at either end).
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file

CKPT = "/scratch/home/glh52/glm-epinet-pyt/checkpoints/seed_1/DNABERT2/label_noise_r00/base"
HUB_ID = "zhihan1996/DNABERT-2-117M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_MAX_LENGTH = 75
T_FITTED = 1.1496275663375854  # from prior fit, LBFGS/NLL, held-out 10% of promoter_all train, seed 42

REPO = "/scratch/home/glh52/glm-epinet-pyt/.claude/worktrees/fix-epinet-batch-z"
DNABERT_UQ_METHODS = ["mc_dropout", "conv_epinet", "laplace", "ensemble_k5", "ensemble_k3"]


def invert_binary_entropy(u_total):
    u_total = np.clip(u_total, 1e-12, 1.0)
    lo = np.full_like(u_total, 0.5)
    hi = np.full_like(u_total, 1.0 - 1e-12)
    for _ in range(60):
        mid = (lo + hi) / 2
        h = -(mid * np.log(mid) + (1 - mid) * np.log(1 - mid)) / np.log(2)
        lo = np.where(h > u_total, mid, lo)
        hi = np.where(h > u_total, hi, mid)
    return (lo + hi) / 2


def ece_binned(conf, correct, n_bins=15):
    n = len(conf)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, edges[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        if len(idx) == 0:
            continue
        ece += (len(idx) / n) * abs(conf[idx].mean() - correct[idx].mean())
    return ece


def nll_brier(p1, label):
    eps = 1e-12
    p1c = np.clip(p1, eps, 1 - eps)
    p_true = np.where(label == 1, p1c, 1 - p1c)
    return -np.log(p_true).mean(), ((p1 - label) ** 2).mean()


def metrics_row(p1, pred, label, **extra):
    correct = (pred == label).astype(int)
    conf = np.where(pred == 1, p1, 1 - p1)
    nll, brier = nll_brier(p1, label)
    row = {"n": len(p1), "accuracy": correct.mean(), "nll": nll, "brier": brier,
           "ece_15bin": ece_binned(conf, correct, 15)}
    row.update(extra)
    return row


@torch.no_grad()
def get_logits(tokenizer, model, sequences, batch_size=64):
    out = []
    for i in range(0, len(sequences), batch_size):
        enc = tokenizer(sequences[i:i + batch_size], truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        out.append(model(**enc).logits.float().cpu())
    return torch.cat(out, dim=0)


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=MODEL_MAX_LENGTH, padding_side="right", use_fast=True, trust_remote_code=True)
    tokenizer.eos_token = tokenizer.pad_token
    config = transformers.AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(HUB_ID, config=config, trust_remote_code=True)
    model.load_state_dict(load_file(os.path.join(CKPT, "model.safetensors")), strict=True)
    model = model.to(DEVICE).eval()

    pooled_rows, strat_rows = [], []

    # ---------------- aleatoric_boundary ----------------
    mscore = pd.read_csv(f"{REPO}/data_gen/aleatoric_boundary/csv_data/margin_scores.csv")
    p1 = mscore["prob_class1"].values.astype(float)
    pred = mscore["pred_base"].values.astype(int)
    label = mscore["label"].values.astype(int)
    quartile = mscore["quartile"].values
    z = np.log(np.clip(p1, 1e-12, 1 - 1e-12) / np.clip(1 - p1, 1e-12, 1 - 1e-12))
    p1_scaled = 1 / (1 + np.exp(-z / T_FITTED))

    for name, probs in [("base", p1), ("base_scaled", p1_scaled)]:
        pooled_rows.append(metrics_row(probs, pred, label, dataset="aleatoric_boundary", method=name, stratum="pooled"))
        for q in sorted(set(quartile)):
            m = quartile == q
            strat_rows.append(metrics_row(probs[m], pred[m], label[m], dataset="aleatoric_boundary", method=name, stratum=q))

    uqb = pd.read_csv(f"{REPO}/data_gen/aleatoric_boundary/csv_data/uncertainty_by_method.csv")
    for m in DNABERT_UQ_METHODS:
        sub = uqb[uqb["method"] == m]
        conf = invert_binary_entropy(sub["U_total"].values.astype(float))
        pr = sub["pred"].values.astype(int)
        lb = sub["label"].values.astype(int)
        qt = sub["quartile"].values
        p1_m = np.where(pr == 1, conf, 1 - conf)
        pooled_rows.append(metrics_row(p1_m, pr, lb, dataset="aleatoric_boundary", method=m, stratum="pooled"))
        for q in sorted(set(qt)):
            mm = qt == q
            strat_rows.append(metrics_row(p1_m[mm], pr[mm], lb[mm], dataset="aleatoric_boundary", method=m, stratum=q))

    # ---------------- promoter_alisim_dense ----------------
    alisim = pd.read_csv(f"{REPO}/data_gen/promoter_alisim/csv_data/promoter_alisim_dense.csv")
    logits = get_logits(tokenizer, model, list(alisim["sequence"]))
    probs_a = F.softmax(logits, dim=-1).numpy()[:, 1]
    probs_a_scaled = F.softmax(logits / T_FITTED, dim=-1).numpy()[:, 1]
    pred_a = (probs_a >= 0.5).astype(int)
    label_a = alisim["original_label"].values.astype(int)
    bl = alisim["branch_length"].values

    for name, probs in [("base", probs_a), ("base_scaled", probs_a_scaled)]:
        pooled_rows.append(metrics_row(probs, pred_a, label_a, dataset="promoter_alisim_dense", method=name, stratum="pooled"))
        for t in sorted(set(bl)):
            m = bl == t
            strat_rows.append(metrics_row(probs[m], pred_a[m], label_a[m], dataset="promoter_alisim_dense", method=name, stratum=t))

    uqa = pd.read_csv(f"{REPO}/data_gen/promoter_alisim/uncertainty_eval/dense_grid/per_example_uncertainty.csv")
    for m in DNABERT_UQ_METHODS:
        sub = uqa[uqa["method"] == m]
        u_total = (sub["U_aleatoric"] + sub["U_epistemic"]).values.astype(float)
        conf = invert_binary_entropy(u_total)
        pr = sub["pred"].values.astype(int)
        lb = sub["labels"].values.astype(int)
        blm = sub["branch_length"].values
        p1_m = np.where(pr == 1, conf, 1 - conf)
        pooled_rows.append(metrics_row(p1_m, pr, lb, dataset="promoter_alisim_dense", method=m, stratum="pooled"))
        for t in sorted(set(blm)):
            mm = blm == t
            strat_rows.append(metrics_row(p1_m[mm], pr[mm], lb[mm], dataset="promoter_alisim_dense", method=m, stratum=t))

    pooled = pd.DataFrame(pooled_rows)
    strat = pd.DataFrame(strat_rows)
    order = ["base", "base_scaled", "mc_dropout", "conv_epinet", "laplace", "ensemble_k5", "ensemble_k3"]
    pooled["method"] = pd.Categorical(pooled["method"], categories=order, ordered=True)
    strat["method"] = pd.Categorical(strat["method"], categories=order, ordered=True)
    pooled = pooled.sort_values(["dataset", "method"])
    strat = strat.sort_values(["dataset", "method", "stratum"])

    pooled.to_csv("/home/glh52/.claude/jobs/80f6ac1e/tmp/pooled_ece.csv", index=False)
    strat.to_csv("/home/glh52/.claude/jobs/80f6ac1e/tmp/stratified_ece.csv", index=False)
    pd.set_option("display.width", 200)
    print("=== POOLED ===")
    print(pooled.round(4).to_string(index=False))
    print("\n=== STRATIFIED ===")
    print(strat.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
