"""
Adds `base` and `base_scaled` (temperature-scaled) to the DNABERT2-only
ECE/NLL/Brier comparison, on the same two new datasets (aleatoric_boundary,
promoter_alisim dense grid). These were missing from the first pass because
that pass only pulled from the 9-method UQ eval CSVs, which never included
a plain no-decomposition base-method run.

- Temperature T is freshly fit on this checkpoint (checkpoints/seed_1/
  DNABERT2/label_noise_r00/base), NOT reused from the stale
  temp_scaling_factors_1.tsv value (0.9238) -- that value was fit on the
  original manuscript's trained_models_1/DNABERT2/promoter_all checkpoint,
  which no longer exists on disk and was trained with different
  hyperparameters (see prior discussion). Fit via the exact
  nn_proj/models/DNABERT2/scaling.py:fit_temperature() procedure (LBFGS on
  log(T), NLL loss), on a held-out 10% stratified split of promoter_all's
  training data (data_seed=42 -- not necessarily the original training
  run's exact data_seed, a reasonable choice for this supplementary fit).

- aleatoric_boundary: no rerun needed. margin_scores.csv already has the
  base checkpoint's raw p1 per test example. For binary softmax only the
  logit *difference* determines the output, so the effective logit gap is
  exactly recoverable as z = log(p1/(1-p1)); temperature scaling divides
  both logits by T, so p1_scaled = sigmoid(z/T) is an exact reconstruction,
  not an approximation.

- promoter_alisim_dense: genuinely new inference needed (base was never
  run on this data at all) -- one single-pass (no K-sampling) forward pass
  over the 6400 simulated sequences.
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


def load_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=MODEL_MAX_LENGTH, padding_side="right",
        use_fast=True, trust_remote_code=True,
    )
    tokenizer.eos_token = tokenizer.pad_token
    config = transformers.AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )
    sd = load_file(os.path.join(CKPT, "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    return tokenizer, model.to(DEVICE).eval()


def fit_temperature(logits, labels, max_iter=50):
    # Verbatim from nn_proj/models/DNABERT2/scaling.py
    import torch.nn as nn
    logits = logits.float()
    labels = labels.long()
    log_T = torch.zeros((), requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = loss_fn(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).detach().cpu().item())


@torch.no_grad()
def get_logits(tokenizer, model, sequences, batch_size=64):
    out = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits.float().cpu()
        out.append(logits)
    return torch.cat(out, dim=0)


def nll_brier(p1, label):
    eps = 1e-12
    p1c = np.clip(p1, eps, 1 - eps)
    p_true = np.where(label == 1, p1c, 1 - p1c)
    nll = -np.log(p_true).mean()
    brier = ((p1 - label) ** 2).mean()
    return nll, brier


def ece_binned(conf, correct, n_bins, adaptive=False):
    n = len(conf)
    if adaptive:
        order = np.argsort(conf)
        edges_idx = np.linspace(0, n, n_bins + 1).astype(int)
        bins = [order[edges_idx[i]:edges_idx[i + 1]] for i in range(n_bins)]
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(conf, edges[1:-1], right=True)
        bins = [np.where(bin_ids == b)[0] for b in range(n_bins)]
    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        ece += (len(idx) / n) * abs(conf[idx].mean() - correct[idx].mean())
    return ece


def summarize_from_p1(p1, pred, label, dataset, method):
    correct = (pred == label).astype(int)
    conf = np.where(pred == 1, p1, 1 - p1)
    nll, brier = nll_brier(p1, label)
    row = {"dataset": dataset, "method": method, "n": len(p1), "accuracy": correct.mean(),
           "nll": nll, "brier": brier}
    for b in (10, 15, 20, 25):
        row[f"ece_{b}bin"] = ece_binned(conf, correct, b)
    row["ace_15bin_equalmass"] = ece_binned(conf, correct, 15, adaptive=True)
    return row


def main():
    from datasets import load_dataset, ClassLabel

    tokenizer, model = load_model()

    print("[fit T] loading promoter_all train split for held-out val slice")
    ds_all = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", split="train")
    ds = ds_all.filter(lambda ex: ex["task"] == "promoter_all")
    if "labels" not in ds.column_names and "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")
    if not isinstance(ds.features["labels"], ClassLabel):
        ds = ds.class_encode_column("labels")
    split = ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="labels")
    val = split["test"]
    val_logits = get_logits(tokenizer, model, list(val["sequence"]))
    val_labels = torch.tensor(val["labels"])
    T = fit_temperature(val_logits, val_labels)
    print(f"[fit T] T = {T:.6f} (n_val={len(val)})")

    rows = []

    # --- aleatoric_boundary: reconstruct from saved p1, no rerun ---
    mscore = pd.read_csv(
        "/scratch/home/glh52/glm-epinet-pyt/.claude/worktrees/fix-epinet-batch-z/"
        "data_gen/aleatoric_boundary/csv_data/margin_scores.csv"
    )
    p1 = mscore["prob_class1"].values.astype(float)
    pred = mscore["pred_base"].values.astype(int)
    label = mscore["label"].values.astype(int)
    rows.append(summarize_from_p1(p1, pred, label, "aleatoric_boundary", "base"))

    z = np.log(np.clip(p1, 1e-12, 1 - 1e-12) / np.clip(1 - p1, 1e-12, 1 - 1e-12))
    p1_scaled = 1 / (1 + np.exp(-z / T))
    pred_scaled = (p1_scaled >= 0.5).astype(int)  # identical to pred (T>0 monotonic), sanity
    assert np.array_equal(pred_scaled, pred), "temperature scaling changed argmax -- should never happen for T>0"
    rows.append(summarize_from_p1(p1_scaled, pred, label, "aleatoric_boundary", "base_scaled"))

    # --- promoter_alisim_dense: genuinely new inference ---
    print("[alisim] running base inference on promoter_alisim_dense.csv")
    alisim = pd.read_csv(
        "/scratch/home/glh52/glm-epinet-pyt/.claude/worktrees/fix-epinet-batch-z/"
        "data_gen/promoter_alisim/csv_data/promoter_alisim_dense.csv"
    )
    logits = get_logits(tokenizer, model, list(alisim["sequence"]))
    probs = F.softmax(logits, dim=-1).numpy()
    p1_a = probs[:, 1]
    pred_a = (p1_a >= 0.5).astype(int)
    label_a = alisim["original_label"].values.astype(int)
    rows.append(summarize_from_p1(p1_a, pred_a, label_a, "promoter_alisim_dense", "base"))

    p1_a_scaled = F.softmax(logits / T, dim=-1).numpy()[:, 1]
    pred_a_scaled = (p1_a_scaled >= 0.5).astype(int)
    assert np.array_equal(pred_a_scaled, pred_a), "temperature scaling changed argmax -- should never happen for T>0"
    rows.append(summarize_from_p1(p1_a_scaled, pred_a, label_a, "promoter_alisim_dense", "base_scaled"))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + out.round(4).to_string(index=False))
    out.to_csv("/home/glh52/.claude/jobs/80f6ac1e/tmp/dnabert_base_basescaled.csv", index=False)
    with open("/home/glh52/.claude/jobs/80f6ac1e/tmp/fitted_temperature.txt", "w") as f:
        f.write(f"T={T}\nn_val={len(val)}\ndata_seed=42\ncheckpoint={CKPT}\n")


if __name__ == "__main__":
    main()
