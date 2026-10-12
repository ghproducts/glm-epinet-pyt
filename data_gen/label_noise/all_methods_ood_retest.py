#!/usr/bin/env python3
"""All 7 UQ methods, one DNABERT2/promoter_all checkpoint family (seed 1),
evaluated on the same three input severities used in the evidential retest:
real, character-shuffled, and uniform-random synthetic DNA.

Reuses: checkpoints/seed_1/DNABERT2/label_noise_r00/{base,epinet} (clean,
r=0 promoter_all -- this project's current pipeline, not the older
manuscript-era checkpoint), temp_scaling_factors_1.tsv's DNABERT2/promoter_all
row, and the evidential checkpoint + retest CSV already produced this
session (checkpoints/evidential_test/DNABERT2/checkpoint-5064,
data_gen/label_noise/evidential_ood_retest.csv) rather than recomputing.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from scipy.stats import mannwhitneyu

from nn_proj.common.datasets import load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.common.conformal import (
    score_nonconformity, fit_conformal_threshold, predict_sets,
)
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithConvPrior
from nn_proj.models.epinet.feature_fns import NT_feature_fn
from nn_proj.models.laplace.laplace_head import (
    LaplaceConfig, fit_diagonal_laplace, LaplaceSeqClassifier, predict_laplace,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUB_ID = "zhihan1996/DNABERT-2-117M"
BASE_CKPT = "checkpoints/seed_1/DNABERT2/label_noise_r00/base"
EPI_CKPT = "checkpoints/seed_1/DNABERT2/label_noise_r00/epinet"
TEMPERATURE = 0.923768  # temp_scaling_factors_1.tsv, DNABERT2 promoter_all
BASES = ["A", "C", "G", "T"]


def load_base_arch():
    config = transformers.AutoConfig.from_pretrained(HUB_ID, num_labels=2, trust_remote_code=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, config=config, trust_remote_code=True,
    )
    return model, config


def build_variant(test_ds, kind: str, seed: int = 42):
    rng = random.Random(seed)
    seqs = list(test_ds["sequence"])
    if kind == "real":
        new_seqs = seqs
    elif kind == "shuffled":
        new_seqs = ["".join(random.Random(seed + i).sample(s, len(s))) for i, s in enumerate(seqs)]
    elif kind == "random_dna":
        new_seqs = ["".join(random.Random(seed + i).choices(BASES, k=len(s))) for i, s in enumerate(seqs)]
    else:
        raise ValueError(kind)
    df = test_ds.to_pandas()
    df["sequence"] = new_seqs
    from datasets import Dataset
    return Dataset.from_pandas(df, preserve_index=False)


def mwu(a, b, label):
    u, p = mannwhitneyu(a, b, alternative="greater")
    return {"comparison": label, "U": u, "p": p}


def summarize(name, score_col, per_variant_df, results_table, mwu_table):
    real = per_variant_df["real"][score_col]
    for kind in ["real", "shuffled", "random_dna"]:
        d = per_variant_df[kind]
        results_table.append({
            "method": name, "variant": kind, "n": len(d),
            "accuracy": (d["pred"] == d["labels"]).mean() if "pred" in d and "labels" in d else float("nan"),
            f"{score_col}_mean": d[score_col].mean(), f"{score_col}_std": d[score_col].std(),
        })
    for kind in ["shuffled", "random_dna"]:
        r = mwu(per_variant_df[kind][score_col], real, f"{name}: {kind} > real")
        r["method"] = name
        mwu_table.append(r)


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        HUB_ID, model_max_length=75, padding_side="right", use_fast=True, trust_remote_code=True,
    )
    raw_test = load_NT_tasks(task="promoter_all", split="test")
    raw_train = load_NT_tasks(task="promoter_all", split="train")

    variants_raw = {k: build_variant(raw_test, k) for k in ["real", "shuffled", "random_dna"]}

    def _prep(ds):
        tokenized, collator = prep_for_trainer(ds, tokenizer, metadata_cols=())
        return tokenized.remove_columns(["sequence"]), collator

    variants = {k: _prep(v) for k, v in variants_raw.items()}

    results_table, mwu_table = [], []

    # ---------------- base + base_scaled (share one forward pass) ----------------
    base_model, config = load_base_arch()
    base_model.load_state_dict(load_file(os.path.join(BASE_CKPT, "model.safetensors")), strict=True)
    base_model = base_model.to(DEVICE).eval()

    def run_base(temperature):
        out = {}
        for kind, (ds, collator) in variants.items():
            loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
            rows = []
            with torch.no_grad():
                for batch in loader:
                    labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                    inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                    logits = base_model(**inputs).logits / temperature
                    unc = compute_uncertainty(logits.unsqueeze(0).cpu())
                    for i in range(logits.shape[0]):
                        rows.append({
                            "labels": int(labels[i]), "pred": int(unc["predicted_class"][i]),
                            "U_total": float(unc["normalized_total_uncertainty"][i]),
                        })
            out[kind] = pd.DataFrame(rows)
        return out

    print("=== base ===")
    base_out = run_base(1.0)
    summarize("base", "U_total", base_out, results_table, mwu_table)

    print("=== base_scaled ===")
    scaled_out = run_base(TEMPERATURE)
    summarize("base_scaled", "U_total", scaled_out, results_table, mwu_table)

    # ---------------- mc_dropout ----------------
    print("=== mc_dropout ===")
    enable_mc_dropout(base_model, p=0.1)
    mc_out = {}
    for kind, (ds, collator) in variants.items():
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
        rows = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                logits_all = torch.stack([base_model(**inputs).logits for _ in range(16)], dim=0).cpu()
                unc = compute_uncertainty(logits_all)
                for i in range(logits_all.shape[1]):
                    rows.append({
                        "labels": int(labels[i]), "pred": int(unc["predicted_class"][i]),
                        "U_total": float(unc["normalized_total_uncertainty"][i]),
                        "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                        "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                    })
        mc_out[kind] = pd.DataFrame(rows)
    summarize("mc_dropout", "U_total", mc_out, results_table, mwu_table)
    summarize("mc_dropout", "U_epistemic", mc_out, results_table, mwu_table)
    summarize("mc_dropout", "U_aleatoric", mc_out, results_table, mwu_table)
    base_model.eval()  # turn dropout back off before reuse below

    # ---------------- conv_epinet ----------------
    print("=== conv_epinet ===")
    epi_base, _ = load_base_arch()
    epi_cfg = EpinetConfig(num_classes=2, include_inputs=True, vocab_size=config.vocab_size)
    wrapper = EpinetWrapper(epi_base, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    epi_model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=16).to(DEVICE)

    # ProjectedMLP.core is lazily built on first forward (its input dim
    # depends on the concatenated [z, hidden] width, only known once real
    # data flows through) -- a freshly-constructed model has no
    # train_head.core/prior_head.core parameters yet, so load_state_dict
    # would reject the checkpoint's saved core.* keys as "unexpected". Run
    # one dummy forward pass first to trigger the lazy build, matching
    # what happened during training before the checkpoint was saved.
    _dummy_ds, _dummy_collator = variants["real"]
    _dummy_batch = _dummy_collator([_dummy_ds[i] for i in range(2)])
    _dummy_batch = {k: v.to(DEVICE) for k, v in _dummy_batch.items() if k != "labels"}
    with torch.no_grad():
        epi_model.wrapper(_dummy_batch, n_index_samples=1)

    epi_model.load_state_dict(load_file(os.path.join(EPI_CKPT, "model.safetensors")), strict=True)
    epi_model.eval()

    epi_out = {}
    for kind, (ds, collator) in variants.items():
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
        rows = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels" if "labels" in batch else "label").to(DEVICE)
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                logits_all = epi_model.wrapper(inputs, n_index_samples=16, return_all=True).cpu()
                unc = compute_uncertainty(logits_all)
                for i in range(logits_all.shape[1]):
                    rows.append({
                        "labels": int(labels[i]), "pred": int(unc["predicted_class"][i]),
                        "U_total": float(unc["normalized_total_uncertainty"][i]),
                        "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                        "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                    })
        epi_out[kind] = pd.DataFrame(rows)
    summarize("conv_epinet", "U_total", epi_out, results_table, mwu_table)
    summarize("conv_epinet", "U_epistemic", epi_out, results_table, mwu_table)
    summarize("conv_epinet", "U_aleatoric", epi_out, results_table, mwu_table)
    del epi_base, wrapper, epi_model
    torch.cuda.empty_cache()

    # ---------------- laplace ----------------
    print("=== laplace ===")
    train_ds, train_collator = _prep(raw_train)
    lap_cfg = LaplaceConfig(classifier_attr="classifier", prior_precision=1000.0, max_examples=4000)
    weight_var, bias_var, n_fit = fit_diagonal_laplace(base_model, train_ds, train_collator, lap_cfg, batch_size=64)
    print(f"  fit on {n_fit} training examples")
    lap_model = LaplaceSeqClassifier(base_model, weight_var, bias_var).to(DEVICE)

    lap_out = {}
    for kind, (ds, collator) in variants.items():
        rows = predict_laplace(lap_model, ds, collator, k_samples=16, batch_size=64, outfile=None)
        lap_out[kind] = pd.DataFrame(rows)
    summarize("laplace", "U_total", lap_out, results_table, mwu_table)
    summarize("laplace", "U_epistemic", lap_out, results_table, mwu_table)
    summarize("laplace", "U_aleatoric", lap_out, results_table, mwu_table)

    # ---------------- conformal ----------------
    print("=== conformal ===")
    real_ds, real_collator = variants["real"]
    loader = torch.utils.data.DataLoader(real_ds, batch_size=64, shuffle=False, collate_fn=real_collator)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels" if "labels" in batch else "label")
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            probs = F.softmax(base_model(**inputs).logits, dim=-1).cpu()
            all_probs.append(probs)
            all_labels.append(labels)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(all_labels), generator=g)
    n_cal = len(all_labels) // 2
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
    cal_scores = score_nonconformity(all_probs[cal_idx], all_labels[cal_idx])
    qhat = fit_conformal_threshold(cal_scores, alpha=0.10)
    print(f"  qhat (target coverage 0.90) = {qhat:.4f}")

    def conformal_set_sizes(ds, collator, restrict_idx=None):
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels" if "labels" in batch else "label")
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                probs = F.softmax(base_model(**inputs).logits, dim=-1).cpu()
                probs_list.append(probs)
                labels_list.append(labels)
        probs_all = torch.cat(probs_list)
        labels_all = torch.cat(labels_list)
        if restrict_idx is not None:
            probs_all, labels_all = probs_all[restrict_idx], labels_all[restrict_idx]
        membership = predict_sets(probs_all, qhat, method="score")
        set_size = membership.sum(dim=1).float()
        covered = membership.gather(1, labels_all.view(-1, 1).long()).squeeze(1)
        pred = probs_all.argmax(dim=-1)
        return pd.DataFrame({
            "labels": labels_all.numpy(), "pred": pred.numpy(),
            "set_size": set_size.numpy(), "covered": covered.numpy(),
        })

    conf_out = {
        "real": conformal_set_sizes(real_ds, real_collator, restrict_idx=eval_idx),
        "shuffled": conformal_set_sizes(*variants["shuffled"]),
        "random_dna": conformal_set_sizes(*variants["random_dna"]),
    }
    print(f"  empirical coverage on eval half of real test: {conf_out['real']['covered'].mean():.4f}")
    summarize("conformal", "set_size", conf_out, results_table, mwu_table)

    # ---------------- evidential (reuse already-computed CSV) ----------------
    print("=== evidential (reusing prior retest) ===")
    evid_csv = "data_gen/label_noise/evidential_ood_retest.csv"
    if os.path.exists(evid_csv):
        evid_df = pd.read_csv(evid_csv)
        evid_out = {k: evid_df[evid_df["kind"] == k] for k in ["real", "shuffled", "random_dna"]}
        summarize("evidential", "vacuity", evid_out, results_table, mwu_table)
    else:
        print(f"  {evid_csv} not found, skipping evidential (rerun evidential_ood_retest.py first)")

    # ---------------- write results ----------------
    results_df = pd.DataFrame(results_table)
    mwu_df = pd.DataFrame(mwu_table)
    results_df.to_csv("data_gen/label_noise/all_methods_ood_results.csv", index=False)
    mwu_df.to_csv("data_gen/label_noise/all_methods_ood_mwu.csv", index=False)
    print("\n=== RESULTS ===")
    print(results_df.to_string(index=False))
    print("\n=== MANN-WHITNEY (proxy > real) ===")
    print(mwu_df.to_string(index=False))


if __name__ == "__main__":
    main()
