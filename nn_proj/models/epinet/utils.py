import os
import math
import random
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from datasets import load_dataset, ClassLabel # type: ignore

from peft import LoraConfig, TaskType, get_peft_model # type: ignore

# custom imports
from epinet import EpinetWrapper, EpinetConfig, ConvAdditivePriorEpinet, MLPEpinetWithPrior
from feature_fns import NT_feature_fn, NT_multi_feature_fn, NT_tokens_feature_fn
from utils import compute_metrics, compute_uncertainty


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 512 # Max length of toneized sequences. only controls dummy init, not enforced in training/inference

from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier
from nn_proj.models.epinet.feature_fns import NT_feature_fn, NT_tokens_feature_fn

# ---------------------------
# Build model + epinet from scratch
# ---------------------------
def build_model_and_tokenizer(config: EpinetConfig, model_name: str) -> (HFEpinetSeqClassifier, AutoTokenizer):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True, num_labels=config.num_classes
    )
    base.config.output_hidden_states = True
    wrapper = EpinetWrapper(base, NT_multi_feature_fn, config, epinet=MLPEpinetWithPrior)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=8).to(DEVICE)

    # build epinet internals
    with torch.no_grad():
        dummy = tok("ACGT", truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
        _ = model(**dummy)  # builds epinet internals

    return model, tok


# ------------------------------
# load model from checkpoint for inference
# ------------------------------
def load_model_and_tokenizer(config: EpinetConfig, checkpoint: str):
    # Rebuild architecture exactly like training and load weights
    model, tok = build_model_and_tokenizer(config)

    # dummy init
    with torch.no_grad():
        dummy = tok("ACGT", truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
        _ = model(**dummy)  # builds epinet internals

    state_path = os.path.join(checkpoint, "model.safetensors")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Could not find checkpoint weights at: {state_path}")
    sd = load_file(state_path)
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()

    return model, tok



# Metrics 
def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(axis=-1)
    labels = eval_pred.label_ids
    try:
        return {"f1_score": f1_score(labels, preds)}
    except Exception:
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}
    

@torch.no_grad()
def compute_uncertainty(epi_out_logits: torch.Tensor):
    """
    Accepts [K, B, C] (preferred) or [B, C].
    Returns a dict of [B]-shaped tensors (predicted_class is long).
    """
    if epi_out_logits.dim() == 2:
        epi_out_logits = epi_out_logits.unsqueeze(0)   # [1, B, C]
    S, B, C = epi_out_logits.shape

    # per-sample probabilities
    per_logp = F.log_softmax(epi_out_logits, dim=-1)   # [S,B,C]
    per_p = per_logp.exp()                              # [S,B,C]

    mean_p = per_p.mean(dim=0)                         # [B,C]
    eps = 1e-9

    # Total (predictive) entropy
    pred_ent = -(mean_p * (mean_p.clamp_min(eps).log())).sum(dim=-1)   # [B]

    # Expected entropy over z (aleatoric)
    per_ent = -(per_p * per_logp).sum(dim=-1)                           # [S,B]
    exp_ent = per_ent.mean(dim=0)                                       # [B]

    # Epistemic = total - aleatoric
    epi = pred_ent - exp_ent                                            # [B]

    # Normalize by log(C)
    norm = torch.log(torch.tensor(float(C), device=epi_out_logits.device))
    out = {
        "predicted_class": mean_p.argmax(dim=-1),                       # [B], long
        "normalized_total_uncertainty": pred_ent / norm,                # [B]
        "normalized_epistemic_uncertainty": epi / norm,                 # [B]
        "normalized_aleatoric_uncertainty": exp_ent / norm,             # [B]
        "max_confidence": mean_p.max(dim=-1).values,                    # [B]
    }

    # Voting agreement
    per_preds = epi_out_logits.argmax(dim=-1)                           # [S,B]
    votes = (per_preds == out["predicted_class"].unsqueeze(0)).sum(dim=0)
    out["vote_percentage"] = votes.float() / float(S)                   # [B]

    # Also return mean probs if you want to store them
    out["mean_probs"] = mean_p                                          # [B,C]

    # additional pre-softmax averaged results
    logits_mean = epi_out_logits.mean(dim=0)                            # [B,C]
    prob_from_logits = F.softmax(logits_mean, dim=-1)                   # [B,C]
    out["predicted_class_logitmean"] = prob_from_logits.argmax(dim=-1)  # [B]
    out["max_confidence_logitmean"]  = prob_from_logits.max(dim=-1).values

    return out


#-----------------------
# inference function
#-----------------------

def _collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}

@torch.no_grad()
def predict(
    model,
    dataset,
    device: torch.device,
    k_samples: int = 16,
    batch_size: int = 32,
    outfile: Optional[str] = "val_uncertainty.csv",
    save_logits_npz: Optional[str] = None, 
    use_amp: bool = False,
) -> List[Dict[str, Any]]:
    """
    One base forward + K epinet head forwards per batch (return_all=True),
    computes uncertainty from the full stack, and optionally saves outputs.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    rows: List[Dict[str, Any]] = []
    all_logits = []
    all_labels = []

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_amp and device.type == "cuda") else torch.cpu.amp.autocast(enabled=False)

    for batch in tqdm(loader, desc="Predicting", total=len(loader)):
        labels = batch.pop("labels").to(device)               # [B]
        inputs = {k: v.to(device) for k, v in batch.items()}

        with amp_ctx:
            # one base forward; K head forwards -> [K, B, C]
            logits_all = model.wrapper(inputs, n_index_samples=k_samples, return_all=True)

        unc = compute_uncertainty(logits_all)           

        B = labels.size(0)
        for i in range(B):
            rows.append({
                "label": int(labels[i]),
                "pred": int(unc["predicted_class"][i]),
                "max_confidence": float(unc["max_confidence"][i]),
                "pred-pre_average": float(unc["predicted_class_logitmean"][i]),
                "max_confidence-pre_average": float(unc["max_confidence_logitmean"][i]),
                "U_total": float(unc["normalized_total_uncertainty"][i]),
                "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "vote_pct": float(unc["vote_percentage"][i]),
            })

        if save_logits_npz is not None:
            all_logits.append(logits_all.cpu())               # [K,B,C]
            all_labels.append(labels.cpu())

    if outfile is not None:
        outdir = os.path.dirname(outfile)
        if outdir:  
            os.makedirs(outdir, exist_ok=True)
        pd.DataFrame(rows).to_csv(outfile, index=False)
        print(f"[uncertainty] wrote {outfile} with {len(rows)} rows")

    if save_logits_npz is not None:
        import numpy as np
        # stack batches into [K, N, C] and [N]
        logits = torch.cat(all_logits, dim=1).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        np.savez_compressed(save_logits_npz, logits_all=logits, labels=labels)
        print(f"[uncertainty] saved logits stack to {save_logits_npz} (shape {logits.shape})")

    return rows