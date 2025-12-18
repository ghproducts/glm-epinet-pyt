from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import sklearn
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)

def enable_mc_dropout(model, p=0.1):
    model.eval()
    for m in model.modules():
        if isinstance(m, dropout_types):
            if p is not None:
                m.p = p
            m.train()
    return model



# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)



"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)



"""
Compute uncertainty metrics from network outputs.
"""
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


