# evidential.py
"""
Evidential deep learning for classification (Sensoy, Kaplan & Kandemir
2018, "Evidential Deep Learning to Quantify Classification Uncertainty",
https://arxiv.org/abs/1806.01768).

Unlike conformal prediction and Laplace, this method is not post-hoc: it
changes both the output head's activation and the training loss. Mirroring
how `nn_proj.models.epinet` is structured as its own backbone-agnostic
package (`EpinetConfig`/`EpinetWrapper`/`HFEpinetSeqClassifier` in
epinet.py, wired up per-backbone in each `train_epinet.py`), this module
holds the parts every backbone would share, and each backbone gets its own
`train_evidential.py` for the HF-Trainer plumbing (checkpoint loading,
tokenizer, dataset) -- see nn_proj/models/DNABERT2/train_evidential.py for
the one implemented here (the task's "at least one backbone").

The idea
--------
Standard classification treats the softmax output as *the* probability
vector. Evidential deep learning instead treats the network's non-negative
"evidence" output e >= 0 (per class) as parameters of a Dirichlet
distribution over the categorical probability simplex: alpha = e + 1, so
alpha_c >= 1 always (a uniform Dirichlet, i.e. total ignorance, exactly
when e = 0 for every class). The predicted probability is the Dirichlet
mean alpha_c / S where S = sum_c(alpha_c), and "vacuity" -- the paper's
belief-mass-remaining-unassigned term, K / S for K classes -- is the
paper's proposed uncertainty signal: S grows only when the network has
accumulated evidence for *some* class, so vacuity shrinks as evidence
accumulates and saturates near 1 (maximum uncertainty, S = K) when e ~= 0
for every class.

Loss (Sensoy et al. eq. 3 and 5): the expected sum-of-squares loss between
the one-hot label and the Dirichlet-distributed probability vector,

    L_i = sum_c (y_ic - alpha_ic/S_i)^2 + alpha_ic(S_i - alpha_ic) / (S_i^2 (S_i+1))

(the digamma/cross-entropy variant, eq. 4, is also implemented as an
alternative -- see `dirichlet_loss(..., loss_type="ce")`), plus a
KL-divergence regularizer that shrinks evidence for whichever classes are
*not* the true label (eq. 3's second term):

    KL[ Dir(alpha_tilde_i) || Dir(1,...,1) ],   alpha_tilde_i = y_i + (1 - y_i) * alpha_i

scaled by an annealing coefficient lambda_t = min(1, t / annealing_step)
that ramps from 0 to 1 over the first `annealing_step` epochs (eq. 5's "we
... slowly increase the effect of the KL divergence in order to prevent
premature convergence"). Without this ramp the KL term can suppress
evidence everywhere before the network has learned anything, per the
paper's own ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from nn_proj.common.utils import compute_uncertainty

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvidentialConfig:
    num_classes: int
    evidence_activation: str = "softplus"  # "softplus" or "relu"
    loss_type: str = "mse"                 # "mse" (eq. 5) or "ce" (eq. 4, digamma)
    annealing_step: int = 10               # epochs until the KL term reaches full weight


def evidence_activation(logits: torch.Tensor, kind: str = "softplus") -> torch.Tensor:
    if kind == "softplus":
        return F.softplus(logits)
    if kind == "relu":
        return F.relu(logits)
    raise ValueError(f"Unknown evidence_activation: {kind!r}")


def _kl_dirichlet_uniform(alpha_tilde: torch.Tensor) -> torch.Tensor:
    """
    KL[Dir(alpha_tilde) || Dir(1,...,1)] per example (Sensoy et al. eq. 3).
    Closed form for KL between two Dirichlets Dir(alpha) and Dir(beta):

        lgamma(sum(alpha)) - sum(lgamma(alpha))
      - lgamma(sum(beta))  + sum(lgamma(beta))
      + sum_c (alpha_c - beta_c) * (digamma(alpha_c) - digamma(sum(alpha)))

    specialized to beta = (1, ..., 1), where lgamma(beta_c) = 0.
    """
    C = alpha_tilde.shape[-1]
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

    lgamma_C = torch.lgamma(torch.tensor(float(C), device=alpha_tilde.device, dtype=alpha_tilde.dtype))
    term1 = torch.lgamma(S_tilde).squeeze(-1) - torch.lgamma(alpha_tilde).sum(-1) - lgamma_C
    term2 = ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(-1)
    return term1 + term2


def dirichlet_loss(
    evidence: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epoch: float,
    annealing_step: int,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Per-batch mean evidential loss (Sensoy et al. eq. 3-5). `epoch` is a
    float (HF Trainer's `state.epoch`), so the KL anneal ramps smoothly
    within an epoch, not just at epoch boundaries.
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    y = F.one_hot(labels, num_classes=num_classes).to(alpha.dtype)

    if loss_type == "mse":
        p = alpha / S
        err = (y - p) ** 2
        var = alpha * (S - alpha) / (S * S * (S + 1.0))
        data_loss = (err + var).sum(dim=-1)
    elif loss_type == "ce":
        data_loss = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    lambda_t = min(1.0, float(epoch) / max(annealing_step, 1))
    alpha_tilde = y + (1.0 - y) * alpha
    kl = _kl_dirichlet_uniform(alpha_tilde)

    return (data_loss + lambda_t * kl).mean()


class EvidentialWrapper(nn.Module):
    """
    Thin wrapper: reinterprets a standard AutoModelForSequenceClassification's
    pre-softmax logits as Dirichlet evidence via a non-negative activation.
    No architecture change is needed beyond that reinterpretation -- Sensoy
    et al.'s own reference implementation likewise just swaps the final
    activation (softmax -> ReLU) on an otherwise unchanged linear head.
    """

    def __init__(self, base_model: nn.Module, cfg: EvidentialConfig):
        super().__init__()
        self.base = base_model
        self.cfg = cfg

    def forward(self, batch: Any) -> torch.Tensor:
        inputs = batch.data if hasattr(batch, "data") else batch
        out = self.base(**inputs)
        return evidence_activation(out.logits, self.cfg.evidence_activation)


class HFEvidentialSeqClassifier(nn.Module):
    """
    HF-Trainer-compatible wrapper, structured like
    `nn_proj.models.epinet.epinet.HFEpinetSeqClassifier`. Unlike that
    class, this one does *not* compute the loss inside `forward()`: the
    annealing coefficient needs the current epoch, which HF `Trainer` only
    exposes on `Trainer.state` -- not inside `model.forward()` -- so loss
    computation is delegated to `EvidentialTrainer.compute_loss` in each
    backbone's train_evidential.py, which has access to `self.state.epoch`.
    `forward()` therefore always returns `loss=None`; `logits` is the raw
    evidence tensor (argmax(evidence) == argmax(alpha), so
    `compute_metrics`/`preprocess_logits_for_metrics` need no changes).
    """

    def __init__(self, wrapper: EvidentialWrapper):
        super().__init__()
        self.wrapper = wrapper

    @property
    def cfg(self) -> EvidentialConfig:
        return self.wrapper.cfg

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        evidence = self.wrapper(batch)
        return {"loss": None, "logits": evidence}


@torch.no_grad()
def predict_evidential(
    model: HFEvidentialSeqClassifier,
    dataset,
    collator: Any,
    batch_size: int = 32,
    outfile: Optional[str] = None,
    metadata_cols: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Writes the same base columns as
    `nn_proj.models.epinet.epinet.predict` (via `compute_uncertainty`
    applied to the log of the Dirichlet mean probability, i.e. a K=1
    stack -- there is a single Dirichlet mean per example, not an ensemble
    of samples, so `U_epistemic` is identically 0 here, same as it would be
    for plain "base"), plus the two evidential-specific columns the paper
    proposes: `vacuity` = C / sum(alpha), and `dirichlet_strength` =
    sum(alpha) for anyone who wants the raw evidence total rather than the
    normalized vacuity.
    """
    if "sequence" in dataset.column_names:
        dataset = dataset.remove_columns(["sequence"])

    if metadata_cols is not None:
        metadata_cols = [c for c in metadata_cols if c in dataset.column_names]
    else:
        input_label_keys = {"input_ids", "attention_mask", "token_type_ids", "labels", "label"}
        metadata_cols = [c for c in dataset.column_names if c not in input_label_keys]

    def _plain(v):
        return v.item() if isinstance(v, torch.Tensor) else v

    metas = [{col: _plain(dataset[col][i]) for col in metadata_cols} for i in range(len(dataset))]
    dataset = dataset.remove_columns(metadata_cols)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    rows: List[Dict[str, Any]] = []
    model.eval()
    num_classes = model.cfg.num_classes

    idx = 0
    for batch in tqdm.tqdm(loader, desc="Predicting (evidential)"):
        labels_key = "labels" if "labels" in batch else ("label" if "label" in batch else None)
        labels = batch.pop(labels_key).to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}

        out = model(**inputs)
        evidence = out["logits"].detach().cpu().float()  # [B,C]
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)  # [B,1]
        probs = alpha / S

        unc = compute_uncertainty(torch.log(probs.clamp_min(1e-12)).unsqueeze(0))  # [1,B,C]
        vacuity = num_classes / S.squeeze(-1)  # [B]

        B = evidence.shape[0]
        for i in range(B):
            row = {
                "labels": int(labels[i]),
                "pred": int(unc["predicted_class"][i]),
                "max_confidence": float(unc["max_confidence"][i]),
                "U_total": float(unc["normalized_total_uncertainty"][i]),
                "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "vote_pct": float(unc["vote_percentage"][i]),
                "vacuity": float(vacuity[i]),
                "dirichlet_strength": float(S[i, 0]),
            }
            rows.append({**row, **metas[idx]})
            idx += 1

    if outfile is not None:
        outdir = os.path.dirname(outfile)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        pd.DataFrame(rows).to_csv(outfile, index=False)
        print(f"[evidential] wrote {outfile} with {len(rows)} rows")

    return rows
