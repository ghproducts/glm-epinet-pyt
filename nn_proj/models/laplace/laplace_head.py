# laplace_head.py
"""
Last-layer diagonal Laplace approximation (Daxberger, Kristiadi, Immer,
Eschenhagen, Bauer & Hennig 2021, "Laplace Redux -- Effortless Bayesian
Deep Learning", https://arxiv.org/abs/2106.14806), implemented by hand
rather than via `laplace-torch` (that package's HF-dict-batch predictive
path and this repo's shared-venv curvlinops pin both conflicted with it).

Only the final linear classifier head is treated as Bayesian; the frozen
backbone is a deterministic feature extractor h(x). The diagonal of the
generalized Gauss-Newton matrix for a linear-softmax head,
`GGN_{w_cj,w_cj} = p_c(1-p_c) * h_j^2` (`p = softmax(z)`), is accumulated
per example, combined with a scalar Gaussian prior precision, and inverted
for a posterior variance per weight -- no autograd through the backbone,
matching `laplace-torch`'s own `hessian_structure="diag"`.

`LaplaceSeqClassifier.sample_logits` draws K Monte-Carlo weight samples
per batch from that posterior and returns a `[K, B, C]` logit stack, so
`nn_proj.common.utils.compute_uncertainty` works unmodified -- one backbone
forward per batch, K cheap head-only forwards, mirroring
`nn_proj.models.epinet.epinet.predict`'s structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from nn_proj.common.utils import compute_uncertainty

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_submodule(model: nn.Module, dotted_name: str) -> nn.Module:
    mod = model
    for part in dotted_name.split("."):
        mod = getattr(mod, part)
    return mod


@dataclass
class LaplaceConfig:
    classifier_attr: str = "classifier"  # dotted name of the model's final nn.Linear head
    prior_precision: float = 1.0         # scalar Gaussian prior precision (~ weight decay)
    max_examples: Optional[int] = None   # cap the fitting pass for speed; see fit_diagonal_laplace


@torch.no_grad()
def fit_diagonal_ggn(
    model: nn.Module,
    dataset,
    collator: Any,
    cfg: LaplaceConfig,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fits the *raw* (unregularized -- no prior precision added yet) diagonal
    GGN/Fisher approximation described in the module docstring on `dataset`
    (typically the training set, or a subset of it for speed -- see
    `cfg.max_examples`). Forward-only: no backprop through the backbone,
    just a forward hook on the classifier layer to capture its input
    features.

    Split out from `fit_diagonal_laplace` so `tune_prior_precision` can fit
    the curvature once and cheaply re-evaluate multiple candidate
    `prior_precision` values against it, rather than refitting per
    candidate.

    Returns (weight_ggn [C, H], bias_ggn [C], n_examples_used), on
    `model`'s device (not moved to cpu).
    """
    classifier = _get_submodule(model, cfg.classifier_attr)
    if not isinstance(classifier, nn.Linear):
        raise ValueError(f"model.{cfg.classifier_attr} is a {type(classifier)}, expected nn.Linear.")

    C, H = classifier.weight.shape
    device = classifier.weight.device

    weight_ggn = torch.zeros(C, H, device=device)
    bias_ggn = torch.zeros(C, device=device)

    captured: Dict[str, torch.Tensor] = {}

    def _hook(_module, inputs, _output):
        captured["h"] = inputs[0].detach()

    handle = classifier.register_forward_hook(_hook)

    model.eval()
    if "sequence" in dataset.column_names:
        dataset = dataset.remove_columns(["sequence"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    n_seen = 0
    try:
        for batch in tqdm.tqdm(loader, desc="Fitting diagonal Laplace"):
            labels_key = "labels" if "labels" in batch else ("label" if "label" in batch else None)
            if labels_key is not None:
                batch = {k: v for k, v in batch.items() if k != labels_key}
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            logits = out.logits if hasattr(out, "logits") else out["logits"]
            probs = F.softmax(logits.float(), dim=-1)  # [B,C]
            h = captured["h"].float()                   # [B,H]
            pq = probs * (1.0 - probs)                   # [B,C]

            weight_ggn += torch.einsum("bc,bh->ch", pq, h ** 2)
            bias_ggn += pq.sum(dim=0)
            n_seen += h.shape[0]

            if cfg.max_examples is not None and n_seen >= cfg.max_examples:
                break
    finally:
        handle.remove()

    if cfg.max_examples is not None and n_seen < len(dataset):
        # Rescale the subset's curvature estimate up to what the full
        # dataset would contribute on average -- an approximation (assumes
        # the subset is representative), not exact. Documented as such in
        # .docs/NEW_UQ_METHODS.md.
        scale = len(dataset) / max(n_seen, 1)
        weight_ggn = weight_ggn * scale
        bias_ggn = bias_ggn * scale

    return weight_ggn, bias_ggn, n_seen


@torch.no_grad()
def fit_diagonal_laplace(
    model: nn.Module,
    dataset,
    collator: Any,
    cfg: LaplaceConfig,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fits the diagonal GGN/Fisher approximation (via `fit_diagonal_ggn`) and
    adds `cfg.prior_precision` to get the posterior variance. See
    `fit_diagonal_ggn` for the fitting details and `tune_prior_precision`
    below for an opt-in way to choose `cfg.prior_precision` instead of
    hand-picking it.

    Returns (weight_var [C, H], bias_var [C], n_examples_used).
    """
    weight_ggn, bias_ggn, n_seen = fit_diagonal_ggn(model, dataset, collator, cfg, batch_size=batch_size)
    weight_var = 1.0 / (cfg.prior_precision + weight_ggn)
    bias_var = 1.0 / (cfg.prior_precision + bias_ggn)
    return weight_var.cpu(), bias_var.cpu(), n_seen


@torch.no_grad()
def tune_prior_precision(
    model: nn.Module,
    fit_dataset,
    fit_collator: Any,
    val_dataset,
    val_collator: Any,
    cfg: LaplaceConfig,
    candidates: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    k_samples: int = 32,
    batch_size: int = 32,
) -> Tuple[float, Dict[float, float]]:
    """
    Opt-in `prior_precision` selection: grid search over `candidates`,
    minimizing Monte-Carlo predictive NLL on a held-out validation slice
    (`val_dataset`, disjoint from `fit_dataset`) -- the same spirit as
    `scaling.py`'s temperature fit (minimize NLL on held-out data), and
    standard practice for Laplace approximations (Daxberger et al. 2021;
    `laplace-torch`'s own `optimize_prior_precision` does the analogous
    search, via gradient-based marginal-likelihood optimization rather than
    a plain grid here -- a grid is simpler to get right by hand and is a
    perfectly standard alternative for a scalar hyperparameter).

    Fits the raw (unregularized) GGN once via `fit_diagonal_ggn` -- the
    expensive forward pass over `fit_dataset` -- then, for each candidate,
    only adds the prior and re-scores the validation NLL (cheap: one base
    forward + `k_samples` head samples per validation batch).

    `cfg.prior_precision` is ignored (the candidates are used instead);
    every other `cfg` field (`classifier_attr`, `max_examples`) behaves as
    in `fit_diagonal_laplace`.

    Returns `(best_precision, {candidate: mean_val_nll})`. Off by default
    everywhere this is wired up (`--tune_prior_precision` in
    `inference_laplace.py`) -- the hardcoded-constant default behavior is
    unchanged unless a caller opts in.
    """
    weight_ggn, bias_ggn, _ = fit_diagonal_ggn(model, fit_dataset, fit_collator, cfg, batch_size=batch_size)

    if "sequence" in val_dataset.column_names:
        val_dataset = val_dataset.remove_columns(["sequence"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_collator)
    val_batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]] = []
    for batch in val_loader:
        labels_key = "labels" if "labels" in batch else ("label" if "label" in batch else None)
        labels = batch[labels_key]
        inputs = {k: v for k, v in batch.items() if k != labels_key}
        val_batches.append((inputs, labels))

    device = weight_ggn.device
    results: Dict[float, float] = {}
    for precision in candidates:
        weight_var = 1.0 / (precision + weight_ggn)
        bias_var = 1.0 / (precision + bias_ggn)
        lap_model = LaplaceSeqClassifier(
            model, weight_var, bias_var, classifier_attr=cfg.classifier_attr
        ).to(device)
        lap_model.eval()

        total_nll, n_total = 0.0, 0
        for inputs, labels in val_batches:
            inputs_dev = {k: v.to(device) for k, v in inputs.items()}
            labels_dev = labels.to(device).long()

            lap_model(**inputs_dev)
            logits_all = lap_model.sample_logits(n_samples=k_samples)  # [K,B,C]
            probs = F.softmax(logits_all.float(), dim=-1).mean(dim=0)  # MC-averaged predictive, [B,C]
            nll = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels_dev, reduction="sum")

            total_nll += nll.item()
            n_total += labels_dev.shape[0]

        results[float(precision)] = total_nll / max(n_total, 1)

    best_precision = min(results, key=results.get)
    return best_precision, results


class LaplaceSeqClassifier(nn.Module):
    """
    Wraps a frozen, already-fine-tuned AutoModelForSequenceClassification
    with a sampleable last-layer posterior.

    `forward()` behaves like the plain base model and returns the
    mean-weight logits (identical to `base_model(**batch).logits` -- the
    checkpoint's own classifier weights *are* the posterior mean; nothing
    about the base model's weights changes). `sample_logits(K)` reuses the
    features captured by a forward hook during the most recent `forward()`
    call to draw K posterior samples of the classifier's output without
    rerunning the backbone.
    """

    def __init__(
        self,
        base_model: nn.Module,
        weight_var: torch.Tensor,
        bias_var: torch.Tensor,
        classifier_attr: str = "classifier",
    ):
        super().__init__()
        self.base = base_model
        self.classifier_attr = classifier_attr
        classifier = _get_submodule(base_model, classifier_attr)
        if not isinstance(classifier, nn.Linear):
            raise ValueError(f"base_model.{classifier_attr} is a {type(classifier)}, expected nn.Linear.")

        self.register_buffer("weight_mean", classifier.weight.detach().clone())
        self.register_buffer("bias_mean", classifier.bias.detach().clone())
        self.register_buffer("weight_std", weight_var.clamp_min(0).sqrt().to(classifier.weight.device))
        self.register_buffer("bias_std", bias_var.clamp_min(0).sqrt().to(classifier.weight.device))

        self._features: Optional[torch.Tensor] = None
        classifier.register_forward_hook(self._capture_hook)

    def _capture_hook(self, _module, inputs, _output):
        self._features = inputs[0].detach()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        loss = F.cross_entropy(out.logits, labels) if labels is not None else None
        return {"loss": loss, "logits": out.logits}

    @torch.no_grad()
    def sample_logits(self, n_samples: int = 1) -> torch.Tensor:
        """[K, B, C] logit stack sampled from the fitted posterior, reusing
        the features cached by the most recent `forward()` call (so call
        `forward()` first)."""
        if self._features is None:
            raise RuntimeError("sample_logits() needs a prior forward() call to populate cached features.")
        feat = self._features.float()  # [B, H]
        C, H = self.weight_mean.shape
        eps_w = torch.randn(n_samples, C, H, device=feat.device, dtype=feat.dtype)
        eps_b = torch.randn(n_samples, C, device=feat.device, dtype=feat.dtype)
        W = self.weight_mean.unsqueeze(0) + eps_w * self.weight_std.unsqueeze(0)  # [K,C,H]
        b = self.bias_mean.unsqueeze(0) + eps_b * self.bias_std.unsqueeze(0)      # [K,C]
        logits = torch.einsum("bh,kch->kbc", feat, W) + b.unsqueeze(1)           # [K,B,C]
        return logits


@torch.no_grad()
def predict_laplace(
    model: LaplaceSeqClassifier,
    dataset,
    collator: Any,
    k_samples: int = 16,
    batch_size: int = 32,
    outfile: Optional[str] = None,
    metadata_cols: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Same output-column convention as
    `nn_proj.models.epinet.epinet.predict`: the base columns come from
    `compute_uncertainty` applied to the [K,B,C] posterior-sample logit
    stack, so any downstream code that already knows that schema (e.g.
    nn_proj.analysis) needs no changes to consume it.
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

    idx = 0
    for batch in tqdm.tqdm(loader, desc="Predicting (laplace)"):
        labels_key = "labels" if "labels" in batch else ("label" if "label" in batch else None)
        labels = batch.pop(labels_key).to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}

        model(**inputs)  # one base forward; populates cached pre-classifier features
        logits_all = model.sample_logits(n_samples=k_samples).detach().cpu()  # [K,B,C]
        unc = compute_uncertainty(logits_all)

        B = logits_all.shape[1]
        for i in range(B):
            row = {
                "labels": int(labels[i]),
                "pred": int(unc["predicted_class"][i]),
                "max_confidence": float(unc["max_confidence"][i]),
                "pred-pre_average": float(unc["predicted_class_logitmean"][i]),
                "max_confidence-pre_average": float(unc["max_confidence_logitmean"][i]),
                "U_total": float(unc["normalized_total_uncertainty"][i]),
                "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "vote_pct": float(unc["vote_percentage"][i]),
            }
            rows.append({**row, **metas[idx]})
            idx += 1

    if outfile is not None:
        outdir = os.path.dirname(outfile)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        pd.DataFrame(rows).to_csv(outfile, index=False)
        print(f"[laplace] wrote {outfile} with {len(rows)} rows")

    return rows
