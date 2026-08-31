"""Calibration, selective-prediction, and OOD-detection metrics.

Every metric here operates on a prediction frame: one row per evaluated
sequence, as written by ``nn_proj.models.epinet.predict``. The columns used
are ``labels``, ``pred``, ``max_confidence``, and the normalised uncertainty
scores ``U_total`` / ``U_aleatoric`` / ``U_epistemic``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# np.trapz was removed in NumPy 2.0 in favour of np.trapezoid.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Uncertainty columns for which a *higher* value means "more uncertain".
# ``max_confidence`` is the exception: higher means more confident.
UNCERTAINTY_COLS = ("U_total", "U_aleatoric", "U_epistemic")
CONFIDENCE_COLS = ("max_confidence", "max_confidence-pre_average", "vote_pct")


def higher_is_more_uncertain(score_col: str) -> bool:
    """Return whether larger values of ``score_col`` mean more uncertainty."""
    if score_col in CONFIDENCE_COLS:
        return False
    return True


def correctness(df: pd.DataFrame, label_col: str = "labels", pred_col: str = "pred") -> np.ndarray:
    """Boolean correctness vector, tolerant of int/str label dtype mismatch.

    ``inference.py`` maps predictions back through ``id2label``, which can
    round-trip an integer label column into strings. Compare as strings when
    the dtypes disagree so a dtype accident cannot silently zero out accuracy.
    """
    labels, preds = df[label_col], df[pred_col]
    if labels.dtype != preds.dtype:
        labels, preds = labels.astype(str), preds.astype(str)
    return (labels.to_numpy() == preds.to_numpy())


def _finite(values: np.ndarray, *others: np.ndarray):
    """Drop positions where ``values`` is not finite, from every array given."""
    good = np.isfinite(values)
    return (values[good],) + tuple(o[good] for o in others)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def compute_ece(
    df: pd.DataFrame,
    label_col: str = "labels",
    pred_col: str = "pred",
    conf_col: str = "max_confidence",
    n_bins: int = 50,
    binning: str = "equal_mass",
    clip: tuple = (0.0, 1.0),
) -> float:
    """Expected calibration error (Guo et al., 2017).

    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)| where

      acc(B_m)  = fraction of examples in bin m whose predicted class is correct
      conf(B_m) = mean predicted top-class probability of examples in bin m

    Parameters
    ----------
    n_bins:
        Number of bins M. Reported results use 50.
    binning:
        ``"equal_mass"`` places bin edges at confidence quantiles, so every bin
        holds ~n/M examples. ``"equal_width"`` uses fixed-width edges on
        [0, 1]. Equal-mass is the reported default; equal-width is provided for
        the bin-sensitivity analysis.

    Returns NaN when no finite confidences are present.
    """
    conf = df[conf_col].to_numpy(dtype=float)
    correct = correctness(df, label_col, pred_col).astype(float)
    conf, correct = _finite(conf, correct)

    if conf.size == 0:
        return float("nan")

    lo, hi = clip
    conf = np.clip(conf, lo, hi)

    if binning == "equal_mass":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = lo, hi
        edges = np.unique(edges)  # ties collapse repeated quantiles
    elif binning == "equal_width":
        edges = np.linspace(lo, hi, n_bins + 1)
    else:
        raise ValueError(f"binning must be 'equal_mass' or 'equal_width', got {binning!r}")

    if edges.size < 2:
        return float("nan")

    bin_ids = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, edges.size - 2)

    n = conf.size
    ece = 0.0
    for b in range(edges.size - 1):
        m = bin_ids == b
        n_b = int(m.sum())
        if n_b == 0:
            continue
        ece += (n_b / n) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def ece_bin_sensitivity(
    df: pd.DataFrame,
    bin_counts: Sequence[int] = (5, 10, 15, 20, 30, 50, 100),
    binnings: Sequence[str] = ("equal_mass", "equal_width"),
    **kwargs,
) -> pd.DataFrame:
    """ECE across bin counts and binning schemes.

    Addresses the concern that a single (M, scheme) choice can drive the
    reported number. Returns a long frame: ``n_bins, binning, ece``.
    """
    rows = []
    for binning in binnings:
        for m in bin_counts:
            rows.append({
                "n_bins": m,
                "binning": binning,
                "ece": compute_ece(df, n_bins=m, binning=binning, **kwargs),
            })
    return pd.DataFrame(rows)


def reliability_bins(
    df: pd.DataFrame,
    label_col: str = "labels",
    pred_col: str = "pred",
    conf_col: str = "max_confidence",
    n_bins: int = 20,
    binning: str = "equal_width",
    min_count: int = 1,
) -> pd.DataFrame:
    """Per-bin accuracy/confidence table underlying a reliability diagram.

    Returns ``bin_left, bin_right, count, accuracy, confidence``. Bins holding
    fewer than ``min_count`` examples are returned with NaN accuracy so callers
    can drop them without changing the bin grid across models.
    """
    conf = df[conf_col].to_numpy(dtype=float)
    correct = correctness(df, label_col, pred_col).astype(float)
    conf, correct = _finite(conf, correct)
    conf = np.clip(conf, 0.0, 1.0)

    if binning == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.unique(np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size >= 2:
            edges[0], edges[-1] = 0.0, 1.0

    bin_ids = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, max(edges.size - 2, 0))

    rows = []
    for b in range(edges.size - 1):
        m = bin_ids == b
        count = int(m.sum())
        enough = count >= min_count
        rows.append({
            "bin_left": edges[b],
            "bin_right": edges[b + 1],
            "count": count,
            "accuracy": float(correct[m].mean()) if enough else np.nan,
            "confidence": float(conf[m].mean()) if enough else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Proper scoring rules
# --------------------------------------------------------------------------

def _probability_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Recover the full class-probability matrix from ``prob_*`` columns."""
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        return None
    return df[prob_cols].to_numpy(dtype=float)


def compute_nll(df: pd.DataFrame, label_col: str = "labels", eps: float = 1e-12) -> float:
    """Negative log-likelihood of the true class.

    Requires the per-class ``prob_*`` columns; returns NaN when the prediction
    frame stored only the top-class confidence.
    """
    probs = _probability_matrix(df)
    if probs is None:
        return float("nan")
    prob_cols = [c[len("prob_"):] for c in df.columns if c.startswith("prob_")]
    col_index = {name: i for i, name in enumerate(prob_cols)}
    labels = df[label_col].astype(str).to_numpy()

    idx = np.array([col_index.get(l, -1) for l in labels])
    valid = idx >= 0
    if not valid.any():
        return float("nan")
    p_true = probs[np.arange(len(idx))[valid], idx[valid]]
    return float(-np.log(np.clip(p_true, eps, 1.0)).mean())


def compute_brier(df: pd.DataFrame, label_col: str = "labels") -> float:
    """Multiclass Brier score: mean squared error of the probability vector.

    Requires the per-class ``prob_*`` columns; returns NaN otherwise.
    """
    probs = _probability_matrix(df)
    if probs is None:
        return float("nan")
    prob_cols = [c[len("prob_"):] for c in df.columns if c.startswith("prob_")]
    col_index = {name: i for i, name in enumerate(prob_cols)}
    labels = df[label_col].astype(str).to_numpy()

    idx = np.array([col_index.get(l, -1) for l in labels])
    valid = idx >= 0
    if not valid.any():
        return float("nan")
    probs, idx = probs[valid], idx[valid]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(idx)), idx] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# --------------------------------------------------------------------------
# Selective prediction (risk-coverage)
# --------------------------------------------------------------------------

def confidence_oracle_curve(
    df: pd.DataFrame,
    score_col: str = "U_total",
    correct_col: str = "correct",
    higher_score_more_uncertain: Optional[bool] = None,
) -> tuple:
    """Risk-coverage curve for selective classification.

    Rejects the most-uncertain examples first and reports the error rate on
    what remains.

      x: coverage, the fraction of examples retained
      y: error rate among retained examples

    The oracle curve rejects genuinely-wrong predictions first (the best any
    ranking could do); the random curve is flat at the base error rate.

    This is the direct test of whether uncertainty is *adaptive* — whether it
    identifies which specific examples are unreliable — as opposed to merely
    being globally better calibrated. A method can improve ECE by flattening
    every prediction while leaving this curve unchanged.

    Returns
    -------
    curve_df : coverage, confidence_error, oracle_error, random_error
    auco : area between the confidence and oracle curves (lower is better)
    aurc : area under the confidence curve (lower is better)
    """
    if higher_score_more_uncertain is None:
        higher_score_more_uncertain = higher_is_more_uncertain(score_col)

    score = df[score_col].to_numpy(dtype=float)
    if correct_col in df.columns:
        correct = df[correct_col].to_numpy(dtype=bool)
    else:
        correct = correctness(df)

    score, correct = _finite(score, correct)
    errors = (~correct.astype(bool)).astype(float)

    n = len(score)
    if n == 0:
        raise ValueError(f"No finite values in {score_col!r}.")

    conf_order = np.argsort(-score) if higher_score_more_uncertain else np.argsort(score)
    oracle_order = np.argsort(-errors)

    def retained_error(order):
        e = errors[order]
        rejected = np.concatenate([[0.0], np.cumsum(e)])
        n_retained = n - np.arange(0, n + 1)
        remaining = errors.sum() - rejected
        keep = n_retained > 0
        # increasing coverage, 0 -> 1
        return (n_retained[keep] / n)[::-1], (remaining[keep] / n_retained[keep])[::-1]

    coverage, conf_err = retained_error(conf_order)
    _, oracle_err = retained_error(oracle_order)
    random_err = np.full_like(conf_err, errors.mean(), dtype=float)

    curve_df = pd.DataFrame({
        "coverage": coverage,
        "confidence_error": conf_err,
        "oracle_error": oracle_err,
        "random_error": random_err,
    })
    return (curve_df,
            float(_trapezoid(conf_err - oracle_err, coverage)),
            float(_trapezoid(conf_err, coverage)))


def compute_aurc(df: pd.DataFrame, score_col: str = "U_total", correct_col: str = "correct", **kwargs) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    return confidence_oracle_curve(df, score_col, correct_col, **kwargs)[2]


def compute_auco(df: pd.DataFrame, score_col: str = "U_total", correct_col: str = "correct", **kwargs) -> float:
    """Area between the risk-coverage curve and its oracle. Lower is better."""
    return confidence_oracle_curve(df, score_col, correct_col, **kwargs)[1]


def accuracy_by_uncertainty(
    df: pd.DataFrame,
    score_col: str = "U_total",
    n_bins: int = 10,
    label_col: str = "labels",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Accuracy within equal-mass bins of an uncertainty score.

    Adaptive uncertainty produces a monotone decreasing accuracy profile;
    uniformly shrunk confidence produces a flat one.
    """
    score = df[score_col].to_numpy(dtype=float)
    correct = correctness(df, label_col, pred_col).astype(float)
    score, correct = _finite(score, correct)
    if score.size == 0:
        return pd.DataFrame(columns=["bin", "score_mean", "count", "accuracy"])

    edges = np.unique(np.quantile(score, np.linspace(0.0, 1.0, n_bins + 1)))
    bin_ids = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, max(edges.size - 2, 0))

    rows = []
    for b in range(max(edges.size - 1, 0)):
        m = bin_ids == b
        if not m.any():
            continue
        rows.append({
            "bin": b,
            "score_mean": float(score[m].mean()),
            "count": int(m.sum()),
            "accuracy": float(correct[m].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# OOD detection
# --------------------------------------------------------------------------

def compute_auroc(
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    score_col: str = "U_total",
    higher_score_more_uncertain: Optional[bool] = None,
) -> float:
    """AUROC for separating OOD from ID examples using an uncertainty score.

    ID examples are the negative class, OOD the positive class, so 0.5 is
    chance and values below 0.5 mean the score is *anti*-correlated with
    being out of distribution.
    """
    if higher_score_more_uncertain is None:
        higher_score_more_uncertain = higher_is_more_uncertain(score_col)

    id_scores = id_df[score_col].to_numpy(dtype=float)
    ood_scores = ood_df[score_col].to_numpy(dtype=float)
    id_scores = id_scores[np.isfinite(id_scores)]
    ood_scores = ood_scores[np.isfinite(ood_scores)]

    if id_scores.size == 0 or ood_scores.size == 0:
        return float("nan")

    y = np.concatenate([np.zeros(id_scores.size, dtype=int), np.ones(ood_scores.size, dtype=int)])
    s = np.concatenate([id_scores, ood_scores])
    if not higher_score_more_uncertain:
        s = -s
    return float(roc_auc_score(y, s))
