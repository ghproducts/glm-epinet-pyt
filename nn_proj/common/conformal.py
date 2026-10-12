# conformal.py
"""
Split conformal prediction for classification (Vovk, Gammerman & Shafer
2005; Romano, Sesia & Candès 2020, "Classification with Valid and Adaptive
Coverage Sets", https://arxiv.org/abs/2006.02544, for the "aps" score
below).

Post-hoc and model-agnostic: everything here operates on already-computed
softmax probabilities plus integer labels, so this module has no
dependency on any specific backbone -- it lives in `nn_proj.common`
alongside `utils.py`/`datasets.py` rather than under `nn_proj.models`,
the same way `blast.py`/`mmseqs.py` are backbone-agnostic baselines.

The recipe (Angelopoulos & Bates 2021, "A Gentle Introduction to Conformal
Prediction", Algorithm 1):

  1. Pick a *nonconformity score* s(x, y) that is small when the model is
     confident and correct.
  2. On a calibration split disjoint from, but *exchangeable with*, the
     points being evaluated, compute `qhat`, the ceil((n+1)(1-alpha))/n
     empirical quantile of the calibration scores.
  3. At test time, the prediction set for a fresh x is every class c with
     s(x, c) <= qhat.

Under only the exchangeability of calibration and test points (no
distributional assumption on the model or the data), this gives marginal
coverage >= 1 - alpha: P(y_test in set(x_test)) >= 1 - alpha.

IMPORTANT -- exchangeability is not optional: an earlier version of the
per-backbone plumbing (e.g. `nn_proj/models/DNABERT2/conformal.py`) drew
the calibration split from the *training* data (reusing `scaling.py`'s
90/10 `train_test_split` convention) and evaluated against the dataset's
separate, official `test` split. That is a real exchangeability violation,
not a theoretical nitpick: on `promoter_all`/DNABERT2 it produced a
measured, reproducible ~1-3 point *under*-coverage at
`target_coverage=0.80` across 5 independent calibration draws (see
`.docs/NEW_UQ_METHODS_FIXES.md` for the exact numbers). The fix applied
there -- and the pattern any new per-backbone `conformal.py` should
follow -- is to draw the calibration subset from a random slice of the
*same* pool being evaluated (e.g. a stratified split of the official test
set into a calibration fraction and an evaluation fraction), never from
training-derived data. This necessarily means "coverage" is reported
against only the held-out evaluation fraction of the test split, not the
full official test set the other UQ methods in this repo evaluate
against -- an expected, necessary trade-off of split conformal, not a
bug, but one that must stay documented wherever it's used.

Two scores are implemented:

  * "score" (the textbook baseline, sometimes called "least ambiguous set"
    or LAC): s(x, y) = 1 - softmax(x)_y. Angelopoulos & Bates note this is
    the safest default when in doubt.
  * "aps" (Romano et al. 2020): s(x, y) = cumulative softmax mass of every
    class ranked at or above y. Intended to degrade more gracefully than
    "score" under distribution shift, since it accounts for the whole
    ranking rather than only the true class's isolated probability.

CAVEAT -- read before trusting "aps" coverage numbers: the reference APS
construction *randomizes* whether the class that crosses the cumulative
threshold is included, which is what makes its marginal coverage exactly
1 - alpha rather than merely "at least" 1 - alpha (Romano et al., Algorithm
1, the "U" term). This implementation always includes that boundary class
(deterministic), which is conservative -- "aps" prediction sets here are
very slightly larger, and empirical coverage very slightly higher, than
the paper's exact construction. This is a deliberate, documented corner
cut (see .docs/NEW_UQ_METHODS.md); a slightly conservative deterministic
score is far easier to get right under time pressure than a correctly
-seeded randomized one, and conservative-coverage-by-construction is a
much safer failure mode than under-coverage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def score_nonconformity(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """s(x, y) = 1 - p_y. probs: [N, C] (already softmax'd), labels: [N] -> [N]."""
    py = probs.gather(1, labels.view(-1, 1).long()).squeeze(1)
    return 1.0 - py


def aps_nonconformity(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    s(x, y) = sum of probabilities of every class ranked >= y (Romano et al.
    2020, eq. 4, non-randomized -- see module docstring). Sort classes by
    probability descending, take the cumulative sum, and read off the
    cumulative mass at y's rank.
    """
    order = probs.argsort(dim=-1, descending=True)                # [N,C]
    sorted_probs = torch.gather(probs, 1, order)                  # [N,C]
    cumsum = sorted_probs.cumsum(dim=-1)                          # [N,C]
    rank = (order == labels.view(-1, 1)).float().argmax(dim=-1)   # [N], rank of true label
    return cumsum.gather(1, rank.view(-1, 1)).squeeze(1)


def fit_conformal_threshold(cal_scores: torch.Tensor, alpha: float = 0.1) -> float:
    """
    Standard split-conformal threshold: the
    ceil((n+1)(1-alpha))/n empirical quantile of the calibration
    nonconformity scores (Angelopoulos & Bates 2021, Algorithm 1). Using
    this finite-sample-corrected quantile rather than the plain (1-alpha)
    quantile is what gives the 1 - alpha marginal coverage guarantee
    exactly (in expectation over calibration draws), not just
    asymptotically as n -> infinity.
    """
    n = cal_scores.shape[0]
    q_level = min(1.0, float(np.ceil((n + 1) * (1 - alpha))) / n)
    return float(torch.quantile(cal_scores.float(), q_level).item())


def predict_sets(probs: torch.Tensor, qhat: float, method: str = "score") -> torch.Tensor:
    """
    Boolean [N, C] membership matrix: `membership[i, c]` is True iff class c
    is in example i's prediction set. `method` must match whatever
    nonconformity score `qhat` was calibrated with.
    """
    if method == "score":
        scores = 1.0 - probs                                      # [N,C], per-class s(x,c)
        return scores <= qhat

    if method == "aps":
        order = probs.argsort(dim=-1, descending=True)             # [N,C]
        sorted_probs = torch.gather(probs, 1, order)
        cumsum = sorted_probs.cumsum(dim=-1)
        included_sorted = cumsum <= qhat
        # Deterministically include the boundary class that first pushes the
        # cumulative mass past qhat (see module docstring: conservative
        # relative to Romano et al.'s randomized version), and always
        # include the single most likely class so no example gets an empty
        # set.
        first_over = (~included_sorted).float().argmax(dim=-1)
        idx = torch.arange(probs.shape[-1], device=probs.device).unsqueeze(0)
        included_sorted = included_sorted | (idx == first_over.unsqueeze(1))
        included_sorted[:, 0] = True
        membership = torch.zeros_like(included_sorted)
        membership.scatter_(1, order, included_sorted)
        return membership

    raise ValueError(f"Unknown conformal method: {method!r}")


@dataclass
class ConformalSummary:
    """Aggregate numbers that matter for conformal prediction -- analogous
    to how ECE is *the* number for calibration methods elsewhere in this
    repo. Per-example set size/membership belongs in the output CSV;
    empirical coverage and mean set size are the two summary statistics
    worth logging/reporting alongside it."""
    qhat: float
    alpha: float
    method: str
    n_calibration: int
    n_test: int
    empirical_coverage: float
    mean_set_size: float
