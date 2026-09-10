"""Alternative to `compute_uncertainty`'s entropy/mutual-information (BALD)
decomposition: a variance-based split via the law of total variance, which
holds as an exact algebraic identity (no approximation, none of the
axiom violations Wimmer et al. 2023 proved for mutual information --
not maximal under complete ignorance, not monotone under mean-preserving
spreads, not invariant under location shifts).

For a [K, B, C] stack of per-sample softmax probabilities p_k (one sample
per posterior/index/dropout draw k), treat each class c's one-hot indicator
y_c as a Bernoulli(p_k,c) draw conditional on sample k. The law of total
variance,

    Var(y_c) = E_k[Var(y_c | k)] + Var_k[E(y_c | k)]
             = E_k[p_k,c (1 - p_k,c)]  +  Var_k[p_k,c]
             = aleatoric_c            +  epistemic_c

holds exactly for every class c and sums, marginalizing over k, to
Var(y_c) = p_bar_c (1 - p_bar_c) where p_bar is the mean prediction --
i.e. total_c is exactly the per-class term of the Gini impurity of the mean
prediction. Summing over classes gives a single scalar per example:

    total     = sum_c p_bar_c (1 - p_bar_c)      = Gini impurity of p_bar
    aleatoric = sum_c mean_k[p_k,c (1 - p_k,c)]  = mean Gini impurity of the K samples
    epistemic = sum_c Var_k[p_k,c]               = spread of the K samples' per-class probabilities

and total == aleatoric + epistemic exactly (checked in the self-test below),
unlike BALD's entropy/MI split, which is only an identity for entropy
specifically and has no such variance-based analogue guarantee.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_uncertainty_variance(epi_out_logits: torch.Tensor) -> dict:
    """
    Accepts [K, B, C] (preferred) or [B, C]. Returns a dict of [B]-shaped
    tensors, mirroring `nn_proj.common.utils.compute_uncertainty`'s
    interface/key names so it's a drop-in alternative for comparison.
    """
    if epi_out_logits.dim() == 2:
        epi_out_logits = epi_out_logits.unsqueeze(0)
    S, B, C = epi_out_logits.shape

    p = torch.softmax(epi_out_logits, dim=-1)          # [S,B,C]
    p_bar = p.mean(dim=0)                               # [B,C]

    total = (p_bar * (1.0 - p_bar)).sum(dim=-1)                       # [B], Gini of mean pred
    aleatoric = (p * (1.0 - p)).mean(dim=0).sum(dim=-1)               # [B], mean per-sample Gini
    epistemic = p.var(dim=0, unbiased=False).sum(dim=-1)              # [B], spread across samples

    # Normalize by the max possible Gini impurity for C classes, (C-1)/C,
    # to match compute_uncertainty's normalized_* convention.
    norm = (C - 1) / C

    return {
        "predicted_class": p_bar.argmax(dim=-1),
        "normalized_total_uncertainty": total / norm,
        "normalized_epistemic_uncertainty": epistemic / norm,
        "normalized_aleatoric_uncertainty": aleatoric / norm,
        "max_confidence": p_bar.max(dim=-1).values,
    }


def _self_test():
    torch.manual_seed(0)
    logits = torch.randn(16, 5, 3) * 2
    unc = compute_uncertainty_variance(logits)
    p = torch.softmax(logits, dim=-1)
    p_bar = p.mean(dim=0)
    total_raw = (p_bar * (1 - p_bar)).sum(-1)
    ale_raw = (p * (1 - p)).mean(0).sum(-1)
    epi_raw = p.var(dim=0, unbiased=False).sum(-1)
    assert torch.allclose(total_raw, ale_raw + epi_raw, atol=1e-6), "law of total variance identity failed"
    print("self-test passed: total == aleatoric + epistemic exactly, max diff",
          (total_raw - (ale_raw + epi_raw)).abs().max().item())


if __name__ == "__main__":
    _self_test()
