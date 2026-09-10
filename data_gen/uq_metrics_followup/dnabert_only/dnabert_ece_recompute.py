"""
Fixed ECE + NLL + Brier for DNABERT2 methods only, on the two new datasets
(aleatoric_boundary, promoter_alisim dense grid) -- no old label_noise/
OOD-severity datasets, per request.

Key trick: no per-example softmax probability was saved for these methods
(the prior audit found only 2/~30 method x axis cells retained it). But for
binary classification, normalized total entropy U_total is an exact,
monotonic bijection of the predicted-class confidence on [0.5, 1] --
knowing U_total (and which class was predicted) exactly determines the
model's confidence in that prediction. This is a real mathematical
inversion, not an approximation or fabrication: entropy H(p) for a Bernoulli
is strictly decreasing in p over [0.5, 1], so given a target entropy value
there is exactly one p in that range solving it, found here by bisection.

DNABERT2 methods covered: mc_dropout, conv_epinet, laplace, ensemble_k5,
ensemble_k3 -- all five use the standard BALD decomposition where
U_total = U_aleatoric + U_epistemic exactly (verified: max sum = 1.000000
for all five on the AliSim file). `evidential` is EXCLUDED here: its
Dirichlet-based U_aleatoric/vacuity do not sum to a normalized entropy, and
recovering per-example confidence from its saved (vacuity, dirichlet_strength)
columns alone is not a clean invertible relationship the way binary entropy
is (see chat) -- would need per-class Dirichlet alphas, not saved. Flagged
as a real, honestly-reported gap, not filled in.

`cnn_mc_dropout`, `cnn_ensemble`, `rf_kmer` are excluded entirely: not
DNABERT2, out of scope for this request.
"""
import numpy as np
import pandas as pd

DNABERT_METHODS = ["mc_dropout", "conv_epinet", "laplace", "ensemble_k5", "ensemble_k3"]


def invert_binary_entropy(u_total: np.ndarray) -> np.ndarray:
    """u_total = H_nat(p) / ln(2), p in [0.5, 1]. Returns p via bisection."""
    u_total = np.clip(u_total, 1e-12, 1.0)
    lo = np.full_like(u_total, 0.5)
    hi = np.full_like(u_total, 1.0 - 1e-12)
    for _ in range(60):
        mid = (lo + hi) / 2
        h = -(mid * np.log(mid) + (1 - mid) * np.log(1 - mid)) / np.log(2)
        # h is decreasing in mid over [0.5,1]: h(0.5)=1, h(1)=0
        go_right = h > u_total  # need larger mid to decrease h further... wait h decreasing so larger mid -> smaller h
        lo = np.where(h > u_total, mid, lo)
        hi = np.where(h > u_total, hi, mid)
    return (lo + hi) / 2


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
        bin_conf = conf[idx].mean()
        bin_acc = correct[idx].mean()
        ece += (len(idx) / n) * abs(bin_conf - bin_acc)
    return ece


def nll_brier(p1, label):
    # p1 = model's probability of class 1
    eps = 1e-12
    p1c = np.clip(p1, eps, 1 - eps)
    p_true = np.where(label == 1, p1c, 1 - p1c)
    nll = -np.log(p_true).mean()
    brier = ((p1 - label) ** 2).mean()
    return nll, brier


def summarize(df, method_col, u_total_col, pred_col, label_col, dataset_name):
    rows = []
    for m in DNABERT_METHODS:
        sub = df[df[method_col] == m]
        if len(sub) == 0:
            continue
        u_total = sub[u_total_col].values.astype(float)
        pred = sub[pred_col].values.astype(int)
        label = sub[label_col].values.astype(int)
        conf = invert_binary_entropy(u_total)  # confidence in predicted class
        correct = (pred == label).astype(int)
        p1 = np.where(pred == 1, conf, 1 - conf)
        nll, brier = nll_brier(p1, label)
        row = {
            "dataset": dataset_name, "method": m, "n": len(sub),
            "accuracy": correct.mean(),
            "nll": nll, "brier": brier,
        }
        for b in (10, 15, 20, 25):
            row[f"ece_{b}bin"] = ece_binned(conf, correct, b, adaptive=False)
        row["ace_15bin_equalmass"] = ece_binned(conf, correct, 15, adaptive=True)
        rows.append(row)
    return pd.DataFrame(rows)


boundary = pd.read_csv(
    "/scratch/home/glh52/glm-epinet-pyt/.claude/worktrees/fix-epinet-batch-z/"
    "data_gen/aleatoric_boundary/csv_data/uncertainty_by_method.csv"
)
alisim = pd.read_csv(
    "/scratch/home/glh52/glm-epinet-pyt/.claude/worktrees/fix-epinet-batch-z/"
    "data_gen/promoter_alisim/uncertainty_eval/dense_grid/per_example_uncertainty.csv"
)
alisim["u_total"] = alisim["U_aleatoric"] + alisim["U_epistemic"]

res_boundary = summarize(boundary, "method", "U_total", "pred", "label", "aleatoric_boundary")
res_alisim = summarize(alisim, "method", "u_total", "pred", "labels", "promoter_alisim_dense")

out = pd.concat([res_boundary, res_alisim], ignore_index=True)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
out.to_csv("/home/glh52/.claude/jobs/80f6ac1e/tmp/dnabert_ece_nll_brier.csv", index=False)
print("\nExcluded: evidential (Dirichlet aleatoric/vacuity don't invert to a single confidence "
      "the way binary entropy does -- see script docstring), cnn_mc_dropout/cnn_ensemble/rf_kmer "
      "(not DNABERT2).")
