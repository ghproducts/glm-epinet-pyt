#!/usr/bin/env python3
"""Analysis 1: total vs. epistemic-alone detection power, every method, every axis.

BALD decomposition identity: U_total = U_aleatoric + U_epistemic exactly,
per-example, by construction. This script performs pure aggregation on
already-computed per-example (or, where only that survives, per-seed/
per-rate aggregate) data -- no new model inference.

Axes:
  A. label-noise rate      (data_gen/label_noise/decomp_compare_label_noise*.csv)
  B. OOD severity          (data_gen/label_noise/{all_methods_ood_*,decomp_compare_ood_*}.csv)
  C. margin-boundary quartile (data_gen/aleatoric_boundary/csv_data/*.csv, per-example)
  D. AliSim branch-length  (data_gen/promoter_alisim/.../dense_grid/per_example_uncertainty.csv, per-example)

Output: analysis1_total_vs_epistemic.csv, one row per (axis, method, score in
{U_total, U_epistemic, U_aleatoric}), with the test statistic used (spearman
rho+p, or Mann-Whitney U+p for discrete 2/3-level axes), sample size, and a
`total_vs_epistemic` verdict column filled in only where both scores were
independently testable.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "inputs")
LN = os.path.join(IN, "label_noise")
AB = os.path.join(IN, "aleatoric_boundary")
AS_D = os.path.join(IN, "promoter_alisim", "dense_grid")

rows = []  # each: axis, method, score, stat_type, stat_value, p, n, note


def add(axis, method, score, stat_type, stat_value, p, n, note=""):
    rows.append(dict(axis=axis, method=method, score=score, stat_type=stat_type,
                      stat_value=stat_value, p=p, n=n, note=note))


# ---------------------------------------------------------------------------
# AXIS A: label-noise rate (ordinal, 5 levels: 0.00,0.05,0.10,0.20,0.40)
# Only aggregate (per-rate[-per-seed]) means survive for this axis -- no
# per-example CSVs were ever saved for the label-noise-rate scripts. Spearman
# is computed across the available (rate, mean-per-seed-or-per-rate) points;
# this is legitimately a much smaller-n correlation than the per-example
# axes below (n = #rates x #seeds, not #examples), and is reported as such.
# ---------------------------------------------------------------------------
RATE_MAP = {"r00": 0.00, "r05": 0.05, "r10": 0.10, "r20": 0.20, "r40": 0.40}


def spearman_axisA(df, rate_col, cols_map, method_name, n_points_note):
    d = df.copy()
    d["rate_num"] = d[rate_col].map(RATE_MAP)
    for score_name, col in cols_map.items():
        if col is None or col not in d.columns:
            continue
        sub = d[["rate_num", col]].dropna()
        if len(sub) < 3:
            continue
        rho, p = spearmanr(sub["rate_num"], sub[col])
        add("label_noise_rate", method_name, score_name, "spearman", rho, p,
            len(sub), n_points_note)


# conv_epinet: has bald_U_total_mean directly + epistemic + aleatoric, 3 seeds x 5 rates = 15 pts
df = pd.read_csv(os.path.join(LN, "decomp_compare_label_noise.csv"))
spearman_axisA(df, "rate", {
    "U_total": "bald_U_total_mean",
    "U_epistemic": "bald_U_epistemic_mean",
    "U_aleatoric": "bald_U_aleatoric_mean",
}, "conv_epinet", "n=rate x seed points (3 seeds x 5 rates)")

# mc_dropout: 3 seeds x 5 rates, U_total derived as sum (BALD linearity)
df = pd.read_csv(os.path.join(LN, "decomp_compare_label_noise_mcdropout.csv"))
df["bald_U_total_mean"] = df["bald_U_epistemic_mean"] + df["bald_U_aleatoric_mean"]
spearman_axisA(df, "rate", {
    "U_total": "bald_U_total_mean",
    "U_epistemic": "bald_U_epistemic_mean",
    "U_aleatoric": "bald_U_aleatoric_mean",
}, "mc_dropout", "n=rate x seed points (3 seeds x 5 rates); U_total = sum of means (exact, BALD linearity)")

# laplace: 3 seeds x 5 rates
df = pd.read_csv(os.path.join(LN, "decomp_compare_label_noise_laplace.csv"))
df["bald_U_total_mean"] = df["bald_U_epistemic_mean"] + df["bald_U_aleatoric_mean"]
spearman_axisA(df, "rate", {
    "U_total": "bald_U_total_mean",
    "U_epistemic": "bald_U_epistemic_mean",
    "U_aleatoric": "bald_U_aleatoric_mean",
}, "laplace", "n=rate x seed points (3 seeds x 5 rates); U_total = sum of means")

# ensemble_k5: 1 row per rate (5 pts) -- Spearman on n=5 is very low powered, reported plainly
df = pd.read_csv(os.path.join(LN, "decomp_compare_label_noise_ensemble.csv"))
df["bald_U_total_mean"] = df["bald_U_epistemic_mean"] + df["bald_U_aleatoric_mean"]
spearman_axisA(df, "rate", {
    "U_total": "bald_U_total_mean",
    "U_epistemic": "bald_U_epistemic_mean",
    "U_aleatoric": "bald_U_aleatoric_mean",
}, "ensemble_k5", "n=5 rate points only, no seed replication -- low-powered correlation")

# ensemble_k3
df = pd.read_csv(os.path.join(LN, "decomp_compare_label_noise_ensemble_k3.csv"))
df["bald_U_total_mean"] = df["bald_U_epistemic_mean"] + df["bald_U_aleatoric_mean"]
spearman_axisA(df, "rate", {
    "U_total": "bald_U_total_mean",
    "U_epistemic": "bald_U_epistemic_mean",
    "U_aleatoric": "bald_U_aleatoric_mean",
}, "ensemble_k3", "n=5 rate points only, no seed replication")

# evidential: only vacuity (epistemic-analogue) + U_aleatoric, 5 pts
df = pd.read_csv(os.path.join(LN, "decomp_compare_label_noise_evidential.csv"))
df["U_total_proxy"] = df["vacuity_mean"] + df["U_aleatoric_mean"]
spearman_axisA(df, "rate", {
    "U_total_proxy(vacuity+aleatoric)": "U_total_proxy",
    "vacuity(epistemic-analogue)": "vacuity_mean",
    "U_aleatoric": "U_aleatoric_mean",
}, "evidential", "n=5 rate points; evidential has no true BALD U_epistemic (K=1) -- vacuity used as its epistemic-analogue; U_total_proxy = vacuity + U_aleatoric is NOT the BALD identity, just an ad hoc sum for comparability")

# rf: 5 pts
df = pd.read_csv(os.path.join(LN, "decomp_compare_label_noise_rf.csv"))
df["bald_U_total_mean"] = df["bald_U_epistemic_mean"] + df["bald_U_aleatoric_mean"]
spearman_axisA(df, "rate", {
    "U_total": "bald_U_total_mean",
    "U_epistemic": "bald_U_epistemic_mean",
    "U_aleatoric": "bald_U_aleatoric_mean",
}, "rf_kmer", "n=5 rate points only")

# cnn: mc_dropout & ensemble sampling variants, 5 rates each, no seed replication
df = pd.read_csv(os.path.join(LN, "decomp_compare_label_noise_cnn.csv"))
df["bald_U_total_mean"] = df["bald_U_epistemic_mean"] + df["bald_U_aleatoric_mean"]
for samp, sub in df.groupby("sampling"):
    spearman_axisA(sub, "rate", {
        "U_total": "bald_U_total_mean",
        "U_epistemic": "bald_U_epistemic_mean",
        "U_aleatoric": "bald_U_aleatoric_mean",
    }, f"cnn_{samp}", "n=5 rate points only")


# ---------------------------------------------------------------------------
# AXIS B: OOD severity (3-level discrete: real / shuffled / random_dna)
# Methods with real per-example-derived MWU on U_total/primary score, already
# computed this session and reported in .docs/ALL_METHODS_OOD_SEVERITY.md /
# all_methods_ood_mwu.csv -- reused verbatim, not recomputed, per task
# instructions. Methods with only aggregate mean/std (the 5 extra methods)
# get U_total_mean via the exact BALD sum but NO MWU p-value (would require
# per-example data not persisted for those methods) -- reported as a gap.
# ---------------------------------------------------------------------------

# --- 7 core methods: real MWU already computed (ALL_METHODS_OOD_SEVERITY.md) ---
ood_core_total = {
    # method: (real, shuffled, p_shuf, random_dna, p_rand, score_label)
    "base":        (0.352, 0.397, 6.4e-4, 0.443, 1.9e-34, "U_total"),
    "base_scaled": (0.326, 0.368, 2.0e-3, 0.407, 8.6e-34, "U_total"),
    "mc_dropout":  (0.390, 0.401, 0.355,  0.401, 8.0e-11, "U_total"),
    "conv_epinet": (0.415, 0.464, 2.8e-5, 0.551, 1.7e-43, "U_total"),
    "laplace":     (0.352, 0.401, 5.6e-4, 0.448, 2.0e-36, "U_total"),
    "evidential":  (0.237, 0.245, 7.3e-3, 0.268, 1.4e-49, "vacuity(primary score)"),
    "conformal":   (1.037, 1.051, 0.064,  1.036, 0.531,   "set_size(primary score)"),
}
ood_core_epi = {
    "mc_dropout":  (0.0122, 0.0159, 4.2e-6, 0.0156, 2.7e-58),
    "conv_epinet": (0.0714, 0.0730, 1.9e-3, 0.0844, 8.7e-27),
    "laplace":     (0.00165, 0.00196, 5.2e-4, 0.00193, 1.4e-17),
}
for m, (real, shuf, p_shuf, rand, p_rand, label) in ood_core_total.items():
    add("ood_severity", m, label, "mwu(shuffled_vs_real)", None, p_shuf, 1584,
        f"mean(real)={real}, mean(shuffled)={shuf}; from .docs/ALL_METHODS_OOD_SEVERITY.md, reused verbatim")
    add("ood_severity", m, label, "mwu(random_dna_vs_real)", None, p_rand, 1584,
        f"mean(real)={real}, mean(random_dna)={rand}; from .docs/ALL_METHODS_OOD_SEVERITY.md, reused verbatim")
for m, (real, shuf, p_shuf, rand, p_rand) in ood_core_epi.items():
    add("ood_severity", m, "U_epistemic", "mwu(shuffled_vs_real)", None, p_shuf, 1584,
        f"mean(real)={real}, mean(shuffled)={shuf}; from .docs/ALL_METHODS_OOD_SEVERITY.md")
    add("ood_severity", m, "U_epistemic", "mwu(random_dna_vs_real)", None, p_rand, 1584,
        f"mean(real)={real}, mean(random_dna)={rand}; from .docs/ALL_METHODS_OOD_SEVERITY.md")
    # aleatoric MWU for these 3, from decomp_compare_ood_mwu.csv (mc_dropout/conv_epinet only) -- laplace's not separately in that file
mwu_bald = pd.read_csv(os.path.join(LN, "decomp_compare_ood_mwu.csv"))
res_bald = pd.read_csv(os.path.join(LN, "decomp_compare_ood_results.csv"))
for m in ["mc_dropout", "conv_epinet"]:
    for comp, var in [("shuffled > real", "shuffled"), ("random_dna > real", "random_dna")]:
        r = mwu_bald[(mwu_bald.method == m) & (mwu_bald.score == "bald_U_aleatoric") &
                     (mwu_bald.comparison == f"{m}/bald_U_aleatoric: {comp}")]
        if len(r):
            add("ood_severity", m, "U_aleatoric", f"mwu({var}_vs_real)", None, float(r.p.iloc[0]), 1584,
                "from decomp_compare_ood_mwu.csv")

# --- 5 extra methods: aggregate-only, U_total via BALD sum, no total MWU possible ---
extra_files = {
    "ensemble_k5": "decomp_compare_ood_results_ensemble.csv",
    "ensemble_k3": "decomp_compare_ood_results_ensemble_k3.csv",
    "rf_kmer": "decomp_compare_ood_results_rf.csv",
}
extra_mwu_files = {
    "ensemble_k5": "decomp_compare_ood_mwu_ensemble.csv",
    "ensemble_k3": "decomp_compare_ood_mwu_ensemble_k3.csv",
    "rf_kmer": "decomp_compare_ood_mwu_rf.csv",
}
for m, fname in extra_files.items():
    d = pd.read_csv(os.path.join(LN, fname))
    piv = d.pivot_table(index="variant", columns="score", values="mean")
    piv = piv.reindex(["real", "shuffled", "random_dna"])
    total = piv["bald_U_epistemic"] + piv["bald_U_aleatoric"]
    mwu = pd.read_csv(os.path.join(LN, extra_mwu_files[m]))
    for var in ["shuffled", "random_dna"]:
        add("ood_severity", m, "U_total", f"mwu({var}_vs_real)", None, np.nan, 1584,
            f"NO per-example U_total available -- mean(real)={piv.loc['real','bald_U_epistemic']+piv.loc['real','bald_U_aleatoric']:.4f}, "
            f"mean({var})={piv.loc[var,'bald_U_epistemic']+piv.loc[var,'bald_U_aleatoric']:.4f} (BALD sum of means, exact); "
            f"MWU not computable from aggregate mean/std alone -- GAP, would need per-example U_total")
        for score in ["bald_U_epistemic", "bald_U_aleatoric"]:
            label = "U_epistemic" if score == "bald_U_epistemic" else "U_aleatoric"
            row = mwu[(mwu.score == score) & (mwu.comparison.str.contains(f"{var} > real"))]
            if len(row):
                add("ood_severity", m, label, f"mwu({var}_vs_real)", None, float(row.p.iloc[0]), 1584,
                    f"from {extra_mwu_files[m]}")

# cnn methods
cnn_res = pd.read_csv(os.path.join(LN, "decomp_compare_ood_results_cnn.csv"))
cnn_mwu = pd.read_csv(os.path.join(LN, "decomp_compare_ood_mwu_cnn.csv"))
for m in ["cnn_mc_dropout", "cnn_ensemble"]:
    sub = cnn_res[cnn_res.method == m]
    piv = sub.pivot_table(index="variant", columns="score", values="mean").reindex(["real", "shuffled", "random_dna"])
    for var in ["shuffled", "random_dna"]:
        add("ood_severity", m, "U_total", f"mwu({var}_vs_real)", None, np.nan, 1584,
            f"NO per-example U_total available -- mean(real)={piv.loc['real','bald_U_epistemic']+piv.loc['real','bald_U_aleatoric']:.4f}, "
            f"mean({var})={piv.loc[var,'bald_U_epistemic']+piv.loc[var,'bald_U_aleatoric']:.4f} (BALD sum of means); "
            f"MWU not computable -- GAP")
        for score, label in [("bald_U_epistemic", "U_epistemic"), ("bald_U_aleatoric", "U_aleatoric")]:
            row = cnn_mwu[(cnn_mwu.method == m) & (cnn_mwu.score == score) &
                          (cnn_mwu.comparison.str.contains(f"{var} > real"))]
            if len(row):
                add("ood_severity", m, label, f"mwu({var}_vs_real)", None, float(row.p.iloc[0]), 1584,
                    "from decomp_compare_ood_mwu_cnn.csv")


# ---------------------------------------------------------------------------
# AXIS C: margin-boundary quartile -- full per-example data, all 9 methods.
# U_total already saved per-example; compute Spearman(margin, score) and
# Mann-Whitney (Q1 vs Q4) directly, exactly like the pre-existing
# epistemic/aleatoric numbers in results_summary.csv, so all three scores
# are on equal footing.
# ---------------------------------------------------------------------------
umeth = pd.read_csv(os.path.join(AB, "csv_data", "uncertainty_by_method.csv"))
for method, sub in umeth.groupby("method"):
    sub = sub.dropna(subset=["margin"])
    for score_col, label in [("U_total", "U_total"), ("U_epistemic", "U_epistemic"), ("U_aleatoric", "U_aleatoric")]:
        d = sub.dropna(subset=[score_col])
        if len(d) == 0:
            continue
        rho, p = spearmanr(d["margin"], d[score_col])
        add("margin_boundary", method, label, "spearman_vs_margin", rho, p, len(d), "")
        q1 = d[d.quartile == "Q1"][score_col]
        q4 = d[d.quartile == "Q4"][score_col]
        if len(q1) and len(q4):
            U, p_mwu = mannwhitneyu(q1, q4, alternative="two-sided")
            add("margin_boundary", method, label, "mwu(Q1_vs_Q4)", U, p_mwu, f"{len(q1)}+{len(q4)}",
                f"mean(Q1)={q1.mean():.4f}, mean(Q4)={q4.mean():.4f}")

# evidential's "vacuity" is its epistemic-analogue and is a separate column
evid = umeth[umeth.method == "evidential"].dropna(subset=["vacuity", "margin"])
if len(evid):
    rho, p = spearmanr(evid["margin"], evid["vacuity"])
    add("margin_boundary", "evidential", "vacuity(epistemic-analogue)", "spearman_vs_margin", rho, p, len(evid), "")
    q1 = evid[evid.quartile == "Q1"]["vacuity"]
    q4 = evid[evid.quartile == "Q4"]["vacuity"]
    U, p_mwu = mannwhitneyu(q1, q4, alternative="two-sided")
    add("margin_boundary", "evidential", "vacuity(epistemic-analogue)", "mwu(Q1_vs_Q4)", U, p_mwu,
        f"{len(q1)}+{len(q4)}", f"mean(Q1)={q1.mean():.4f}, mean(Q4)={q4.mean():.4f}")


# ---------------------------------------------------------------------------
# AXIS D: AliSim branch-length -- dense grid, full per-example data, 9 methods.
# U_total computed per-example as U_aleatoric + U_epistemic (exact). Branch
# lengths are snapped to the nominal 16-point grid to correct float32
# round-trip noise between the two scripts that wrote this combined csv.
# ---------------------------------------------------------------------------
GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]


def snap(x):
    return min(GRID, key=lambda g: abs(g - x))


pe = pd.read_csv(os.path.join(AS_D, "per_example_uncertainty.csv"))
pe["bl"] = pe["branch_length"].apply(snap)
pe["U_total"] = pe["U_aleatoric"] + pe["U_epistemic"]

for method, sub in pe.groupby("method"):
    for score_col in ["U_total", "U_epistemic", "U_aleatoric"]:
        d = sub.dropna(subset=[score_col])
        rho, p = spearmanr(d["bl"], d[score_col])
        add("alisim_branch_length", method, score_col, "spearman_vs_branch_length", rho, p, len(d), "")
        lo = d[d.bl == 0.0][score_col]
        hi = d[d.bl == 3.0][score_col]
        if len(lo) and len(hi):
            U, p_mwu = mannwhitneyu(hi, lo, alternative="greater")
            add("alisim_branch_length", method, score_col, "mwu(t=3.0_gt_t=0.0)", U, p_mwu,
                f"{len(lo)}+{len(hi)}", f"mean(t=0)={lo.mean():.4f}, mean(t=3.0)={hi.mean():.4f}")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "analysis1_total_vs_epistemic.csv"), index=False)
print(f"Wrote {len(out)} rows to analysis1_total_vs_epistemic.csv")
print(out["axis"].value_counts())
