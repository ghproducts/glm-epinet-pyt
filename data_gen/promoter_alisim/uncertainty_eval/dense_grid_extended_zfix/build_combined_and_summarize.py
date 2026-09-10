#!/usr/bin/env python3
"""Builds the full 19-point-grid, 9-decomposable-method `per_example_uncertainty.csv`
for the promoter_alisim dense+extended grid, and recomputes dose-response /
Spearman / Mann-Whitney / shape-characterization tables over it.

Assembly:
  - conv_epinet: `per_example_conv_epinet_zfix_19pt.csv` (this dir) -- FULL
    19-point re-run under the new epinet_zfix checkpoint.
  - mc_dropout, evidential, laplace, ensemble_k5, ensemble_k3, cnn_mc_dropout,
    cnn_ensemble, rf_kmer: the ORIGINAL 16-point rows from
    `../dense_grid/per_example_uncertainty.csv` (unaffected by the epinet fix)
    + the 3 NEW branch-length rows from `per_example_other_methods_extended_only.csv`
    (this dir).

Output (this dir): `per_example_uncertainty_19pt.csv`,
`results_summary_dose_response.csv`, `results_summary_spearman.csv`,
`results_summary_mwu_endpoints.csv`, `results_summary_shape.csv`,
`results_summary.md`.
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DENSE_GRID_DIR = os.path.join(os.path.dirname(_THIS_DIR), "dense_grid")
if _DENSE_GRID_DIR not in sys.path:
    sys.path.insert(0, _DENSE_GRID_DIR)

import pandas as pd

from run_uncertainty_eval_dense import summarize, characterize_shape, ENDPOINT_LO  # noqa: E402

OUT_DIR = _THIS_DIR
ENDPOINT_HI_NEW = 12.0  # extended grid's new top end, replaces the old 3.0 endpoint for MWU

OTHER_METHODS = ["mc_dropout", "evidential", "laplace", "ensemble_k5", "ensemble_k3",
                  "cnn_mc_dropout", "cnn_ensemble", "rf_kmer"]
METHOD_ORDER = ["conv_epinet", "mc_dropout", "evidential", "laplace", "ensemble_k5", "ensemble_k3",
                "cnn_mc_dropout", "cnn_ensemble", "rf_kmer"]


def summarize_19pt(df: pd.DataFrame):
    """Like run_uncertainty_eval_dense.summarize(), but the MWU endpoint
    comparison uses the NEW grid endpoint (12.0) instead of the old (3.0),
    since the grid has been extended."""
    from scipy.stats import mannwhitneyu, spearmanr
    dose_rows, spear_rows, mwu_rows = [], [], []
    for method, sub in df.groupby("method"):
        for bl, g in sub.groupby("branch_length"):
            dose_rows.append({
                "method": method, "branch_length": bl, "n": len(g),
                "U_epistemic_mean": g["U_epistemic"].mean(), "U_epistemic_std": g["U_epistemic"].std(),
                "U_aleatoric_mean": g["U_aleatoric"].mean(), "U_aleatoric_std": g["U_aleatoric"].std(),
                "accuracy": (g["pred"] == g["labels"]).mean(),
            })
        for score in ["U_epistemic", "U_aleatoric"]:
            rho, p = spearmanr(sub["branch_length"], sub[score])
            spear_rows.append({"method": method, "score": score, "spearman_rho": rho, "spearman_p": p, "n": len(sub)})
        lo = sub[sub["branch_length"] == ENDPOINT_LO]
        hi = sub[sub["branch_length"] == ENDPOINT_HI_NEW]
        for score in ["U_epistemic", "U_aleatoric"]:
            u, p = mannwhitneyu(hi[score], lo[score], alternative="greater")
            mwu_rows.append({
                "method": method, "score": score, "comparison": f"t={ENDPOINT_HI_NEW} > t={ENDPOINT_LO}",
                "U": u, "p": p, "n_lo": len(lo), "n_hi": len(hi),
                "mean_lo": lo[score].mean(), "mean_hi": hi[score].mean(),
            })
    dose_df = pd.DataFrame(dose_rows).sort_values(["method", "branch_length"]).reset_index(drop=True)
    spear_df = pd.DataFrame(spear_rows).reset_index(drop=True)
    mwu_df = pd.DataFrame(mwu_rows).reset_index(drop=True)
    return dose_df, spear_df, mwu_df


def write_markdown_summary(dose_df, spear_df, mwu_df, shape_df, path, methods_all):
    lines = [f"# promoter_alisim EXTENDED (19-point) dense-grid uncertainty evaluation: results summary "
             f"({len(methods_all)} configurations)", ""]
    lines.append(
        "Full 19-point branch-length grid (16-point dense grid "
        "{0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0} "
        "+ 3-point extension {5.0, 8.0, 12.0}). `conv_epinet` is a FULL re-run under the newly "
        "retrained `epinet_zfix` checkpoint (trained under the per-example-z fix from the start, "
        "not just patched at eval time -- see README.md's retrain section). The other 8 "
        "method-configurations are unchanged at the original 16 points and newly computed only at "
        "the 3 extension points, since none of them touch nn_proj.models.epinet."
    )
    lines.append("")
    lines.append(f"Methods covered ({len(methods_all)} configurations): " + ", ".join(methods_all))
    lines.append("")
    lines.append("## Dose-response: mean U_epistemic / U_aleatoric / accuracy by branch_length")
    lines.append("")
    for method, sub in dose_df.groupby("method", sort=False):
        lines.append(f"### {method}")
        lines.append("")
        cols = ["branch_length", "n", "U_epistemic_mean", "U_epistemic_std", "U_aleatoric_mean", "U_aleatoric_std", "accuracy"]
        lines.append(sub[cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("## Spearman correlation: branch_length vs. score (per method, full 19-point grid)")
    lines.append("")
    lines.append(spear_df.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    lines.append(f"## Endpoint check: Mann-Whitney U, t={ENDPOINT_HI_NEW} > t={ENDPOINT_LO} (per method, per score)")
    lines.append("")
    lines.append(mwu_df.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    lines.append("## Shape characterization: where does U_epistemic peak, and what happens after? "
                  "(peak location + peak-to-baseline / peak-to-endpoint magnitude, per method)")
    lines.append("")
    lines.append(shape_df.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def main():
    epi = pd.read_csv(os.path.join(OUT_DIR, "per_example_conv_epinet_zfix_19pt.csv"))
    orig16 = pd.read_csv(os.path.join(_DENSE_GRID_DIR, "per_example_uncertainty.csv"))
    orig16_others = orig16[orig16["method"].isin(OTHER_METHODS)]
    ext_others = pd.read_csv(os.path.join(OUT_DIR, "per_example_other_methods_extended_only.csv"))

    combined = pd.concat([epi, orig16_others, ext_others], ignore_index=True)
    # The original dense_grid per_example_uncertainty.csv's DNABERT2-method
    # rows carry branch_length through a float32 round-trip somewhere in the
    # HF Arrow dataset pipeline (metadata extraction in
    # run_uncertainty_eval_dense.py), so e.g. 0.01 appears as both 0.01 and
    # 0.0099999997764825 depending on which method computed it -- a
    # pre-existing artifact of the ORIGINAL dense grid, not introduced here
    # (visible already in dense_grid/per_example_uncertainty.csv). Round to
    # collapse these before grouping so every method's rows land on exactly
    # the same 19 nominal branch-length values.
    combined["branch_length"] = combined["branch_length"].round(6)
    combined_path = os.path.join(OUT_DIR, "per_example_uncertainty_19pt.csv")
    combined.to_csv(combined_path, index=False)
    n_bl = combined["branch_length"].nunique()
    print(f"wrote {combined_path} ({len(combined)} rows, {combined['method'].nunique()} methods, "
          f"{n_bl} distinct branch lengths)")
    assert n_bl == 19, f"expected 19 distinct branch lengths, got {n_bl}: {sorted(combined['branch_length'].unique())}"

    dose_df, spear_df, mwu_df = summarize_19pt(combined)
    shape_df = characterize_shape(dose_df)

    dose_df.to_csv(os.path.join(OUT_DIR, "results_summary_dose_response.csv"), index=False)
    spear_df.to_csv(os.path.join(OUT_DIR, "results_summary_spearman.csv"), index=False)
    mwu_df.to_csv(os.path.join(OUT_DIR, "results_summary_mwu_endpoints.csv"), index=False)
    shape_df.to_csv(os.path.join(OUT_DIR, "results_summary_shape.csv"), index=False)

    methods_all = [m for m in METHOD_ORDER if m in combined["method"].unique()]
    write_markdown_summary(dose_df, spear_df, mwu_df, shape_df,
                            os.path.join(OUT_DIR, "results_summary.md"), methods_all)

    print("\n=== SHAPE CHARACTERIZATION (19-point grid, all 9 decomposable methods) ===")
    print(shape_df.to_string(index=False))
    print("\n=== SPEARMAN (19-point grid) ===")
    print(spear_df.to_string(index=False))


if __name__ == "__main__":
    main()
