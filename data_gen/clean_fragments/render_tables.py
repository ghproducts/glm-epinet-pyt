#!/usr/bin/env python3
"""Render results/clean1500/full_*.csv as markdown tables -> results/clean1500/full_tables.md."""
import pandas as pd
import sys
R = sys.argv[1] if len(sys.argv) > 1 else "results/clean1500_es"
BB = {"NT_transformer": "NT", "DNABERT2": "DNABERT2", "hyenaDNA": "hyenaDNA", "CARMANIA": "CARMANIA",
      "CARMANIA-prok": "CARMANIA (multi-species)"}
METHODS = ["softmax", "temperature", "mc_dropout", "epinet", "laplace"]
SC = ["msp", "energy", "entropy", "mc_entropy", "mc_epistemic", "epi_entropy", "epi_epistemic",
      "lap_entropy", "lap_epistemic", "cp_size", "maha", "knn"]
SN = {"msp": "MSP", "energy": "energy", "entropy": "entropy", "mc_entropy": "MC entropy", "mc_epistemic": "MC epistemic",
      "epi_entropy": "Epinet entropy", "epi_epistemic": "Epinet epistemic", "lap_entropy": "Laplace entropy",
      "lap_epistemic": "Laplace epistemic", "cp_size": "conformal set size", "maha": "Mahalanobis", "knn": "kNN"}
out = []
def md(df):   # tabulate isn't installed
    h = list(df.columns)
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] + ["| " + " | ".join(map(str, r)) + " |" for r in df.values])
C = pd.read_csv(f"{R}/full_calibration.csv", dtype={"error": str})
for r in [x for x in ["phylum", "family", "gene_taxa"] if x in set(C["rank"])]:
    sets = {"phylum": ["id_test", "novel_genus", "novel_family"], "family": ["id_test", "novel_genus"], "gene_taxa": ["id_test", "taxa_out"]}[r]
    rows = []
    for s in sets:
        for e in ["0"]:
            for b, bn in BB.items():
                x = C[(C["rank"] == r) & (C.set == s) & (C.error == e) & (C.backbone == b)].set_index("method")
                if x.empty: continue
                row = {"set": s, "error": e + "%", "backbone": bn}
                for m in METHODS:
                    row[f"{m} acc"] = f"{x.loc[m, 'accuracy_mean']:.3f}" if m in x.index else "-"
                for m in METHODS:
                    row[f"{m} ECE"] = f"{x.loc[m, 'ece_mean']:.3f}" if m in x.index else "-"
                rows.append(row)
    out.append(f"### Calibration, {r} models (in-vocabulary fragments; mean over seeds)\n\n" + md(pd.DataFrame(rows)))
D = pd.read_csv(f"{R}/full_detection.csv", dtype={"error": str})
for r in [x for x in ["phylum", "family", "gene_taxa"] if x in set(D["rank"])]:
    for metric, nm in [("auroc", "AUROC"), ("fpr95", "FPR@95%TPR (lower is better)")]:
        for e in ["0"]:
            rows = []
            for t in (["taxa_out", "gene_out"] if r == "gene_taxa" else ["novel_genus", "novel_family", "nonbacterial"]):
                for b, bn in BB.items():
                    x = D[(D["rank"] == r) & (D.error == e) & (D.tier == t) & (D.backbone == b)].set_index("score")
                    if x.empty: continue
                    rows.append({"tier": t, "backbone": bn, **{SN[s]: (f"{x.loc[s, metric + '_mean']:.3f}" if s in x.index else "-") for s in SC}})
            out.append(f"### Detection {nm}, {r} models, {e}% error (id_test vs tier at the same error; mean over seeds)\n\n" + md(pd.DataFrame(rows)))
M = pd.read_csv(f"{R}/full_darkmatter.csv")
for r in ["family", "phylum"]:
    for f in [0.01, 0.05, 0.10]:
        rows = []
        for b, bn in BB.items():
            for s in SC:
                x = M[(M["rank"] == r) & (M.backbone == b) & (M.score == s) & (M.fpr == f)]
                if x.empty: continue
                x = x.iloc[0]
                rows.append({"backbone": bn, "score": SN[s],
                             "flagged UP .25 / .75 / 1.0": f"{x.flag_up0_25_mean:.3f} / {x.flag_up0_75_mean:.3f} / {x.flag_up1_0_mean:.3f}",
                             "slope": f"{x.slope_mean:+.4f}",
                             "p": f"{x.p_median:.3g}", "increasing": "yes" if x.n_increasing else "no",
                             "AUROC id vs UP .25 / .75 / 1.0": f"{x['auroc_up0.25_mean']:.3f} / {x['auroc_up0.75_mean']:.3f} / {x['auroc_up1.0_mean']:.3f}"})
        out.append(f"### Dark matter, {r} models, {f:.0%} FPR (mean over seeds)\n\n" + md(pd.DataFrame(rows)))
import os
if os.path.exists(f"{R}/full_conformal.csv"):
    F = pd.read_csv(f"{R}/full_conformal.csv")
    for r in [x for x in ["phylum", "family", "gene_taxa"] if x in set(F["rank"])]:
        rows = []
        for s in [x for x in ["id_test", "novel_genus", "novel_family", "nonbacterial", "taxa_out", "gene_out"] if x in set(F[F["rank"] == r].set)]:
            for b, bn in BB.items():
                x = F[(F["rank"] == r) & (F.set == s) & (F.backbone == b)]
                if x.empty: continue
                x = x.iloc[0]
                rows.append({"set": s, "backbone": bn, "coverage (target 0.90)": "-" if pd.isna(x.coverage) else f"{x.coverage:.3f}",
                             "mean set size": f"{x.mean_set_size:.2f}", "singleton sets": f"{x.frac_singleton:.3f}", "empty sets": f"{x.frac_empty:.3f}"})
        out.append(f"### Split conformal (alpha 0.1, calibrated on val), {r} models\n\n" + md(pd.DataFrame(rows)))
open(f"{R}/full_tables.md", "w").write("\n\n".join(out) + "\n")
print("\n\n".join(out))
