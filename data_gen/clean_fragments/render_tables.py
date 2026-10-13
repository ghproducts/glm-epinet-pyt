#!/usr/bin/env python3
"""Render results/clean1500/full_*.csv as markdown tables -> results/clean1500/full_tables.md."""
import pandas as pd
import sys
R = sys.argv[1] if len(sys.argv) > 1 else "results/clean1500_es"
BB = {"NT_transformer": "NT", "DNABERT2": "DNABERT2", "hyenaDNA": "hyenaDNA", "CARMANIA": "CARMANIA"}
SC = ["msp", "energy", "entropy", "mc_entropy", "mc_epistemic", "epi_entropy", "epi_epistemic", "maha", "knn"]
SN = {"msp": "MSP", "energy": "energy", "entropy": "entropy", "mc_entropy": "MC entropy", "mc_epistemic": "MC epistemic",
      "epi_entropy": "Epinet entropy", "epi_epistemic": "Epinet epistemic", "maha": "Mahalanobis", "knn": "kNN"}
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
                for m in ["softmax", "temperature", "mc_dropout", "epinet"]:
                    row[f"{m} acc"] = f"{x.loc[m, 'accuracy_mean']:.3f}"
                for m in ["softmax", "temperature", "mc_dropout", "epinet"]:
                    row[f"{m} ECE"] = f"{x.loc[m, 'ece_mean']:.3f}"
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
                    rows.append({"tier": t, "backbone": bn, **{SN[s]: f"{x.loc[s, metric + '_mean']:.3f}" for s in SC}})
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
open(f"{R}/full_tables.md", "w").write("\n\n".join(out) + "\n")
print("\n\n".join(out))
