#!/usr/bin/env python3
"""ZymoBIOMICS even community (ERR3152364) through the NT clean1500 models: does uncertainty find the reads the
model cannot know (two yeasts, ~4% of reads) and the bacteria from genera held out of training (E. coli,
L. fermentum)?

Inputs: results/clean1500_es/NT_transformer_<rank>_s1/scores_{zymo,zymo_oriented,id_test}.parquet
(eval_clean1500.py --extra-only). 'zymo' = windows as sequenced; 'zymo_oriented' = reverse-complemented to the
reference's forward strand where the read maps to '-' (diagnostic: training fragments are forward-strand only).

Groups: training species (L. monocytogenes, P. aeruginosa, B. subtilis, S. enterica, S. aureus), E. faecalis
(genus in training, species not), held-out genus (E. coli, L. fermentum), yeast (S. cerevisiae, C. neoformans),
unassigned (read did not map confidently). Writes results/zymo/*.csv, zymo_mock_community.xlsx and histograms.

    python data_gen/zymo/analyze_zymo.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import export_results as E  # noqa: E402

ROOT = Path("/scratch/home/glh52/glm-epinet-pyt"); OUT = ROOT / "results/zymo"; OUT.mkdir(parents=True, exist_ok=True)
TRAIN_SP = {"Listeria_monocytogenes", "Pseudomonas_aeruginosa", "Bacillus_subtilis", "Salmonella_enterica", "Staphylococcus_aureus"}
GROUP = {**{s: "training species" for s in TRAIN_SP}, "Enterococcus_faecalis": "genus in training",
         "Escherichia_coli": "held-out genus", "Lactobacillus_fermentum": "held-out genus",
         "Saccharomyces_cerevisiae": "yeast", "Cryptococcus_neoformans": "yeast", "unassigned": "unassigned"}
METHODS = {"softmax": ("base_conf", "base_pred"), "temperature": ("temp_conf", "base_pred"), "MC dropout": ("mc_conf", "mc_pred"),
           "Epinet": ("epi_conf", "epi_pred"), "Laplace": ("lap_conf", "lap_pred")}
# detection scores, higher = more novel
SCORES = {"1 - max prob (softmax)": lambda d: 1 - d.base_conf, "energy": lambda d: d.energy, "entropy (softmax)": lambda d: d.entropy,
          "MC dropout entropy": lambda d: d.mc_entropy, "MC dropout epistemic": lambda d: d.mc_epistemic,
          "Epinet entropy": lambda d: d.epi_entropy, "Epinet epistemic": lambda d: d.epi_epistemic,
          "Laplace entropy": lambda d: d.lap_entropy, "Laplace epistemic": lambda d: d.lap_epistemic,
          "conformal set size": lambda d: d.cp_size, "Mahalanobis": lambda d: d.maha, "kNN distance": lambda d: d.knn}
FPRS = (0.01, 0.05, 0.10)


def load(rank, name):
    p = ROOT / f"results/clean1500_es/NT_transformer_{rank}_s1"
    d = pd.read_parquet(p / f"scores_{name}.parquet"); idt = pd.read_parquet(p / "scores_id_test.parquet")
    cfg = json.load(open(ROOT / json.load(open(p / "meta.json"))["base"] / "config.json"))
    lab2id = {int(v): int(k) for k, v in cfg["id2label"].items()}
    d["y"] = [lab2id.get(int(t), -1) for t in d[rank]]; d["group"] = d.species.map(GROUP)
    return d, idt


def main():
    cal, det, flag, hist, strand = [], [], [], [], []
    for rank in ("family", "phylum"):
        for name, view in (("zymo", "as sequenced"), ("zymo_oriented", "strand-oriented")):
            try: d, idt = load(rank, name)
            except FileNotFoundError: print("missing", rank, name); continue
            key = dict(rank=rank, view=view)
            # classification / calibration per species and per group, window level (labeled bacteria only)
            for m, (cc, pc) in METHODS.items():
                if cc not in d: continue
                for lvl, g in [("all labeled bacteria", d[d.y >= 0])] + [(s, x) for s, x in d[d.y >= 0].groupby("species")] + \
                              [(f"group: {k}", x) for k, x in d[d.y >= 0].groupby("group")]:
                    r = E.cls_metrics(g.y.values, g[pc].values, g[cc].values)
                    cal.append(dict(**key, method=m, subset=lvl, n_windows=len(g), n_reads=g.read.nunique(),
                                    **{k: r[k] for k in ("accuracy", "macro_f1", "ece_50em", "ece_15ew", "mean_conf", "overconfidence")}))
                    if m == "softmax" and not lvl.startswith("group") and lvl != "all labeled bacteria":
                        vote = g.groupby("read").apply(lambda x: x[pc].mode().iloc[0] == x.y.iloc[0], include_groups=False)
                        cal[-1]["read_accuracy_majority_vote"] = float(vote.mean())
            # detection: yeast vs bacteria (confidently assigned), and held-out genus vs training species
            pairs = {"yeast vs all bacteria": (d.group == "yeast", d.group.isin(["training species", "genus in training", "held-out genus"])),
                     "held-out genus vs training species": (d.group == "held-out genus", d.group == "training species"),
                     "yeast vs training species": (d.group == "yeast", d.group == "training species")}
            for sname, fn in SCORES.items():
                s = fn(d).values.astype(float); si = fn(idt).values.astype(float)
                rs = d.assign(s=s).groupby("read").agg(s_mean=("s", "mean"), s_max=("s", "max"), group=("group", "first"))
                for pname, (pos, neg) in pairs.items():
                    r = E.det_metrics(s[neg.values], s[pos.values])
                    pr, nr = rs.group.isin(d[pos].group.unique()), rs.group.isin(d[neg].group.unique())
                    det.append(dict(**key, score=sname, comparison=pname, level="window", **r))
                    for agg_ in ("s_mean", "s_max"):
                        det.append(dict(**key, score=sname, comparison=pname, level=f"read ({agg_[2:]})",
                                        **E.det_metrics(rs[agg_][nr].values, rs[agg_][pr].values)))
                # flagged fraction at thresholds fixed on the clean id_test fragments
                for f in FPRS:
                    t = np.quantile(si, 1 - f); fl = s > t
                    row = dict(**key, score=sname, id_test_fpr=f, flagged_all=fl.mean(),
                               **{f"flagged | {g}": fl[(d.group == g).values].mean() for g in
                                  ["training species", "genus in training", "held-out genus", "yeast", "unassigned"]})
                    row["true yeast window fraction"] = (d.group == "yeast").mean()
                    row["precision for yeast"] = ((d.group == "yeast").values & fl).sum() / max(fl.sum(), 1)
                    flag.append(row)
                    if f == 0.05:
                        fr = rs.assign(fl=rs.s_mean > t)
                        flag[-1]["flagged_reads_all (mean score)"] = fr.fl.mean()
                        flag[-1]["flagged_reads | yeast"] = fr[fr.group == "yeast"].fl.mean()
                # histogram data
                lo, hi = np.nanquantile(np.r_[s, si], [0.001, 0.999]); bins = np.linspace(lo, hi, 61)
                for g, x in [("clean id_test", si)] + [(g, s[(d.group == g).values]) for g in GROUP.values() if g in set(d.group)]:
                    c, _ = np.histogram(x, bins=bins)
                    hist += [dict(**key, score=sname, group=g, bin_lo=bins[i], bin_hi=bins[i + 1], density=c[i] / max(len(x), 1)) for i in range(60)]
            # strand
            if name == "zymo":
                for st, g in d[d.y >= 0].groupby("strand"):
                    strand.append(dict(rank=rank, strand=st, n_windows=len(g), accuracy=(g.base_pred == g.y).mean(),
                                       mean_conf=g.base_conf.mean(), kNN_mean=g.knn.mean()))
    # reference-based tools (data_gen/zymo/tools_zymo.py), as-sequenced windows; thresholds from their clean id_test scores
    for rank in ("family", "phylum"):
        tp = OUT / f"tools_{rank}.parquet"
        if not tp.exists(): continue
        t = pd.read_parquet(tp); t["group"] = t.species.map(GROUP)
        for m, g0 in t.groupby("method", sort=False):
            tool = "kraken2" if m.startswith("Kraken2") else "mmseqs2"
            ref = pd.read_parquet(ROOT / f"results/clean1500_external/{tool}_{rank}/ext_id_test.parquet"); ref = ref[ref.method == m]
            key = dict(rank=rank, view="as sequenced")
            lab = g0[g0.y >= 0]
            for lvl, g in [("all labeled bacteria", lab)] + [(sp, x) for sp, x in lab.groupby("species")] + [(f"group: {k}", x) for k, x in lab.groupby("group")]:
                r = E.cls_metrics(g.y.values, g.pred.values, g.conf.values)
                cal.append(dict(**key, method=m, subset=lvl, n_windows=len(g), n_reads=g.read.nunique(), classification_coverage=float((g.pred >= 0).mean()),
                                **{k: r[k] for k in ("accuracy", "macro_f1", "ece_50em", "ece_15ew", "mean_conf", "overconfidence")}))
            pairs = {"yeast vs all bacteria": (g0.group == "yeast", g0.group.isin(["training species", "genus in training", "held-out genus"])),
                     "held-out genus vs training species": (g0.group == "held-out genus", g0.group == "training species"),
                     "yeast vs training species": (g0.group == "yeast", g0.group == "training species")}
            for c in [c for c in g0.columns if c.startswith("det_")]:
                sname = f"{m}: {c[4:]}"; s_ = g0[c].values.astype(float); si = ref[c].values.astype(float)
                for pname, (pos, neg) in pairs.items():
                    det.append(dict(**key, score=sname, comparison=pname, level="window", **E.det_metrics(s_[neg.values], s_[pos.values])))
                for f in FPRS:            # tied scores: flag a fraction of the ties so the id_test FPR is exactly f
                    th = np.quantile(si, 1 - f); at = (si == th).mean(); w = (f - (si > th).mean()) / at if at > 0 else 0.0
                    fl = (s_ > th) + w * (s_ == th)
                    row = dict(**key, score=sname, id_test_fpr=f, flagged_all=fl.mean(),
                               **{f"flagged | {g}": fl[(g0.group == g).values].mean() for g in
                                  ["training species", "genus in training", "held-out genus", "yeast", "unassigned"]})
                    row["true yeast window fraction"] = (g0.group == "yeast").mean()
                    flag.append(row)
    C, Dt, F, H, S = map(pd.DataFrame, (cal, det, flag, hist, strand))
    for n, x in [("classification", C), ("detection", Dt), ("flagged", F), ("histograms", H), ("strand", S)]:
        x.to_csv(OUT / f"{n}.csv", index=False)
    # figure: the PI's 'hump' view, family + phylum, as sequenced and oriented, six scores
    show = ["1 - max prob (softmax)", "Epinet epistemic", "MC dropout epistemic", "Laplace epistemic", "Mahalanobis", "kNN distance"]
    cols = {"clean id_test": "#888888", "training species": "#1f77b4", "genus in training": "#17becf", "held-out genus": "#ff7f0e",
            "yeast": "#d62728", "unassigned": "#bcbd22"}
    for rank in ("family", "phylum"):
        for view in ("as sequenced", "strand-oriented"):
            h = H[(H["rank"] == rank) & (H.view == view)]
            if h.empty: continue
            fig, ax = plt.subplots(2, 3, figsize=(15, 8))
            for a, sc in zip(ax.ravel(), show):
                for g, x in h[h.score == sc].groupby("group"):
                    a.plot((x.bin_lo + x.bin_hi) / 2, x.density, label=g, color=cols.get(g), lw=2 if g == "yeast" else 1.2)
                a.set_title(sc); a.set_yscale("log"); a.set_ylabel("fraction of windows")
            ax[0, 0].legend(fontsize=8); fig.suptitle(f"Zymo even community, NT {rank} model, windows {view}")
            fig.tight_layout(); fig.savefig(OUT / f"hist_{rank}_{view.replace(' ', '_')}.png", dpi=130); plt.close(fig)
    readme = pd.DataFrame({"notes": [
        "ZymoBIOMICS Microbial Community Standard (even, D6300), nanopore GridION ERR3152364 (Nicholls et al. 2019), 146,859 random reads "
        ">= 1.5 kb, split into 392,422 non-overlapping 1.5 kb windows. Ground truth per read: minimap2 -x map-ont to the Zymo references, "
        "MAPQ >= 20 and >= 50% of the read aligned; else 'unassigned'.",
        "Groups: training species = species in the clean1500 training set; genus in training = E. faecalis; held-out genus = E. coli, "
        "L. fermentum (their genera are held out of training: the benchmark's novel_genus case); yeast = S. cerevisiae, C. neoformans "
        "(eukaryotes; never seen).",
        "View 'as sequenced' = windows as read; 'strand-oriented' = reverse-complemented where the read maps to the '-' strand "
        "(training fragments are forward-strand only, so this isolates the strand effect; it uses the mapping, so it is a diagnostic).",
        "Models: NT family and phylum clean1500 models (seed 1); Epinet heads retrained 2026-10-25 (relative prior 0.3).",
        "flagged: fraction of windows above the score threshold that gives the stated false-positive rate on the clean id_test fragments.",
        "detection: AUROC / AUPR / FPR95 at window level, and per read (mean or max score over the read's windows).",
    ] + [f"sheet '{k}'" for k in ("classification", "detection", "flagged", "strand", "histograms")]})
    with pd.ExcelWriter(OUT / "zymo_mock_community.xlsx", engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for n, x in [("classification", C), ("detection", Dt), ("flagged", F), ("strand", S), ("histograms", H)]:
            x.round(5).to_excel(xw, sheet_name=n, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
