#!/usr/bin/env python3
"""Export every result into one workbook (+ one CSV per sheet) per dataset, with one shared metric set.

    PYTHONPATH=<dir with openpyxl> python scripts/export_results.py --out results/tables

Datasets (one .xlsx each, plus <name>/<sheet>.csv):
  aleatoric_boundary   promoter_all test split, binned by distance to the class boundary
  epistemic_alisim     promoter_all test anchors evolved by AliSim (t <= 3, plus t = 5, 8, 12)
  pbsim_darkmatter     clean 1.5 kb fragments (phylum, family models): ID / novel genus / novel family
                       / nonbacterial, plus the CAMISIM dark matter contig windows
  scorpio_gene_taxa    Scorpio gene classification: test / taxa_out / gene_out

Shared metrics (every labeled set, method and seed; aggregated sheets give mean / sd / n_seeds):
  classification   n, accuracy, error, macro F1 (classes present in the set's labels)
  calibration      ECE with 50 equal-mass bins (the manuscript's definition) and 15 equal-width bins,
                   MCE (15 equal-width), classwise ECE (Kull et al. 2019, 15 equal-width bins per class),
                   mean confidence, overconfidence = mean confidence - accuracy,
                   proper scoring rules NLL and Brier score,
                   bin sensitivity: equal-width and equal-mass ECE at 10 / 15 / 20 / 25 / 50 bins
  adaptivity       (reviewer 2, point 3) per uncertainty score: AURC and excess AURC (risk-coverage),
                   accuracy on the 50 / 80 / 90 / 95% most-certain examples, misclassification-detection
                   AUROC, and accuracy / confidence per uncertainty quintile
  decomposition    mean total / aleatoric / epistemic uncertainty per set or stratum, and the Spearman
                   correlation of each with the shift axis where the axis is continuous
  detection        shifted set vs its ID reference: AUROC, AUPR (shifted = positive), FPR at 95% TPR and
                   delta-AUROC vs the softmax maximum probability, for 1 - max probability, predictive
                   entropy and the epistemic term of every method, the conformal set size and, where
                   stored, energy, Mahalanobis and kNN distance
  conformal        coverage, mean set size, empty and singleton fractions
  K-convergence    (reviewer 2, point 6) the metrics above for MC dropout / Epinet / Laplace at K samples
Dataset-specific: margin quartiles and branch lengths for the promoter sets; dark matter trends.
Out-of-vocabulary examples (true label not a training class) are excluded from classification and
calibration metrics, and sets with < 50% in-vocabulary labels are treated as unlabeled.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve

ROOT = Path("/scratch/home/glh52/glm-epinet-pyt")
PROM = ROOT / "results/promoter_es"
TAX = {"": ROOT / "results/clean1500_es", "-multi": ROOT / "results/clean1500_es_carmania_prok"}
LIN = Path("/scratch/home/glh52/glm-epinet/DATA/clean1500/full_basic_lineage.csv")
GENE_D = Path("/scratch/home/glh52/glm-epinet/DATA/gene_taxa")
GENE_TIERS, GENE_CAP = {"id_test": "test", "taxa_out": "taxa_out", "gene_out": "gene_out"}, 25_000
BB_ORDER = ["NT", "DNABERT2", "hyenaDNA", "CARMANIA", "CARMANIA (multi-species)", "no pretraining", "reference-based tool"]
BB_NAME = {"NT_transformer": "NT", "nonpretrained": "no pretraining", "CARMANIA-multi": "CARMANIA (multi-species)"}
M_ORDER = ["softmax", "temperature", "MC dropout", "Epinet", "Laplace", "evidential", "ensemble (5 seeds)",
           "CNN", "CNN + MC dropout", "CNN ensemble (5 seeds)", "Kraken2 (raw score)", "Kraken2 (isotonic on val)",
           "MMseqs2 (raw score)", "MMseqs2 (isotonic on val)",
           "conformal", "k-mer RF", "CNN + MC dropout", "CNN ensemble"]
PROM_M = {"base": "softmax", "base_scaled": "temperature", "mc_dropout": "MC dropout", "conv_epinet": "Epinet",
          "laplace": "Laplace", "evidential": "evidential", "ensemble_k5": "ensemble (5 seeds)", "conformal": "conformal",
          "rf": "k-mer RF", "cnn_mc_dropout": "CNN + MC dropout", "cnn_ensemble": "CNN ensemble"}
# taxonomic scores_*.parquet: method -> (confidence, prediction, entropy, epistemic) columns
TAX_M = {"softmax": ("base_conf", "base_pred", "entropy", None), "temperature": ("temp_conf", "base_pred", None, None),
         "MC dropout": ("mc_conf", "mc_pred", "mc_entropy", "mc_epistemic"), "Epinet": ("epi_conf", "epi_pred", "epi_entropy", "epi_epistemic"),
         "Laplace": ("lap_conf", "lap_pred", "lap_entropy", "lap_epistemic"), "evidential": ("evid_conf", "evid_pred", "evid_entropy", "evid_vacuity")}
# taxonomic proper_*.parquet prefixes (proper_scores.py)
TAX_P = {"softmax": "base", "temperature": "temp", "MC dropout": "mc", "Epinet": "epi", "Laplace": "lap"}
BINS = (10, 15, 20, 25, 50)
COVER = (0.5, 0.8, 0.9, 0.95)


# ---------------------------------------------------------------- metrics
def ece_em(conf, corr, bins=50):
    if len(conf) < bins: return np.nan
    o = np.argsort(conf, kind="stable"); c, a = conf[o], corr[o]; e = np.linspace(0, len(c), bins + 1).astype(int)
    return float(sum((hi - lo) / len(c) * abs(a[lo:hi].mean() - c[lo:hi].mean()) for lo, hi in zip(e[:-1], e[1:]) if hi > lo))


def ece_ew(conf, corr, bins=15, mce=False):
    e = np.linspace(0, 1, bins + 1); i = np.clip(np.digitize(conf, e[1:-1]), 0, bins - 1)
    gaps = [((i == b).mean(), abs(corr[i == b].mean() - conf[i == b].mean())) for b in range(bins) if (i == b).any()]
    return float(max(g for _, g in gaps)) if mce else float(sum(w * g for w, g in gaps))


def classwise_ece(P, y, bins=15):
    """Kull et al. 2019: mean over classes of the equal-width ECE of p_k against 1[y = k]"""
    return float(np.mean([ece_ew(P[:, k], (y == k).astype(float), bins) for k in range(P.shape[1])]))


def cls_metrics(y, pred, conf, ptrue=None, sq=None, P=None):
    corr = (pred == y).astype(float)
    r = dict(n=len(y), accuracy=corr.mean(), error=1 - corr.mean(),
             macro_f1=f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0),
             ece_50em=ece_em(conf, corr), ece_15ew=ece_ew(conf, corr), mce_15ew=ece_ew(conf, corr, mce=True),
             mean_conf=conf.mean(), overconfidence=conf.mean() - corr.mean(),
             nll=float(-np.log(np.clip(ptrue, 1e-12, None)).mean()) if ptrue is not None else np.nan,
             brier=float((sq - 2 * ptrue + 1).mean()) if ptrue is not None and sq is not None else np.nan,
             classwise_ece_15ew=classwise_ece(P, y) if P is not None else np.nan)
    for b in BINS:
        r[f"ece_{b}ew"] = ece_ew(conf, corr, b); r[f"ece_{b}em"] = ece_em(conf, corr, b)
    return r


def selective(y, pred, U):
    """per uncertainty score: risk-coverage (AURC, excess AURC), accuracy at coverage, misclassification AUROC"""
    err = (pred != y).astype(float); n = len(err); rows = []
    opt = np.cumsum(np.sort(err)) / np.arange(1, n + 1)               # oracle ordering: errors last
    for s, u in U.items():
        if u is None: continue
        o = np.argsort(u, kind="stable"); risk = np.cumsum(err[o]) / np.arange(1, n + 1)
        r = dict(score=s, aurc=risk.mean(), e_aurc=risk.mean() - opt.mean(),
                 misclf_auroc=roc_auc_score(err, u) if 0 < err.sum() < n else np.nan)
        for c in COVER:
            k = max(1, int(round(c * n))); r[f"acc@{int(c * 100)}%"] = 1 - err[o][:k].mean()
        rows.append(r)
    return rows


def quintiles(y, pred, conf, U):
    corr = (pred == y).astype(float); rows = []
    for s, u in U.items():
        if u is None: continue
        q = pd.qcut(pd.Series(u).rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
        for b in q.cat.categories:
            m = (q == b).values
            rows.append(dict(score=s, quintile=b, n=int(m.sum()), u_mean=float(np.mean(u[m])), accuracy=corr[m].mean(), mean_conf=conf[m].mean()))
    return rows


def det_metrics(s_id, s_ood):
    s_id, s_ood = np.asarray(s_id, float), np.asarray(s_ood, float)
    s = np.r_[s_id, s_ood]
    if np.allclose(s, s[0]): return dict(auroc=np.nan, aupr=np.nan, fpr95=np.nan)
    lab = np.r_[np.zeros(len(s_id)), np.ones(len(s_ood))]; f, t, _ = roc_curve(lab, s)
    return dict(auroc=roc_auc_score(lab, s), aupr=average_precision_score(lab, s), fpr95=float(f[np.searchsorted(t, 0.95)]))


def agg(df, keys, vals=None):
    if df.empty: return df
    vals = vals or [c for c in df.columns if c not in keys and c != "seed" and df[c].dtype.kind in "fiub"]
    g = df.groupby(keys, sort=False)[vals]
    m, s, n = g.mean(), g.std(), g.count()
    return m.add_suffix("_mean").join(s.add_suffix("_sd")).join(n[[vals[0]]].rename(columns={vals[0]: "n_seeds"})).reset_index()


def order(df):
    if df.empty: return df
    df = df.copy()
    if "backbone" in df: df["_b"] = df.backbone.map(lambda b: BB_ORDER.index(b) if b in BB_ORDER else 99)
    if "method" in df: df["_m"] = df.method.map(lambda m: M_ORDER.index(m) if m in M_ORDER else 99)
    return df.sort_values([c for c in ("_b", "_m") if c in df], kind="stable").drop(columns=[c for c in ("_b", "_m") if c in df])


def pm(m, s, nd=3):
    if pd.isna(m): return ""
    return f"{m:.{nd}f}" if pd.isna(s) else f"{m:.{nd}f} ± {s:.{nd}f}"


def summary(cal_agg, sel_agg, set_col, sets, metrics=("macro_f1", "ece_50em", "nll", "brier")):
    """one row per backbone x method; per set: the main metrics and AURC of 1 - max prob, 'mean ± sd'"""
    rows = []
    a = sel_agg[sel_agg.score == "1 - max prob"] if not sel_agg.empty else sel_agg
    for (b, m), g in order(cal_agg).groupby(["backbone", "method"], sort=False):
        r = {"backbone": b, "method": m, "seeds": int(g.n_seeds.max())}
        for s in sets:
            x = g[g[set_col] == s]
            for k in metrics:
                r[f"{s} | {k}"] = pm(x[f"{k}_mean"].iloc[0], x[f"{k}_sd"].iloc[0]) if len(x) and f"{k}_mean" in x else ""
            y = a[(a.backbone == b) & (a.method == m) & (a[set_col] == s)] if not a.empty else a
            r[f"{s} | aurc"] = pm(y.aurc_mean.iloc[0], y.aurc_sd.iloc[0]) if len(y) else ""
        rows.append(r)
    return pd.DataFrame(rows)


def det_summary(da, shifted):
    """AUROC (mean ± sd) per backbone x method for each score, one column block per shifted set"""
    rows = []
    for (b, m), g in order(da).groupby(["backbone", "method"], sort=False):
        r = {"backbone": b, "method": m}
        for t in shifted:
            for _, x in g[g.shifted == t].iterrows():
                r[f"{t} | {x.score}"] = pm(x.auroc_mean, x.auroc_sd)
        rows.append(r)
    return pd.DataFrame(rows)


def add_delta(det):
    """delta-AUROC of every score vs the softmax maximum probability, same backbone / seed / shifted set"""
    if det.empty: return det
    ref = det[(det.method == "softmax") & (det.score == "1 - max prob")][["backbone", "seed", "shifted", "auroc"]].rename(columns={"auroc": "_ref"})
    det = det.merge(ref, on=["backbone", "seed", "shifted"], how="left")
    det["delta_auroc_vs_softmax"] = det.auroc - det._ref
    return det.drop(columns="_ref")


# ---------------------------------------------------------------- promoter
def load_prom(ds):
    cells = []
    for f in sorted(glob.glob(str(PROM / ds / "*" / "*.csv"))):
        d = pd.read_csv(f); meth = str(d.method.iloc[0])
        if meth not in PROM_M: continue
        d["backbone"] = BB_NAME.get(d.backbone.iloc[0], d.backbone.iloc[0]); d["method"] = PROM_M[meth]; d["seed"] = d.seed.astype(str)
        if ds == "alisim": d = d[d.branch_length <= 3.0 + 1e-9]
        cells.append(d.reset_index(drop=True))
    return cells


def prom_U(d):
    U = {"1 - max prob": (1 - d.prob_max).values, "entropy": d.U_total.values, "epistemic": d.U_epistemic.values,
         "aleatoric": d.U_aleatoric.values}
    if "vacuity" in d: U["vacuity"] = d.vacuity.values
    if d.U_epistemic.abs().max() < 1e-6:     # deterministic methods: epistemic = 0 by construction (float noise only)
        U["epistemic"] = None; U["aleatoric"] = None
    return U


def prom_tables(ds, strat, axis):
    cells = load_prom(ds); cal, sel, qnt, dec, det, cp = [], [], [], [], [], []
    for d in cells:
        b, m, sd = d.backbone.iloc[0], d.method.iloc[0], d.seed.iloc[0]; key = dict(backbone=b, method=m, seed=sd)
        groups = [("pooled", d)] + [(str(k), g) for k, g in d.groupby(strat)]
        if m == "conformal":
            for st, g in groups:
                cp.append(dict(**key, stratum=st, n=len(g), coverage=g.covered.mean(), mean_set_size=g.set_size.mean(),
                               frac_empty=(g.set_size == 0).mean(), frac_singleton=(g.set_size == 1).mean(), frac_both=(g.set_size == 2).mean()))
        else:
            for st, g in groups:
                y = g.labels.values; P = g[["prob_0", "prob_1"]].values; pt = P[np.arange(len(y)), y]
                cal.append(dict(**key, stratum=st, **cls_metrics(y, g.pred.values, g.prob_max.values, pt, (P ** 2).sum(1), P)))
                U = prom_U(g)
                sel += [dict(**key, stratum=st, **r) for r in selective(y, g.pred.values, U)]
                if st == "pooled": qnt += [dict(**key, stratum=st, **r) for r in quintiles(y, g.pred.values, g.prob_max.values, U)]
                dec.append(dict(**key, stratum=st, n=len(g), u_total=g.U_total.mean(), u_aleatoric=g.U_aleatoric.mean(), u_epistemic=g.U_epistemic.mean()))
            x = d[axis].values
            for c in ("U_total", "U_aleatoric", "U_epistemic"):
                ok = np.isfinite(x) & np.isfinite(d[c].values)
                dec.append(dict(**key, stratum=f"spearman rho vs {axis}", n=int(ok.sum()),
                                **{f"rho_{c[2:].lower()}": spearmanr(x[ok], d[c].values[ok])[0] if d[c].values[ok].std() > 0 else np.nan}))
        if ds == "alisim":
            scores = {"set size": d.set_size.values} if m == "conformal" else {k: v for k, v in prom_U(d).items() if v is not None}
            t = d.branch_length.values
            for sname, s in scores.items():
                si = s[t == 0]
                for tt, msk in [("all t > 0", t > 0)] + [(str(v), t == v) for v in sorted(set(t[t > 0]))]:
                    det.append(dict(**key, score=sname, shifted=tt, **det_metrics(si, s[msk])))
    f = lambda L: pd.DataFrame(L)
    return f(cal), f(sel), f(qnt), f(dec), add_delta(f(det)), f(cp)


# ---------------------------------------------------------------- taxonomic
def tax_models(rank):
    for suf, root in TAX.items():
        for p in sorted(root.glob(f"*_{rank}_s*")):
            if not (p / "scores_id_test.parquet").exists(): continue
            bb = p.name.split(f"_{rank}_s")[0] + suf
            yield BB_NAME.get(bb, bb), p.name.rsplit("_s", 1)[1], p


def tax_labels(rank, p, sets):
    base = json.load(open(p / "meta.json"))["base"]
    cfg = json.load(open((ROOT / base if not base.startswith("/") else Path(base)) / "config.json"))
    lab2id = {int(v): int(k) for k, v in cfg["id2label"].items()}
    if rank == "gene_taxa":
        Y = {}
        for k, f in GENE_TIERS.items():
            d = pd.read_csv(GENE_D / f"{f}.csv", usecols=["labels"])
            d = d.sample(GENE_CAP, random_state=1) if len(d) > GENE_CAP else d
            Y[k] = np.array([lab2id.get(int(l), -1) for l in d["labels"]]) if k != "gene_out" else np.full(len(d), -1)
        return Y
    lin = pd.read_csv(LIN, index_col=0); s2r = dict(zip(lin.species.astype(int), lin[rank].astype(int)))
    return {k: np.array([lab2id.get(s2r.get(int(t), -1), -1) for t in v.taxid]) for k, v in sets.items()}


def tax_tables(rank):
    cal, sel, qnt, dec, det, cp, oov, kc = [], [], [], [], [], [], [], []
    for b, sd, p in tax_models(rank):
        sets = {Path(f).stem.replace("scores_", ""): pd.read_parquet(f) for f in sorted(p.glob("scores_*.parquet"))}
        sets.pop("darkmatter", None)
        Y = tax_labels(rank, p, sets)
        cw = pd.read_csv(p / "classwise.csv") if (p / "classwise.csv").exists() else None
        for name, sc in sets.items():
            y = Y[name]; iv = y >= 0
            oov.append(dict(backbone=b, seed=sd, set=name, n=len(y), frac_in_vocab=iv.mean(), labeled=iv.mean() >= 0.5))
            pr = p / f"proper_{name}.parquet"
            if iv.mean() >= 0.5:
                P = pd.read_parquet(pr) if pr.exists() else None
                for m, (cc, pc, ec, xc) in TAX_M.items():
                    key = dict(backbone=b, method=m, seed=sd, set=name)
                    if P is not None and m in TAX_P:          # proper-score pass: every column from one consistent draw
                        x = TAX_P[m]; conf, pred = P[f"{x}_conf"].values[iv], P[f"{x}_pred"].values[iv]
                        ptrue, sq = P[f"{x}_ptrue"].values[iv], P[f"{x}_sq"].values[iv]
                        ent, epi = P[f"{x}_ent"].values[iv], (P[f"{x}_epi"].values[iv] if m in ("MC dropout", "Epinet", "Laplace") else None)
                        src = "proper_scores"
                    elif cc in sc:
                        conf, pred, ptrue, sq = sc[cc].values[iv], sc[pc].values[iv], None, None
                        ent = sc[ec].values[iv] if ec else None; epi = sc[xc].values[iv] if xc else None; src = "scores"
                    else:
                        continue
                    r = cls_metrics(y[iv], pred, conf, ptrue, sq)
                    if cw is not None and m in TAX_P:
                        c = cw[(cw.set == name) & (cw.method == TAX_P[m])]
                        if len(c): r.update(classwise_ece_15ew=c.classwise_ece.iloc[0], classwise_ece_present_15ew=c.classwise_ece_present.iloc[0])
                    cal.append(dict(**key, frac_in_vocab=iv.mean(), source=src, **r))
                    U = {"1 - max prob": 1 - conf, "entropy": ent, ("vacuity" if m == "evidential" else "epistemic"): epi}
                    sel += [dict(**key, **x) for x in selective(y[iv], pred, U)]
                    qnt += [dict(**key, **x) for x in quintiles(y[iv], pred, conf, U)]
            for m, (cc, pc, ec, xc) in TAX_M.items():
                if ec and ec in sc:
                    dec.append(dict(backbone=b, method=m, seed=sd, set=name, n=len(sc), u_total=sc[ec].mean(),
                                    u_epistemic=sc[xc].mean() if xc and m != "evidential" else np.nan,
                                    u_aleatoric=(sc[ec] - sc[xc]).mean() if xc and m != "evidential" else np.nan))
        ref = sets["id_test"]
        for name, sc in sets.items():
            if name == "id_test": continue
            for m, (cc, _, ec, xc) in TAX_M.items():
                if cc not in ref: continue
                for sname, col in {"1 - max prob": cc, "entropy": ec, ("vacuity" if m == "evidential" else "epistemic"): xc}.items():
                    if col is None or col not in ref: continue
                    v_id, v_ood = ref[col].values, sc[col].values
                    if sname == "1 - max prob": v_id, v_ood = 1 - v_id, 1 - v_ood
                    det.append(dict(backbone=b, method=m, seed=sd, score=sname, shifted=name, **det_metrics(v_id, v_ood)))
            for sname, col, meth in [("energy", "energy", "softmax"), ("Mahalanobis", "maha", "softmax"),
                                     ("kNN distance", "knn", "softmax"), ("set size", "cp_size", "conformal")]:
                if col in ref:
                    det.append(dict(backbone=b, method=meth, seed=sd, score=sname, shifted=name, **det_metrics(ref[col].values, sc[col].values)))
        if (p / "conformal.csv").exists():
            for r in pd.read_csv(p / "conformal.csv").to_dict("records"):
                cp.append(dict(backbone=b, method="conformal", seed=sd, set=r["set"], n=r["n"], coverage=r.get("coverage"),
                               mean_set_size=r["mean_set_size"], median_set_size=r.get("median_set_size"),
                               frac_empty=r["frac_empty"], frac_singleton=r["frac_singleton"]))
        if (p / "kconv.csv").exists():
            k = pd.read_csv(p / "kconv.csv"); k.insert(0, "seed", sd); k.insert(0, "backbone", b)
            k["method"] = k.method.map({"mc": "MC dropout", "epi": "Epinet", "lap": "Laplace"}); kc.append(k)
    f = lambda L: pd.DataFrame(L)
    ec, es, eq, ed, et = ext_tables(rank)
    cal, sel, qnt, dec, det = cal + ec, sel + es, qnt + eq, dec + ed, det + et
    return f(cal), f(sel), f(qnt), f(dec), add_delta(f(det)), f(cp), f(oov), (pd.concat(kc) if kc else pd.DataFrame())


EXT = ROOT / "results/clean1500_external"


def ext_tables(rank):
    """non-GLM models (external_common.py format): k-mer RF / CNN ('no pretraining') and Kraken2 /
    MMseqs2 ('reference-based tool'); same metrics as the GLM methods. Tools abstain (pred = -2):
    counted as errors, with classification coverage reported."""
    cal, sel, qnt, dec, det = [], [], [], [], []
    for p in sorted(EXT.glob(f"*_{rank}")):
        if p.name.startswith("smoke_"): continue
        tool = p.name.startswith(("kraken2", "mmseqs2")); bb = "reference-based tool" if tool else "no pretraining"
        F = {f.stem[4:]: pd.read_parquet(f) for f in sorted(p.glob("ext_*.parquet"))}
        if "id_test" not in F: continue                     # not scored yet
        for m in F["id_test"].method.unique():
            G = {n: d[d.method == m].reset_index(drop=True) for n, d in F.items()}
            for n, d in G.items():
                y = d.y.values; iv = y >= 0; key = dict(backbone=bb, method=m, seed="1", set=n)
                epi = d.epi.values if d.epi.notna().all() and np.abs(d.epi.values).max() > 1e-6 else None
                ent = d.ent.values if d.ent.notna().all() else None
                if iv.mean() >= 0.5:
                    pred, conf = d.pred.values[iv], d.conf.values[iv]
                    r = cls_metrics(y[iv], pred, conf, d.ptrue.values[iv], d.sq.values[iv])
                    cal.append(dict(**key, frac_in_vocab=iv.mean(), source="external", classification_coverage=float((pred >= 0).mean()), **r))
                    U = {"1 - max prob": 1 - conf, "entropy": ent[iv] if ent is not None else None, "epistemic": epi[iv] if epi is not None else None}
                    sel += [dict(**key, **x) for x in selective(y[iv], pred, U)]
                    qnt += [dict(**key, **x) for x in quintiles(y[iv], pred, conf, U)]
                if ent is not None:
                    dec.append(dict(**key, n=len(d), u_total=ent.mean(), u_epistemic=epi.mean() if epi is not None else np.nan,
                                    u_aleatoric=(ent - epi).mean() if epi is not None else np.nan))
            ref = G["id_test"]
            for n, d in G.items():
                if n == "id_test": continue
                for c in [c for c in ref.columns if c.startswith("det_") and ref[c].notna().all()]:
                    nm = c[4:].replace("1 - confidence", "1 - max prob")
                    det.append(dict(backbone=bb, method=m, seed="1", score=nm, shifted=n, **det_metrics(ref[c].values, d[c].values)))
    return cal, sel, qnt, dec, det


# ---------------------------------------------------------------- writing
README_COMMON = [
    "Every value is per seed in the *_per_seed sheets; the other sheets give mean / sd / n_seeds over seeds "
    "(sd is blank for a single seed). Summary sheets show 'mean ± sd'.",
    "accuracy / error / macro F1: top-1 prediction of each method's mean predictive distribution; macro F1 = "
    "unweighted mean of per-class F1 over the classes present in the set's true labels.",
    "ECE (ece_50em, manuscript definition): sum_m |B_m|/n * |acc(B_m) - conf(B_m)|, with conf(B_m) = mean top-1 "
    "probability and acc(B_m) = fraction correct among the examples in bin B_m, over 50 bins of equal size "
    "(equal-mass). ece_<k>ew / ece_<k>em: the same with k equal-width ([0, 1/k), ...) or equal-mass bins (bin "
    "sensitivity). mce_15ew: largest bin gap. classwise_ece_15ew: Kull et al. 2019, the mean over all classes of the "
    "15-bin equal-width ECE of p_k against 1[y = k] (classwise_ece_present_15ew: over the classes present in the set).",
    "NLL = -mean log p(true class); Brier = mean sum_k (p_k - 1[y = k])^2 (range 0-2).",
    "Adaptivity (selective sheets): examples sorted from most to least certain by each score; AURC = mean error rate "
    "over all coverage levels (lower = the score ranks its own errors last); e_aurc = AURC minus the oracle AURC; "
    "acc@c = accuracy on the c most-certain fraction; misclf_auroc = AUROC of the score for separating errors from "
    "correct predictions. Quintile sheets: accuracy and confidence per uncertainty quintile (Q1 = most certain).",
    "Detection: AUROC / AUPR (shifted set = positive) / FPR at 95% TPR, scores oriented so higher = more uncertain or "
    "more novel; delta_auroc_vs_softmax = AUROC minus that of the softmax maximum probability on the same pair.",
    "Uncertainty terms: total = normalized predictive entropy H(mean p)/log C; epistemic = mutual information "
    "(H(mean p) - mean H(p_k), normalized) over the method's K samples (0 by construction for deterministic methods); "
    "aleatoric = total - epistemic. These are the entropy terms of each method's samples, not posterior quantities.",
]


def write(name, sheets, readme, out):
    out.mkdir(parents=True, exist_ok=True); (out / name).mkdir(exist_ok=True)
    rd = pd.DataFrame({"notes": readme + README_COMMON + [f"sheet '{k}': {v[1]}" for k, v in sheets.items()]})
    with pd.ExcelWriter(out / f"{name}.xlsx", engine="openpyxl") as xw:
        rd.to_excel(xw, sheet_name="README", index=False)
        for sh, (df, _) in sheets.items():
            df = df.copy()
            for c in df.columns:
                if df[c].dtype.kind == "f": df[c] = df[c].round(5)
            df.to_excel(xw, sheet_name=sh[:31], index=False)
            df.to_csv(out / name / f"{sh}.csv", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                w = max(len(str(c.value)) if c.value is not None else 0 for c in col[:300])
                ws.column_dimensions[col[0].column_letter].width = min(max(9, w + 2), 160 if ws.title == "README" else 48)
            ws.freeze_panes = "A2"
    print("wrote", out / f"{name}.xlsx", {k: len(v[0]) for k, v in sheets.items()}, flush=True)


def std_sheets(prefix, set_col, cal, sel, qnt, dec, det, cp, sets, det_sets, what):
    p = f"{prefix}_" if prefix else ""
    ca = order(agg(cal, ["backbone", "method", set_col]))
    sa = order(agg(sel, ["backbone", "method", set_col, "score"])) if not sel.empty else sel
    S = {f"{p}summary": (summary(ca, sa, set_col, sets), f"{what}: macro F1, ECE (50 equal-mass), NLL, Brier and AURC (1 - max prob) per {set_col}, mean ± sd"),
         f"{p}calibration": (ca, f"{what}: every classification / calibration metric incl. bin sensitivity"),
         f"{p}calibration_per_seed": (order(cal), "the same, one row per seed"),
         f"{p}selective": (sa, f"{what}: adaptivity - AURC, excess AURC, accuracy at coverage, misclassification AUROC, per uncertainty score"),
         f"{p}selective_per_seed": (order(sel), "the same, one row per seed"),
         f"{p}quintiles": (order(agg(qnt, ["backbone", "method", set_col, "score", "quintile"])) if not qnt.empty else qnt,
                           f"{what}: accuracy / confidence per uncertainty quintile (Q1 = most certain)"),
         f"{p}decomposition": (order(agg(dec, ["backbone", "method", set_col])) if not dec.empty else dec,
                               f"{what}: mean total / aleatoric / epistemic uncertainty (and Spearman rho vs the shift axis)")}
    if not det.empty:
        da = order(agg(det, ["backbone", "method", "score", "shifted"]))
        S[f"{p}detection_summary"] = (det_summary(da, det_sets), f"{what}: detection AUROC (mean ± sd) per score")
        S[f"{p}detection"] = (da, f"{what}: AUROC / AUPR / FPR95 / delta-AUROC vs softmax per score and shifted set")
        S[f"{p}detection_per_seed"] = (order(det), "the same, one row per seed")
    if not cp.empty:
        S[f"{p}conformal"] = (order(agg(cp, ["backbone", "method", set_col])), f"{what}: split conformal (LAC score, alpha 0.1, calibrated on validation data)")
        S[f"{p}conformal_per_seed"] = (order(cp), "the same, one row per seed")
    return S


def prom_kconv(datasets):
    fs = sorted(glob.glob(str(PROM / "kconv" / "kconv_*.csv")))
    if not fs: return pd.DataFrame()
    k = pd.concat([pd.read_csv(f) for f in fs]); k = k[k.dataset.isin(datasets)]
    k["backbone"] = k.backbone.map(lambda b: BB_NAME.get(b, b)); k["method"] = k.method.map(PROM_M)
    return k


def prior_sweep_sheets(ds):
    """Epinet relative prior-scale sweep on the promoter models (4 backbones x 3 seeds x {0, 0.1, 0.3, 1, 3}),
    the selection criterion (validation mixture NLL) and the resulting test metrics"""
    f, v = ROOT / "results/promoter_es_priorsweep/sweep_summary_per_seed.csv", ROOT / "results/promoter_es_priorsweep_val.csv"
    if not f.exists(): return {}
    d = pd.read_csv(f); d = d[d.dataset == ds].copy(); d["backbone"] = d.backbone.map(lambda b: BB_NAME.get(b, b))
    keys = ["backbone", "variant"]
    S = {"prior_sweep": (order(agg(d.drop(columns="dataset"), keys)),
                         "Epinet prior-scale sweep (relative scale: each prior's spread = scale x base logit spread), mean over 3 seeds; "
                         "'prior 0' = learnable head only (no priors); 'old (unfixed)' = heads before the 2026-10-25 fixes; selection by "
                         "validation mixture NLL (sheet prior_sweep_val), chosen scale 0.3"),
         "prior_sweep_per_seed": (order(d), "the same, one row per seed")}
    if v.exists():
        x = pd.read_csv(v); x["backbone"] = x.backbone.map(lambda b: BB_NAME.get(b, b))
        S["prior_sweep_val"] = (order(agg(x, ["backbone", "prior"], ["val_nll", "base_val_ce", "delta_vs_base", "epochs"])),
                                "selection criterion: validation mixture NLL of each Epinet head vs the base model's validation CE")
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results/tables")); out = Path(ap.parse_args().out)
    KC = "K-convergence: MC dropout / Epinet / Laplace metrics on nested prefixes of K_MAX samples (one draw per cell), per seed"

    # aleatoric boundary
    cal, sel, qnt, dec, _, cp = prom_tables("boundary", "quartile", "margin")
    S = std_sheets("", "stratum", cal, sel, qnt, dec, pd.DataFrame(), cp, ["pooled", "Q1", "Q2", "Q3", "Q4"], [], "aleatoric shift")
    k = prom_kconv(["boundary"])
    if not k.empty:
        S["kconv"] = (order(agg(k, ["backbone", "method", "K"], [c for c in k.columns if c not in ("dataset", "backbone", "method", "seed", "K", "n")])), KC + " (mean over seeds)")
        S["kconv_per_seed"] = (order(k), KC)
    S.update(prior_sweep_sheets("boundary"))
    write("aleatoric_boundary", S,
          ["Aleatoric shift: promoter_all test split (1,584 sequences; inputs are in-distribution), stratified by the margin "
           "quartile of a reference model (Q1 = nearest the class boundary: the most ambiguous sequences). stratum 'pooled' = all.",
           "No detection sheet: there is no input shift to detect; adaptivity is measured by the selective / quintile sheets "
           "and the decomposition sheet (Spearman rho of each uncertainty term vs the margin; negative = more uncertain near "
           "the boundary).",
           "Models: promoter_all fine-tunes with early stopping, seeds 1-3; ensemble = seeds 1-5; k-mer RF / CNN baselines "
           "(not affected by the fine-tuning fixes) from the earlier sweep."], out)

    # epistemic AliSim
    cal, sel, qnt, dec, det, cp = prom_tables("alisim", "branch_length", "branch_length")
    cal2, sel2, _, dec2, _, cp2 = prom_tables("alisim_extended", "branch_length", "branch_length")
    nopool = lambda x: x[~x.stratum.astype(str).str.startswith(("pooled", "spearman"))] if not x.empty else x
    cal, sel, dec, cp = pd.concat([cal, nopool(cal2)]), pd.concat([sel, nopool(sel2)]), pd.concat([dec, nopool(dec2)]), pd.concat([cp, nopool(cp2)])
    tsort = lambda x: x.assign(_t=x.stratum.map(lambda s: -1 if not s.replace(".", "").isdigit() else float(s))).sort_values("_t", kind="stable").drop(columns="_t")
    cal, sel, dec, cp = tsort(cal), tsort(sel), tsort(dec), tsort(cp)
    S = std_sheets("", "stratum", cal, sel, qnt, dec, det, cp, ["pooled", "0.0", "0.1", "0.3", "0.5", "1.0", "3.0", "5.0", "12.0"],
                   ["all t > 0", "0.1", "0.5", "1.0", "3.0"], "epistemic shift")
    k = prom_kconv(["alisim"])
    if not k.empty:
        S["kconv"] = (order(agg(k, ["backbone", "method", "K"], [c for c in k.columns if c not in ("dataset", "backbone", "method", "seed", "K", "n")])), KC + " (mean over seeds)")
        S["kconv_per_seed"] = (order(k), KC)
    S.update(prior_sweep_sheets("alisim"))
    write("epistemic_alisim", S,
          ["Epistemic shift: the 400 promoter_all test anchors evolved with AliSim (HKY+G4) at branch lengths t = 0 ... 3 "
           "(16 values) and t = 5, 8, 12 (extended set, reported per t only); labels are inherited from the anchor. "
           "stratum 'pooled' = all t <= 3.",
           "Detection: unmutated anchors (t = 0) vs each branch length and vs all t > 0.",
           "Models: promoter_all fine-tunes with early stopping, seeds 1-3; ensemble = seeds 1-5; k-mer RF / CNN baselines "
           "from the earlier sweep."], out)

    # pbsim + dark matter
    S = {}
    for rank in ("phylum", "family"):
        cal, sel, qnt, dec, det, cp, oov, kc = tax_tables(rank)
        sets = [s for s in ["id_test", "novel_genus", "novel_family"] if s in set(cal.set)]
        S.update(std_sheets(rank, "set", cal, sel, qnt, dec, det, cp, sets, ["novel_genus", "novel_family", "nonbacterial"], f"{rank} models"))
        S[f"{rank}_label_coverage"] = (order(oov), f"{rank} models: fraction of each set whose true label is a training class")
        if not kc.empty:
            S[f"{rank}_kconv"] = (order(kc), KC + f" ({rank} models, 500 fragments per set)")
    S["darkmatter"] = (pd.concat([pd.read_csv(TAX[""] / "full_darkmatter_per_seed.csv")] +
                                 [pd.read_csv(f) for f in sorted(EXT.glob("*/darkmatter.csv")) if not f.parent.name.startswith("smoke_")]),
                       "dark matter: fraction of windows flagged at a given id_test false-positive rate per UP level, slope over the "
                       "15 samples and its p-value, and AUROC id_test vs each UP level, for every score and model")
    write("pbsim_darkmatter", S,
          ["Clean 1.5 kb fragments (no read errors). id_test = held-out regions of training genomes; novel_genus / novel_family "
           "= genera / families held out of training; nonbacterial = archaea, eukaryotes, viruses (labels outside the training "
           "vocabulary: detection only).",
           "Phylum models: 43 classes (labeled sets: id_test, novel_genus, novel_family); family models: 460 classes (labeled: "
           "id_test, novel_genus). One training seed (seed 1) per model so far, so sd is blank.",
           "Dark matter: CAMISIM samples at unknown proportion UP 0.25 / 0.75 / 1.0 (5 samples each), MEGAHIT contigs, "
           "non-overlapping 1.5 kb windows, 20k per sample; unlabeled, so only flagged fractions and score distributions.",
           "Non-GLM rows: 'no pretraining' = k-mer RF (200 trees on 6-mer frequencies) and a from-scratch CNN (single, MC "
           "dropout K = 10, 5-seed ensemble); 'reference-based tool' = Kraken2 and MMseqs2 against a database of the TRAINING "
           "genome regions only. Tools give one label: p(pred) = confidence, the rest spread uniformly (abstentions = uniform, "
           "counted as errors; classification_coverage = fraction not abstained). 'raw score' = Kraken2's clade k-mer fraction / "
           "MMseqs2 identity x coverage; 'isotonic on val' = that score calibrated by isotonic regression on val.csv. "
           "Ranking metrics (AURC, misclassification AUROC, detection) are unchanged by the calibration (monotone)."], out)

    # Scorpio gene_taxa
    cal, sel, qnt, dec, det, cp, oov, kc = tax_tables("gene_taxa")
    S = std_sheets("", "set", cal, sel, qnt, dec, det, cp, ["id_test", "taxa_out"], ["taxa_out", "gene_out"], "gene_taxa models")
    S["label_coverage"] = (order(oov), "fraction of each set whose true label is a training class")
    write("scorpio_gene_taxa", S,
          ["Scorpio gene classification (437 gene classes). id_test = held-out genes of training organisms (25k sample); "
           "taxa_out = the same gene classes from unseen organisms (labels valid); gene_out = unseen gene classes (no valid "
           "labels: detection only; 25k sample). One training seed (seed 1).",
           "Non-GLM rows ('no pretraining'): k-mer RF and a from-scratch CNN (single, MC dropout K = 10, 5-seed ensemble), "
           "sequences cut to their first 3,000 bp like the GLM inputs."], out)


if __name__ == "__main__":
    main()
