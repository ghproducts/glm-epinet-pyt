"""Shared helpers for the non-GLM models on clean1500 / gene_taxa (Kraken2, MMseqs2, k-mer RF, CNN).

They are scored on exactly the sets, row order and label vocabulary of eval_clean1500.py, and
written in one long format that scripts/export_results.py reads next to the GLM results:

  results/clean1500_external/<model>_<rank>/ext_<set>.parquet   one row per (example, method):
      method, y (training-vocabulary class index, -1 = out of vocabulary), pred (class index,
      -2 = abstain), conf (top-1 probability, 0 when abstaining), ptrue, sq (sum_k p_k^2),
      ent / epi (normalized total / epistemic entropy, NaN when the method has none),
      and det_<score> columns: detection scores, higher = more novel
  results/clean1500_external/<model>_<rank>/darkmatter.csv       flagged-fraction trends, same
      schema as results/clean1500_es/full_darkmatter_per_seed.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path("/scratch/home/glh52/glm-epinet-pyt")
D = Path("/scratch/home/glh52/glm-epinet/DATA/clean1500")
GENE_D = Path("/scratch/home/glh52/glm-epinet/DATA/gene_taxa")
TIERS = ["id_test", "novel_genus", "novel_family", "nonbacterial"]
GENE_TIERS, GENE_CAP = {"id_test": "test", "taxa_out": "taxa_out", "gene_out": "gene_out"}, 25_000
OUT = ROOT / "results/clean1500_external"
DM_FPRS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def vocab(rank):
    """training label vocabulary (label value -> class index), taken from a trained GLM config so
    every model shares the same class indices"""
    meta = json.load(open(ROOT / f"results/clean1500_es/NT_transformer_{rank}_s1/meta.json"))
    cfg = json.load(open(ROOT / meta["base"] / "config.json"))
    return {int(v): int(k) for k, v in cfg["id2label"].items()}


def lineage():
    return pd.read_csv(D / "full_basic_lineage.csv", index_col=0)


def eval_sets(rank, darkmatter=True):
    """{name: DataFrame} in eval_clean1500.py's order and sampling; adds column y (class index or -1)."""
    lab2id = vocab(rank)
    if rank == "gene_taxa":
        sets = {}
        for k, f in GENE_TIERS.items():
            d = pd.read_csv(GENE_D / f"{f}.csv")
            d = d.sample(GENE_CAP, random_state=1) if len(d) > GENE_CAP else d
            d = d.reset_index(drop=True)
            d["y"] = [lab2id.get(int(l), -1) for l in d["labels"]] if k != "gene_out" else -1
            sets[k] = d
        return sets
    lin = lineage(); s2r = dict(zip(lin.species.astype(int), lin[rank].astype(int)))
    sets = {t: pd.read_csv(D / f"{t}.csv") for t in TIERS}
    if darkmatter:
        sets["darkmatter"] = pd.read_csv(D / "darkmatter_contig_windows.csv")
    for k, d in sets.items():
        d["y"] = [lab2id.get(s2r.get(int(t), -1), -1) for t in d.taxid] if "taxid" in d else -1
    return sets


def summarize_probs(P):
    """P: [K, N, C] numpy probabilities -> mean p, normalized total and epistemic entropy"""
    C = P.shape[-1]; pm = P.mean(0)
    H = -(pm * np.log(np.clip(pm, 1e-12, None))).sum(-1)
    Hk = -(P * np.log(np.clip(P, 1e-12, None))).sum(-1).mean(0)
    return pm, H / np.log(C), np.clip(H - Hk, 0, None) / np.log(C)


def rows_from_probs(method, y, pm, ent, epi, extra_det=None):
    pred = pm.argmax(1); conf = pm.max(1)
    pt = np.where(y >= 0, pm[np.arange(len(y)), np.clip(y, 0, None)], np.nan)
    df = pd.DataFrame(dict(method=method, y=y, pred=pred, conf=conf, ptrue=pt, sq=(pm ** 2).sum(1), ent=ent, epi=epi))
    df["det_1 - max prob"] = 1 - conf; df["det_entropy"] = ent
    if epi is not None and np.nanmax(np.abs(epi)) > 1e-6: df["det_epistemic"] = epi
    for k, v in (extra_det or {}).items(): df[f"det_{k}"] = v
    return df


def rows_from_label(method, y, pred, conf, C, extra_det=None):
    """single-label predictor (alignment / k-mer tools): p(pred) = conf, the rest spread uniformly;
    abstention (pred = -2) = uniform over all C classes"""
    ab = pred < 0; conf = np.where(ab, 1.0 / C, np.clip(conf, 0, 1))
    other = (1 - conf) / (C - 1)
    pt = np.where(y < 0, np.nan, np.where(ab, 1.0 / C, np.where(pred == y, conf, other)))
    sq = np.where(ab, 1.0 / C, conf ** 2 + (C - 1) * other ** 2)
    df = pd.DataFrame(dict(method=method, y=y, pred=pred, conf=np.where(ab, 0.0, conf), ptrue=pt, sq=sq,
                           ent=np.nan, epi=np.nan))
    df["det_1 - confidence"] = 1 - df.conf.values
    for k, v in (extra_det or {}).items(): df[f"det_{k}"] = v
    return df


def darkmatter_rows(model, rank, dm, idt, scores):
    """flagged fraction per UP level at each id_test false-positive rate, slope over the 15 samples"""
    rows = []
    for sc in scores:
        a, b = idt[sc].values, dm[sc].values
        auc = {f"auroc_up{u}": roc_auc_score(np.r_[np.zeros(len(a)), np.ones((dm.up == u).sum())], np.r_[a, b[dm.up.values == u]])
               for u in (0.25, 0.75, 1.0)}
        for f in DM_FPRS:
            # flag score > t, plus a fraction w of the ties at t, so the id_test false-positive rate is exactly f
            # (tool scores are heavily tied: abstentions, no-hit = 0; a strict '>' would flag nothing)
            t = np.quantile(a, 1 - f); at = (a == t).mean(); w = (f - (a > t).mean()) / at if at > 0 else 0.0
            ps = dm.assign(fl=(b > t) + w * (b == t)).groupby(["up", "sample"]).fl.mean().reset_index()
            sl, _, _, p, _ = stats.linregress(ps.up, ps.fl); mu = ps.groupby("up").fl.mean()
            rows.append(dict(backbone=model, rank=rank, seed=1, score=sc.replace("det_", ""), fpr=f, flag_up0_25=mu[0.25],
                             flag_up0_75=mu[0.75], flag_up1_0=mu[1.0], slope=sl, p=p,
                             increasing=bool(mu[0.25] < mu[0.75] < mu[1.0]), **auc))
    return rows


def write(model, rank, frames, dm_rows=None):
    out = OUT / f"{model}_{rank}"; out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_parquet(out / f"ext_{name}.parquet", index=False)
    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(out / "darkmatter.csv", index=False)
    print("wrote", out, {k: len(v) for k, v in frames.items()}, flush=True)
