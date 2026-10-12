#!/usr/bin/env python3
"""GC path check for the clean1500 NT models on dark matter.

Dark matter GC rises with UP, so a GC-dependent score could produce (or hide) the
UP trend. Three checks, per rank, for kNN and Mahalanobis:
  1. In clean id_test (no novelty), does the score depend on GC? Raw and within
     phylum (score and GC centred on their phylum means).
  2. GC-only prediction: give each dark matter window the id_test flag rate of its
     GC bin, average per sample, and regress on UP. This is the trend GC alone
     would produce through the score's GC dependence in ID data.
  3. GC-reweighted trend: reweight each sample's windows so its GC histogram
     matches the pooled dark matter histogram, then recompute the flagged-fraction
     slope. If the trend survives, it isn't carried by sample-level GC shifts.
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
D = "/scratch/home/glh52/glm-epinet/DATA/clean1500"
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--sub", default="",
    help="subdirectory of the run holding the score parquets, e.g. base_only"); a = ap.parse_args()
gc = lambda s: (s.count("G") + s.count("C")) / max(len(s), 1)
dmseq = pd.read_csv(f"{D}/darkmatter_contig_windows.csv", usecols=["up", "sample", "sequence"])
dm_gc = dmseq.sequence.map(gc).values
ids = pd.read_csv(f"{D}/id_test.csv", usecols=["sequence", "taxid"]); id_gc = ids.sequence.map(gc).values
lin = pd.read_csv(f"{D}/full_basic_lineage.csv", index_col=0)
id_phy = ids.taxid.map(dict(zip(lin.species, lin.phylum))).values
tr = pd.read_csv(f"{D}/train.csv", usecols=["sequence"]).sample(20000, random_state=0).sequence.map(gc)
print(f"GC: train {tr.mean():.4f} | id_test {id_gc.mean():.4f} | dark matter by UP "
      + " ".join(f"{u}:{dm_gc[dmseq.up.values == u].mean():.4f}" for u in (0.25, 0.75, 1.0)))
s_gc = pd.DataFrame({"up": dmseq.up, "sample": dmseq["sample"], "gc": dm_gc}).groupby(["up", "sample"]).gc.mean()
print(f"sample-level corr(UP, mean GC) r = {stats.pearsonr(s_gc.index.get_level_values(0), s_gc.values)[0]:+.3f}")
edges = np.quantile(np.concatenate([dm_gc, id_gc]), np.linspace(0, 1, 21)); edges[0], edges[-1] = -1, 2
dbin, ibin = np.digitize(dm_gc, edges) - 1, np.digitize(id_gc, edges) - 1
pooled = np.bincount(dbin, minlength=20) / len(dbin)
rows = []
for rank in ["family", "phylum"]:
    d = f"results/clean1500/NT_transformer_{rank}_s{a.seed}" + (f"/{a.sub}" if a.sub else "")
    dm = pd.read_parquet(f"{d}/scores_darkmatter.parquet"); idt = pd.read_parquet(f"{d}/scores_id_test.parquet")
    for sc in ["knn", "maha"]:
        x = idt[sc].values
        rho, p = stats.spearmanr(id_gc, x)
        df = pd.DataFrame({"x": x, "g": id_gc, "ph": id_phy})
        rx = df.x - df.groupby("ph").x.transform("mean"); rg = df.g - df.groupby("ph").g.transform("mean")
        rho_w, p_w = stats.spearmanr(rg, rx)
        q = pd.qcut(id_gc, 5, labels=False); qm = [x[q == k].mean() for k in range(5)]
        print(f"\n##### {rank} / {sc}")
        print(f"  id_test: rho(score, GC) {rho:+.3f} (p={p:.1e}); within phylum {rho_w:+.3f} (p={p_w:.1e})")
        print(f"  id_test score by GC fifth (low->high): " + " ".join(f"{v:.4g}" for v in qm))
        rho_d, _ = stats.spearmanr(dm_gc, dm[sc].values)
        print(f"  dark matter windows: rho(score, GC) {rho_d:+.3f}")
        for f in (0.01, 0.05, 0.10):
            T = np.quantile(x, 1 - f); fl_id = x > T; fl_dm = (dm[sc].values > T).astype(float)
            rate = np.array([fl_id[ibin == b].mean() if (ibin == b).sum() >= 50 else f for b in range(20)])
            t = pd.DataFrame({"up": dm.up, "sample": dm["sample"], "fl": fl_dm, "pred": rate[dbin], "b": dbin})
            obs = t.groupby(["up", "sample"]).fl.mean().reset_index()
            pr = t.groupby(["up", "sample"]).pred.mean().reset_index()
            def rw(g):   # reweight to the pooled GC histogram
                h = np.bincount(g.b, minlength=20) / len(g); w = np.where(h[g.b] > 0, pooled[g.b] / h[g.b], 0)
                return np.average(g.fl, weights=w)
            rwt = t.groupby(["up", "sample"])[["fl", "b"]].apply(rw).rename("fl").reset_index()
            out = {}
            for name, z, col in [("observed", obs, "fl"), ("GC-only prediction", pr, "pred"), ("GC-reweighted", rwt, "fl")]:
                sl, _, _, pv, _ = stats.linregress(z.up, z[col]); m = z.groupby("up")[col].mean()
                out[name] = (sl, pv); rows.append(dict(rank=rank, score=sc, fpr=f, kind=name, slope=sl, p=pv,
                                                       f25=m[0.25], f75=m[0.75], f100=m[1.0]))
                print(f"  FPR {f:>4.0%} {name:20s} {m[0.25]:.4f} {m[0.75]:.4f} {m[1.0]:.4f}  slope {sl:+.4f} p={pv:.1e}")
pd.DataFrame(rows).to_csv(f"results/clean1500/gc_path_s{a.seed}.csv", index=False)
