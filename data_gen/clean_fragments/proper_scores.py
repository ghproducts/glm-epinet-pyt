#!/usr/bin/env python3
"""Proper scoring rules, classwise calibration and K-convergence for one clean1500 / gene_taxa model.

eval_clean1500.py stores only the top-1 confidence and prediction per method, which is enough for
accuracy, top-label ECE and detection but not for NLL, Brier or classwise ECE (reviewer 2, point 6).
This pass re-scores every LABELED set (>= 50% of fragments in the training label vocabulary) with
each method and writes, per fragment:

  proper_<set>.parquet   label y (class index, -1 = out of vocabulary) and, for each method
                         m in {base, temp, mc, epi, lap}: m_conf, m_pred, m_ptrue (probability of
                         the true class), m_sq (sum_k p_k^2, so Brier = sq - 2*ptrue + 1),
                         m_ent / m_epi (normalized total / epistemic entropy)
  classwise.csv          classwise ECE (Kull et al. 2019; 15 equal-width bins per class, mean over
                         classes) per set and method, from streaming per-class reliability counts
  kconv.csv              K-convergence (reviewer 2, point 6): KCONV_N fragments per set, K_MAX draws
                         of MC dropout / Epinet / Laplace, metrics on nested prefixes K in K_GRID

Methods and settings are those of eval_clean1500.py (MC p=0.1 with the DNABERT2 exclusion, K=10;
Epinet K=16; Laplace prior and curvature fit as recorded in meta.json, K=16; temperature from
meta.json). The stochastic methods are redrawn here, so their accuracy can differ from the stored
scores_*.parquet values by sampling noise; scores_*.parquet is left untouched.

    python data_gen/clean_fragments/proper_scores.py --backbone NT_transformer --rank family --seed 1
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_clean1500 as E  # noqa: E402
from nn_proj.common.utils import enable_mc_dropout  # noqa: E402
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithConvPrior  # noqa: E402

K = {"mc": 10, "epi": 16, "lap": 16}
CW_BINS = 15
import os, time
KCONV_N = int(os.environ.get("KCONV_N", 500))         # fragments per set for K-convergence (phylum / family models)
K_MAX, K_GRID = 64, [1, 2, 4, 8, 10, 16, 32, 64]
T0 = time.time()


def log(msg): print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def summarize(P, logC):
    """P: [K, B, C] probabilities -> per-example stats of the mean predictive."""
    pm = P.mean(0)
    H = -(pm * pm.clamp_min(1e-12).log()).sum(-1)
    Hk = -(P * P.clamp_min(1e-12).log()).sum(-1).mean(0)
    return pm, H / logC, (H - Hk).clamp_min(0) / logC


class Classwise:
    """Streaming per-class reliability counts: count, sum p, sum 1[y=k] in CW_BINS equal-width bins."""
    def __init__(self, C):
        self.n = torch.zeros(C, CW_BINS, dtype=torch.float64); self.p = torch.zeros_like(self.n); self.y = torch.zeros_like(self.n)

    def add(self, pm, y):
        keep = y >= 0
        pm, y = pm[keep].double().cpu(), y[keep].cpu()
        if len(y) == 0: return
        C = pm.shape[1]
        b = (pm * CW_BINS).long().clamp_(max=CW_BINS - 1)                    # [B, C]
        idx = (torch.arange(C).unsqueeze(0) * CW_BINS + b).reshape(-1)
        onehot = torch.nn.functional.one_hot(y, C).double().reshape(-1)
        self.n.view(-1).index_add_(0, idx, torch.ones_like(onehot)); self.p.view(-1).index_add_(0, idx, pm.reshape(-1))
        self.y.view(-1).index_add_(0, idx, onehot)

    def ece(self):
        N = self.n.sum(1, keepdim=True).clamp_min(1)
        gap = (self.y - self.p).abs()                                        # |sum y - sum p| = n * |acc - conf|
        per = (gap / N).sum(1)
        present = self.y.sum(1) > 0
        return float(per.mean()), float(per[present].mean()), int(present.sum())


def ece_tl(conf, corr, bins=15):
    e = np.linspace(0, 1, bins + 1); i = np.clip(np.digitize(conf, e[1:-1]), 0, bins - 1)
    return float(sum((i == b).mean() * abs(corr[i == b].mean() - conf[i == b].mean()) for b in range(bins) if (i == b).any()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True, choices=["phylum", "family", "gene_taxa"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backbone", required=True, choices=list(E.BACKBONES))
    ap.add_argument("--ckpt-root", default="checkpoints/clean1500_es")
    ap.add_argument("--out-root", default="results/clean1500_es")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-kconv", action="store_true")
    a = ap.parse_args()
    B = a.backbone; HUB, ml, E.FF, HALF = E.BACKBONES[B]
    # fp16 autocast for every sampled pass except hyenaDNA (its FFT long convolution stays fp32); the GPUs are
    # shared, and fp32 MC dropout over ~30k fragments x K passes is otherwise many hours per model
    AC = E.AC = B != "hyenaDNA"
    ck_dir = Path(f"{a.ckpt_root}/{B}/{a.rank}/seed{a.seed}")
    out = Path(f"{a.out_root}/{B}_{a.rank}_s{a.seed}")
    meta = json.load(open(out / "meta.json"))
    base_ck, epi_ck, T, prior = Path(meta["base"]), Path(meta["epinet"]), float(meta["T"]), float(meta["laplace_prior_precision"])
    GENE = a.rank == "gene_taxa"
    if GENE:
        ml = E.GENE_ML[B]
        if ml > 1000: a.bs = min(a.bs, 16)
        label_of = lambda df: df["labels"].astype(int).values
    else:
        lin = pd.read_csv(E.D / "full_basic_lineage.csv", index_col=0)
        s2r = dict(zip(lin.species.astype(int), lin[a.rank].astype(int)))
        label_of = lambda df: np.array([s2r.get(t, -1) for t in df.taxid])
    if B == "CARMANIA":
        from nn_proj.models.CARMANIA.model import CarmaniaForSequenceClassification, CarmaniaSequenceClassificationConfig
        cfg = CarmaniaSequenceClassificationConfig.from_pretrained(base_ck); HUB = cfg.encoder_name or HUB
    else:
        cfg = transformers.AutoConfig.from_pretrained(base_ck, trust_remote_code=True)
    id2lab = {int(k): int(v) for k, v in cfg.id2label.items()}; lab2id = {v: k for k, v in id2lab.items()}
    C = cfg.num_labels; logC = math.log(C)
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True, model_max_length=ml, padding_side="right", use_fast=True)
    if "InstaDeepAI" in HUB: tok.eos_token = tok.pad_token

    def base_model(half=HALF):
        m = CarmaniaForSequenceClassification(cfg) if B == "CARMANIA" else \
            transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg, trust_remote_code=True)
        m.load_state_dict(E.load_weights(base_ck), strict=True)
        if B == "DNABERT2":
            from nn_proj.common.utils import disable_dnabert2_flash_attn; disable_dnabert2_flash_attn()
        return m.to(E.DEV).eval().half() if half else m.to(E.DEV).eval()

    # the same evaluation sets, in the same row order, as eval_clean1500.py
    if GENE:
        sets = {}
        for k, f in E.GENE_TIERS.items():
            d = pd.read_csv(E.GENE_D / f"{f}.csv"); sets[k] = d.sample(E.GENE_CAP, random_state=1) if len(d) > E.GENE_CAP else d
        from nn_proj.common.datasets import load_local_dataset
        sp = load_local_dataset(str(E.GENE_D / "train.csv"), encode_labels=True).train_test_split(test_size=0.1, seed=a.seed, stratify_by_column="labels")
        names = sp["train"].features["labels"].names
        tr = sp["train"].to_pandas()[["sequence", "labels"]]; tr["labels"] = [int(names[i]) for i in tr["labels"]]
        lap_ref = tr
    else:
        sets = {t: pd.read_csv(E.D / f"{t}.csv") for t in E.TIERS}
        lap_ref = pd.read_csv(E.D / "train.csv", usecols=["sequence", "taxid"])
    if a.limit: sets = {k: v.head(a.limit) for k, v in sets.items()}
    Y = {k: np.array([lab2id.get(int(l), -1) for l in label_of(d)]) if k not in E.NO_LABELS else np.full(len(d), -1)
         for k, d in sets.items()}
    labeled = [k for k in sets if (Y[k] >= 0).mean() >= 0.5]
    print(f"{B} {a.rank} s{a.seed}: labeled sets {labeled}; T={T:.4f}, Laplace prior {prior}", flush=True)

    # ---- models
    def make_mc():
        m = base_model(half=False); enable_mc_dropout(m, p=0.1, exclude=E.MC_EXCLUDE.get(B, ())); return m

    def make_epi():
        bm = base_model(half=False); vs = len(tok) if B == "CARMANIA" else cfg.vocab_size
        wrap = EpinetWrapper(bm, E.FF, EpinetConfig(num_classes=C, include_inputs=True, vocab_size=vs), epinet=MLPEpinetWithConvPrior)
        em = HFEpinetSeqClassifier(wrap, k_train=8, k_eval=8).to(E.DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=AC):
            em(**{k: v.to(E.DEV) for k, v in tok("ACGT", padding="max_length", max_length=ml, return_tensors="pt").items()})
        em.load_state_dict(E.load_weights(epi_ck), strict=True); return em.eval()

    def make_lap():
        import datasets as hfds
        from nn_proj.common.datasets import prep_for_trainer
        from nn_proj.models.laplace import LaplaceConfig, LaplaceSeqClassifier
        from nn_proj.models.laplace.laplace_head import fit_diagonal_ggn
        m = base_model(half=False)
        attr = [n for n, x in m.named_modules() if isinstance(x, torch.nn.Linear) and x.out_features == C][-1]
        tdf = lap_ref.sample(frac=1, random_state=2).reset_index(drop=True)      # as posthoc_columns: fit order = random
        y = [lab2id.get(int(l), -1) for l in label_of(tdf.head(E.LAPLACE_FIT))]
        d = tdf.head(E.LAPLACE_FIT).assign(labels=y)[["sequence", "labels"]]; d = d[d.labels >= 0]
        ds, coll = prep_for_trainer(hfds.Dataset.from_pandas(d.reset_index(drop=True)), tok, max_length=ml)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AC):
            w, b, n = fit_diagonal_ggn(m, ds, coll, LaplaceConfig(classifier_attr=attr, max_examples=None), batch_size=a.bs)
        s = len(tdf) / max(n, 1)
        return LaplaceSeqClassifier(m, 1.0 / (prior + w * s), 1.0 / (prior + b * s), classifier_attr=attr).to(E.DEV).eval()

    @torch.no_grad()
    def probs(kind, model, seqs, k):
        """yields [k, B, C] probabilities batch by batch"""
        for b in E.batches(tok, seqs, ml, a.bs):
            with torch.autocast("cuda", dtype=torch.float16, enabled=AC):
                if kind == "base":                      # base and temperature from one forward pass
                    G = E.FF(model, b)[0].float()
                    P = torch.stack([G.softmax(-1), (G / T).softmax(-1)])
                elif kind == "mc":
                    P = torch.stack([E.FF(model, b)[0].float().softmax(-1) for _ in range(k)])
                elif kind == "epi":
                    P = model.wrapper(dict(b), n_index_samples=k, return_all=True).float().softmax(-1)
                else:
                    b.pop("token_type_ids", None) if B == "hyenaDNA" else None
                    model(**b); P = model.sample_logits(k).float().softmax(-1)
            yield P.float()

    rows_cw, cols = [], {k: {} for k in labeled}
    kc_sets = {k: sets[k].sample(min(KCONV_N, len(sets[k])), random_state=3) for k in sets if k != "darkmatter"}
    kc = {}
    if GENE: a.skip_kconv = True      # 3 kb sequences: K-convergence is run on the phylum / family models only
    for kind, maker in [("base", lambda: base_model()), ("mc", make_mc), ("epi", make_epi), ("lap", make_lap)]:
        model = maker(); log(f"{kind}: model ready")
        outs = ["base", "temp"] if kind == "base" else [kind]
        for name in labeled:
            y = torch.as_tensor(Y[name]); cw = {o: Classwise(C) for o in outs}
            acc = {o: {c: [] for c in ("conf", "pred", "ptrue", "sq", "ent", "epi")} for o in outs}; i0 = 0
            for P in probs(kind, model, sets[name].sequence.values, K.get(kind, 1)):
                P = P.to(E.DEV); yb = y[i0:i0 + P.shape[1]].to(E.DEV); i0 += P.shape[1]
                for j, o in enumerate(outs):
                    pm, ent, epi = summarize(P[j:j + 1] if kind == "base" else P, logC); A = acc[o]
                    A["conf"].append(pm.max(1).values.cpu()); A["pred"].append(pm.argmax(1).cpu())
                    A["ptrue"].append(torch.where(yb >= 0, pm.gather(1, yb.clamp_min(0).unsqueeze(1)).squeeze(1), torch.full_like(ent, float("nan"))).cpu())
                    A["sq"].append((pm * pm).sum(1).cpu()); A["ent"].append(ent.cpu()); A["epi"].append(epi.cpu())
                    cw[o].add(pm, yb)
            for o in outs:
                for c, v in acc[o].items(): cols[name][f"{o}_{c}"] = torch.cat(v).numpy()
                e_all, e_present, n_present = cw[o].ece()
                rows_cw.append(dict(set=name, method=o, classwise_ece=e_all, classwise_ece_present=e_present, n_classes=C, n_classes_present=n_present))
                iv = Y[name] >= 0
                log(f"{o} {name}: acc {(cols[name][f'{o}_pred'][iv] == Y[name][iv]).mean():.4f} classwise ECE {e_all:.5f} (present classes {e_present:.4f})")
        if kind in ("mc", "epi", "lap") and not a.skip_kconv:
            for name, d in kc_sets.items():
                st = torch.cat([P.cpu() for P in probs(kind, model, d.sequence.values, K_MAX)], 1)    # [K_MAX, N, C]
                kc[(kind, name)] = (st.half(), np.array([lab2id.get(int(l), -1) for l in label_of(d)]) if name not in E.NO_LABELS else np.full(len(d), -1))
            log(f"{kind}: K-convergence draws done")
        del model
    for name in labeled:
        pd.DataFrame({"y": Y[name], **cols[name]}).to_parquet(out / f"proper_{name}.parquet", index=False)
    pd.DataFrame(rows_cw).to_csv(out / "classwise.csv", index=False)

    if kc:
        rows = []
        for (kind, name), (st, y) in kc.items():
            for k in K_GRID:
                P = st[:k].float(); pm, ent, epi = summarize(P, logC); iv = y >= 0
                r = dict(method=kind, set=name, K=k, n=len(y), mean_ent=float(ent.mean()), mean_epi=float(epi.mean()))
                if iv.mean() >= 0.5:
                    yy = torch.as_tensor(y[iv]); p = pm[iv]; pt = p.gather(1, yy.unsqueeze(1)).squeeze(1)
                    corr = (p.argmax(1) == yy).numpy().astype(float); conf = p.max(1).values.numpy()
                    r.update(accuracy=corr.mean(), ece_15=ece_tl(conf, corr), ece_50em=E.ece(conf, corr, 50),
                             nll=float(-pt.clamp_min(1e-12).log().mean()), brier=float(((p * p).sum(1) - 2 * pt + 1).mean()))
                if name != "id_test" and ("id_test" in [n for (kk, n) in kc if kk == kind]):
                    sti, _ = kc[(kind, "id_test")]; pmi, enti, epii = summarize(sti[:k].float(), logC)
                    lab = np.r_[np.zeros(len(enti)), np.ones(len(ent))]
                    r.update(auroc_entropy=roc_auc_score(lab, np.r_[enti.numpy(), ent.numpy()]),
                             auroc_epistemic=roc_auc_score(lab, np.r_[epii.numpy(), epi.numpy()]))
                rows.append(r)
        pd.DataFrame(rows).to_csv(out / "kconv.csv", index=False)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
