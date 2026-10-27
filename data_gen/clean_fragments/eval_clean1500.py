#!/usr/bin/env python3
"""Evaluate one trained clean1500 model: calibration, OOD detection, robustness, dark matter.

Plan: .docs/plans/TAXA_SHIFT_PLAN.md. Any of the four backbones (--backbone); the
representation scores use the backbone's Epinet feature function (mean-pooled final
hidden states), so kNN/Mahalanobis see the same features the Epinet does.

For each evaluation set (id_test, novel_genus, novel_family, nonbacterial, their
_err0.5/_err5 copies, and the dark matter contig windows) it computes, per
fragment:
  base       logits -> MSP, energy, entropy; mean-pooled embedding
  temp       base logits / T, T fit by NLL on ALL of val.csv (held-out genome
             blocks; never the training split)
  mc         MC dropout, p=0.1, K=10 -> entropy, epistemic (BALD), MSP
  epinet     K=16 index samples -> entropy, epistemic, MSP
  maha       class-conditional Mahalanobis (per-class means, tied covariance)
  knn        cosine distance to nearest training-fragment embedding (k=1)
Reference for maha/knn = embeddings of a fixed sample of TRAINING fragments.

Outputs results/clean1500_es/<backbone>_<rank>_s<seed>/:
  scores_<set>.parquet   one row per fragment, every score + prediction
  calibration.csv        accuracy / ECE per method, IN-VOCABULARY tiers only
  detection.csv          AUROC and FPR@95 per score, id_test vs each tier
  darkmatter.csv         per-sample trend over the 15 samples for every score
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from nn_proj.common.utils import compute_uncertainty, enable_mc_dropout
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithConvPrior
from nn_proj.models.epinet.feature_fns import NT_feature_fn, hyenaDNA_feature_fn

# hub id, eval max length (train length + 8, as for NT), feature fn, fp16 base pass
BACKBONES = {
    "NT_transformer": ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", 264, NT_feature_fn, True),
    "DNABERT2": ("zhihan1996/DNABERT-2-117M", 392, NT_feature_fn, False),
    "hyenaDNA": ("LongSafari/hyenadna-small-32k-seqlen-hf", 1512, hyenaDNA_feature_fn, False),
    "CARMANIA": ("MsAlEhR/carmania-160k-seqlen-human", 1512, NT_feature_fn, False),
}
FF = NT_feature_fn   # set in main()
# fp16 autocast for the fp32 model copies (MC dropout, Epinet, and the base pass when not .half()).
# CARMANIA was trained under fp16 autocast, and in fp32 its SDPA attention falls back to the
# slow math kernel (~6h per base pass, OOM at batch 64).
AUTOCAST = {"CARMANIA"}
AC = False           # set in main()
D = Path("/scratch/home/glh52/glm-epinet/DATA/clean1500")
DEV = "cuda"
TIERS = ["id_test", "novel_genus", "novel_family", "nonbacterial"]
# Scorpio gene_taxa task (--rank gene_taxa): ID test, taxa_out (same gene labels, unseen
# organisms; labels valid) and gene_out (unseen gene classes; its label column is a dummy,
# so it is never used for accuracy/ECE). Big tiers are capped as in scripts/eval_gene_taxa.sh.
GENE_D = Path("/scratch/home/glh52/glm-epinet/DATA/gene_taxa")
GENE_TIERS = {"id_test": "test", "taxa_out": "taxa_out", "gene_out": "gene_out"}
GENE_CAP = 25_000
# MC dropout perturbs only dropout layers that were active during fine-tuning (decided
# 2026-10-15). NT and hyenaDNA train with every dropout layer at 0.1 (train_base.py calls
# enable_mc_dropout before training) and CARMANIA has a single head dropout, so nothing is
# excluded for them; DNABERT-2 fine-tunes with attention-probability dropout 0.0.
MC_EXCLUDE = {"DNABERT2": (".attention.self.dropout",)}
GENE_ML = {"NT_transformer": 508, "DNABERT2": 758, "hyenaDNA": 3008, "CARMANIA": 3008}
NO_LABELS = {"gene_out"}
FPRS = [0.01, 0.05, 0.10]


def ece(conf, corr, bins=50):
    if len(conf) < bins: return np.nan
    o = np.argsort(conf); c, a = conf[o], corr[o]; e = np.linspace(0, len(c), bins + 1).astype(int)
    return sum((hi - lo) / len(c) * abs(a[lo:hi].mean() - c[lo:hi].mean()) for lo, hi in zip(e[:-1], e[1:]) if hi > lo)


def fpr95(id_s, ood_s):
    f, t, _ = roc_curve(np.r_[np.zeros(len(id_s)), np.ones(len(ood_s))], np.r_[id_s, ood_s])
    return float(f[np.searchsorted(t, 0.95)])


PASS_ID = [0]


def batches(tok, seqs, ml, bs):
    """tokenized batches; for long passes, logs progress (~every 5%) with elapsed time and ETA"""
    import time
    nb = (len(seqs) + bs - 1) // bs; step = max(1, nb // 20); t0 = time.time(); PASS_ID[0] += 1
    for j, i in enumerate(range(0, len(seqs), bs)):
        if nb >= 200 and j and j % step == 0:
            el = time.time() - t0
            print(f"    [pass {PASS_ID[0]}] {j}/{nb} batches ({100 * j / nb:.0f}%), {el / 60:.1f} min elapsed, "
                  f"ETA {el / j * (nb - j) / 60:.1f} min", flush=True)
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True, truncation=True, max_length=ml)
        yield {k: v.to(DEV) for k, v in b.items()}


@torch.no_grad()
def run_base(m, tok, seqs, ml, bs):
    E, G = [], []
    for b in batches(tok, seqs, ml, bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=AC):
            mu, pooled = FF(m, b)
        E.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
        G.append(mu.float().cpu())
    return torch.cat(E), torch.cat(G)


@torch.no_grad()
def run_sampled(m, tok, seqs, ml, bs, K, epinet):
    out = {"ent": [], "epi": [], "conf": [], "pred": []}
    for b in batches(tok, seqs, ml, bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=AC):
            st = m.wrapper(dict(b), n_index_samples=K, return_all=True) if epinet else \
                torch.stack([FF(m, b)[0] for _ in range(K)], 0)
        u = compute_uncertainty(st.float())
        out["ent"].append(u["normalized_total_uncertainty"].cpu()); out["epi"].append(u["normalized_epistemic_uncertainty"].cpu())
        out["conf"].append(u["max_confidence"].cpu()); out["pred"].append(u["predicted_class"].cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def load_weights(ck):
    """Checkpoint state dict: model.safetensors, or pytorch_model.bin (hyenaDNA saves .bin
    because of its tied weights)."""
    st = Path(ck) / "model.safetensors"
    return load_file(str(st)) if st.exists() else torch.load(Path(ck) / "pytorch_model.bin", map_location="cpu")


def fit_T(logits, y):
    lt = torch.zeros((), requires_grad=True); opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    lf = torch.nn.CrossEntropyLoss()
    def cl():
        opt.zero_grad(); l = lf(logits / lt.exp(), y); l.backward(); return l
    opt.step(cl); return float(lt.exp())


EXTRA_META = {}          # method settings recorded into meta.json by the column-only modes
LAPLACE_FIT, LAPLACE_TUNE, K_LAPLACE = 20_000, 5_000, 16
LAPLACE_PRIORS = (0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
CONFORMAL_ALPHA = 0.1


def ref_for_laplace(a, gene, tr):
    """Training fragments for the Laplace curvature fit (labels unused) -- the training split, never val/test."""
    if gene:
        return tr
    return pd.read_csv(D / "train.csv", usecols=["sequence", "taxid"])


def posthoc_columns(a, B, base_model, tok, ml, sets, S, val, label_of, lab2id, cfg, gene, train_df):
    """Last-layer diagonal Laplace (nn_proj.models.laplace) and split conformal prediction
    (nn_proj.common.conformal, LAC score 1 - p_true) on the base model, from one pass per set.

    Laplace: curvature from LAPLACE_FIT training fragments, rescaled to the full training set;
    prior precision chosen by predictive NLL on LAPLACE_TUNE validation fragments (held out
    from training). Conformal: threshold from the validation fragments, alpha = 0.1."""
    import datasets as hfds
    from nn_proj.common.datasets import prep_for_trainer
    from nn_proj.common.conformal import score_nonconformity, fit_conformal_threshold, predict_sets
    from nn_proj.models.laplace import LaplaceConfig, LaplaceSeqClassifier
    from nn_proj.models.laplace.laplace_head import fit_diagonal_ggn
    m = base_model(half=False)
    num = cfg.num_labels
    attr = [n for n, x in m.named_modules() if isinstance(x, torch.nn.Linear) and x.out_features == num][-1]
    def hf(df, n=None):
        y = [lab2id.get(int(l), -1) for l in label_of(df)]
        d = df.assign(labels=y)[["sequence", "labels"]]
        d = d[d.labels >= 0]
        if n: d = d.sample(min(n, len(d)), random_state=1)
        ds, coll = prep_for_trainer(hfds.Dataset.from_pandas(d.reset_index(drop=True)), tok, max_length=ml)
        return ds, coll
    train_df = train_df.sample(frac=1, random_state=2).reset_index(drop=True)       # fit order = random
    fit_ds, fit_coll = hf(train_df.head(LAPLACE_FIT))
    lcfg = LaplaceConfig(classifier_attr=attr, max_examples=None)
    with torch.autocast("cuda", dtype=torch.float16, enabled=AC):
        w_ggn, b_ggn, n_seen = fit_diagonal_ggn(m, fit_ds, fit_coll, lcfg, batch_size=a.bs)
    # curvature of the fitted subset, rescaled to the full training set: posterior precision = prior + N * mean GGN
    scale = len(train_df) / max(n_seen, 1); w_ggn = w_ggn * scale; b_ggn = b_ggn * scale
    yv = np.array([lab2id.get(int(l), -1) for l in label_of(val)]); keep = yv >= 0
    vt = val[keep].sample(min(LAPLACE_TUNE, int(keep.sum())), random_state=1)
    yt = torch.tensor([lab2id[int(l)] for l in label_of(vt)])
    def make(prior):
        return LaplaceSeqClassifier(m, 1.0 / (prior + w_ggn), 1.0 / (prior + b_ggn), classifier_attr=attr).to(DEV).eval()
    def logits_and_samples(lap, seqs, k=K_LAPLACE):
        G, ST = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=AC):
            for b in batches(tok, seqs, ml, a.bs):
                b.pop("token_type_ids", None) if B == "hyenaDNA" else None
                G.append(lap(**b)["logits"].float().cpu()); ST.append(lap.sample_logits(k).float().cpu())
        return torch.cat(G), torch.cat(ST, 1)
    # prior precision: minimum MC-predictive NLL on held-out validation fragments
    nll = {}
    for prior in LAPLACE_PRIORS:
        _, st = logits_and_samples(make(prior), vt.sequence.values, k=32)
        pm = st.softmax(-1).mean(0)
        nll[prior] = float(torch.nn.functional.nll_loss(torch.log(pm.clamp_min(1e-12)), yt))
    best = min(nll, key=nll.get); lap = make(best)
    print(f"  laplace: head {attr}, prior precision {best}, val NLL {nll}, fit {n_seen} scaled to {len(train_df)}", flush=True)
    # conformal threshold on validation fragments (exchangeable with id_test: held-out regions of training genomes)
    Gv, _ = logits_and_samples(lap, val.sequence.values[keep])
    qhat = fit_conformal_threshold(score_nonconformity(Gv.softmax(-1), torch.tensor(yv[keep])), alpha=CONFORMAL_ALPHA)
    print(f"  conformal: qhat {qhat:.4f} from {int(keep.sum())} val fragments", flush=True)
    EXTRA_META.update(laplace_prior_precision=best, laplace_val_nll={str(k): v for k, v in nll.items()},
                      laplace_fit_examples=n_seen, laplace_train_size=len(train_df), laplace_head=attr, laplace_k=K_LAPLACE,
                      conformal_alpha=CONFORMAL_ALPHA, conformal_qhat=float(qhat), conformal_score="lac", conformal_cal="val")
    for name, d in sets.items():
        G, ST = logits_and_samples(lap, d.sequence.values)
        u = compute_uncertainty(ST)
        sz = predict_sets(G.softmax(-1), qhat, method="score")
        labeled = name not in NO_LABELS and name != "darkmatter"     # dark matter windows have no taxon labels
        y = np.array([lab2id.get(int(l), -1) for l in label_of(d)]) if labeled else np.full(len(d), -1)
        hit = sz[torch.arange(len(d)), torch.as_tensor(np.clip(y, 0, None), dtype=torch.long)].numpy().astype(float)
        cov = np.where(y >= 0, hit, np.nan)
        S[name] = S[name].assign(lap_conf=u["max_confidence"].numpy(), lap_pred=u["predicted_class"].numpy(),
                                 lap_entropy=u["normalized_total_uncertainty"].numpy(),
                                 lap_epistemic=u["normalized_epistemic_uncertainty"].numpy(),
                                 cp_size=sz.sum(-1).numpy().astype(float), cp_covered=cov)
        print(f"  laplace+conformal pass {name}: {len(d)}", flush=True)
    del lap, m


def evidential_columns(a, B, HUB, cfg, ck_dir, tok, ml, sets, S):
    """Evidential model (nn_proj.models.evidential) from <ckpt>/evidential: Dirichlet mean probability
    (evid_conf/pred), its normalized entropy, and vacuity C/S."""
    from nn_proj.models.evidential import EvidentialConfig, EvidentialWrapper, HFEvidentialSeqClassifier
    d0 = ck_dir / "evidential"
    last = sorted(d0.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[-1]
    best = json.load(open(last / "trainer_state.json")).get("best_model_checkpoint")
    ck = Path(best) if best and Path(best).is_dir() else last
    ecfg = EvidentialConfig(num_classes=cfg.num_labels)          # defaults = the training defaults
    if B == "CARMANIA":
        from nn_proj.models.CARMANIA.model import CarmaniaForSequenceClassification
        base = CarmaniaForSequenceClassification(cfg)
    else:
        base = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg, trust_remote_code=True)
    if B == "DNABERT2":
        from nn_proj.common.utils import disable_dnabert2_flash_attn; disable_dnabert2_flash_attn()
    model = HFEvidentialSeqClassifier(EvidentialWrapper(base, ecfg))
    model.load_state_dict(load_weights(ck), strict=True); model = model.to(DEV).eval()
    EXTRA_META.update(evidential_checkpoint=str(ck), evidential_activation=ecfg.evidence_activation,
                      evidential_loss=ecfg.loss_type, evidential_annealing_step=ecfg.annealing_step)
    print(f"  evidential checkpoint: {ck}", flush=True)
    for name, d in sets.items():
        out = {"conf": [], "pred": [], "ent": [], "vac": []}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=AC):
            for b in batches(tok, d.sequence.values, ml, a.bs):
                b.pop("token_type_ids", None) if B == "hyenaDNA" else None
                ev = model(**b)["logits"].float(); alpha = ev + 1.0; Ssum = alpha.sum(-1, keepdim=True); p = alpha / Ssum
                u = compute_uncertainty(torch.log(p.clamp_min(1e-12)).unsqueeze(0))
                out["conf"].append(u["max_confidence"].cpu()); out["pred"].append(u["predicted_class"].cpu())
                out["ent"].append(u["normalized_total_uncertainty"].cpu()); out["vac"].append((cfg.num_labels / Ssum.squeeze(-1)).cpu())
        o = {k: torch.cat(v).numpy() for k, v in out.items()}
        S[name] = S[name].assign(evid_conf=o["conf"], evid_pred=o["pred"], evid_entropy=o["ent"], evid_vacuity=o["vac"])
        print(f"  evidential pass {name}: {len(d)}", flush=True)
    del model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", required=True, choices=["phylum", "family", "gene_taxa"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-ref", type=int, default=50_000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--skip-darkmatter", action="store_true")
    ap.add_argument("--error-sweep", action="store_true",
                    help="also score the 0.5%%/5%% error-injected copies (off: the benchmark uses clean fragments only)")
    ap.add_argument("--base-only", action="store_true",
                    help="score only the base pass (no error variants, MC, Epinet) into <out>/base_only and exit")
    ap.add_argument("--ckpt-root", default="checkpoints/clean1500_es")
    ap.add_argument("--out-root", default="results/clean1500_es")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows of every set")
    ap.add_argument("--backbone", default="NT_transformer", choices=list(BACKBONES))
    ap.add_argument("--mc-only", action="store_true",
                    help="recompute only the MC-dropout columns of an existing evaluation (reads and rewrites "
                         "<out>/scores_*.parquet and the summary CSVs; base and Epinet scores are kept as they are)")
    ap.add_argument("--epinet-only", action="store_true",
                    help="same, for the Epinet columns (e.g. after retraining the heads); base and MC scores are kept")
    ap.add_argument("--posthoc-only", action="store_true",
                    help="add last-layer Laplace (lap_*) and split-conformal (cp_*) columns to an existing evaluation")
    ap.add_argument("--darkmatter-only", action="store_true",
                    help="rescore only the dark matter windows with every method (base, MC, Epinet, Laplace, conformal), "
                         "e.g. after rebuilding them; the other sets' stored scores are reused unchanged")
    ap.add_argument("--extra-only", default="",
                    help="score one external window set NAME (--extra-csv) with every method, like --darkmatter-only; "
                         "labels from its <rank> column (class taxid, -1 = none) when present, else from taxid")
    ap.add_argument("--extra-csv", default="")
    ap.add_argument("--evidential-only", action="store_true",
                    help="add evidential (evid_*) columns from <ckpt>/evidential to an existing evaluation")
    a = ap.parse_args()
    global FF, AC
    B = a.backbone; HUB, ml, FF, HALF = BACKBONES[B]; AC = B in AUTOCAST
    ck_dir = Path(f"{a.ckpt_root}/{B}/{a.rank}/seed{a.seed}")
    def best_ck(d):
        # the checkpoint load_best_model_at_end selected (what train_clean1500.sh trains the Epinet on);
        # under early stopping the latest checkpoint is not necessarily the best one
        last = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))[-1]
        best = json.load(open(last / "trainer_state.json")).get("best_model_checkpoint")
        return Path(best) if best and Path(best).is_dir() else last
    base_ck = best_ck(ck_dir / "base")
    epi_ck = best_ck(ck_dir / "epinet") if (ck_dir / "epinet").is_dir() else None
    out = Path(f"{a.out_root}/{B}_{a.rank}_s{a.seed}"); out.mkdir(parents=True, exist_ok=True)

    GENE = a.rank == "gene_taxa"
    if GENE:
        ml = GENE_ML[B]
        if ml > 1000:
            a.bs = min(a.bs, 16)   # 3000-token sequences: batch 64 runs out of memory (CARMANIA OOMs at 32 in training)
        label_of = lambda df: df["labels"].astype(int).values          # labels are the gene classes
    else:
        lin = pd.read_csv(D / "full_basic_lineage.csv", index_col=0)
        s2r = dict(zip(lin.species.astype(int), lin[a.rank].astype(int)))
        label_of = lambda df: (df[a.rank].astype(int).values if a.rank in df.columns
                               else np.array([s2r.get(t, -1) for t in df.taxid]))
    if B == "CARMANIA":
        from nn_proj.models.CARMANIA.model import CarmaniaForSequenceClassification, CarmaniaSequenceClassificationConfig
        cfg = CarmaniaSequenceClassificationConfig.from_pretrained(base_ck)
        HUB = cfg.encoder_name or HUB   # human or multi-species (prok-genome) encoder, as trained
    else:
        cfg = transformers.AutoConfig.from_pretrained(base_ck, trust_remote_code=True)
    id2lab = {int(k): int(v) for k, v in cfg.id2label.items()}; lab2id = {v: k for k, v in id2lab.items()}
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True, model_max_length=ml,
                                                     padding_side="right", use_fast=True)
    if "InstaDeepAI" in HUB:
        tok.eos_token = tok.pad_token
    def base_model(half=HALF):
        m = CarmaniaForSequenceClassification(cfg) if B == "CARMANIA" else \
            transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg, trust_remote_code=True)
        m.load_state_dict(load_weights(base_ck), strict=True)
        if B == "DNABERT2":
            from nn_proj.common.utils import disable_dnabert2_flash_attn; disable_dnabert2_flash_attn()
        return m.to(DEV).eval().half() if half else m.to(DEV).eval()

    if GENE:
        sets = {}
        for k, f in GENE_TIERS.items():
            d = pd.read_csv(GENE_D / f"{f}.csv")
            sets[k] = d.sample(GENE_CAP, random_state=1) if len(d) > GENE_CAP else d
        # the exact 10% validation split train_base held out (same loader, seed and stratification),
        # so temperature is fit on data the model never trained on; reference = training part
        from nn_proj.common.datasets import load_local_dataset
        sp = load_local_dataset(str(GENE_D / "train.csv"), encode_labels=True).train_test_split(
            test_size=0.1, seed=a.seed, stratify_by_column="labels")
        names = sp["train"].features["labels"].names
        val, tr = (sp[x].to_pandas()[["sequence", "labels"]] for x in ("test", "train"))
        for x in (val, tr):
            x["labels"] = [int(names[i]) for i in x["labels"]]
        val = val.sample(min(len(val), 20_000), random_state=1)
        ref = tr.sample(a.n_ref, random_state=1)
        a.skip_darkmatter = True
    else:
        sets = {t: pd.read_csv(D / f"{t}.csv") for t in TIERS}
    for t in (TIERS if a.error_sweep and not a.base_only and not GENE else []):
        for r in ("0.5", "5"):
            sets[f"{t}_err{r}"] = pd.read_csv(D / f"{t}_err{r}.csv")
    if not a.skip_darkmatter:
        sets["darkmatter"] = pd.read_csv(D / "darkmatter_contig_windows.csv")
    if a.extra_only:
        sets[a.extra_only] = pd.read_csv(a.extra_csv)
    DM = a.darkmatter_only or bool(a.extra_only)            # score one set, reuse every other set's stored scores
    TGT = a.extra_only or "darkmatter"
    if not GENE:
        val = pd.read_csv(D / "val.csv")
        ref = pd.read_csv(D / "train.csv", usecols=["sequence", "taxid"]).sample(a.n_ref, random_state=1)
    if a.limit:
        sets = {k: v.head(a.limit) for k, v in sets.items()}; val = val.head(a.limit)

    if a.posthoc_only or a.evidential_only:
        T = json.load(open(out / "meta.json"))["T"]; S = {}
        for name, d in sets.items():
            old = pd.read_parquet(out / f"scores_{name}.parquet")
            meta = d.drop(columns=["sequence", "labels"], errors="ignore").reset_index(drop=True)
            assert len(old) == len(meta), f"{name}: {len(old)} stored rows vs {len(meta)} in the set"
            pre = ("lap_", "cp_") if a.posthoc_only else ("evid_",)
            S[name] = old.drop(columns=list(meta.columns) + [c for c in old.columns if c.startswith(pre)])
        if a.posthoc_only:
            posthoc_columns(a, B, base_model, tok, ml, sets, S, val, label_of, lab2id, cfg, GENE, ref_for_laplace(a, GENE, tr if GENE else None))
        else:
            evidential_columns(a, B, HUB, cfg, ck_dir, tok, ml, sets, S)
        for name, d in sets.items():
            meta = d.drop(columns=["sequence", "labels"], errors="ignore").reset_index(drop=True)
            pd.concat([meta, S[name]], axis=1).to_parquet(out / f"scores_{name}.parquet", index=False)
    elif a.mc_only or a.epinet_only:
        # reuse every other score from the existing evaluation, in the same row order
        pre = "mc" if a.mc_only else "epi"; redo = [f"{pre}_{c}" for c in ("conf", "pred", "entropy", "epistemic")]
        T = json.load(open(out / "meta.json"))["T"]; S = {}
        for name, d in sets.items():
            old = pd.read_parquet(out / f"scores_{name}.parquet")
            meta = d.drop(columns=["sequence", "labels"], errors="ignore").reset_index(drop=True)
            assert len(old) == len(meta), f"{name}: {len(old)} stored rows vs {len(meta)} in the set"
            S[name] = old.drop(columns=list(meta.columns) + redo)
        if a.mc_only:
            sm = base_model(half=False); enable_mc_dropout(sm, p=0.1, exclude=MC_EXCLUDE.get(B, ())); K, EPI = 10, False
        else:
            bm = base_model(half=False)
            vs = len(tok) if B == "CARMANIA" else cfg.vocab_size   # as in each backbone's train_epinet.py
            wrap = EpinetWrapper(bm, FF, EpinetConfig(num_classes=cfg.num_labels, include_inputs=True,
                                                      vocab_size=vs), epinet=MLPEpinetWithConvPrior)
            sm = HFEpinetSeqClassifier(wrap, k_train=8, k_eval=8).to(DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=AC):   # flash-attn needs fp16
                sm(**{k: v.to(DEV) for k, v in tok("ACGT", padding="max_length", max_length=ml, return_tensors="pt").items()})
            sm.load_state_dict(load_weights(epi_ck), strict=True); sm.eval(); K, EPI = 16, True
            print(f"  epinet checkpoint: {epi_ck}", flush=True)
        for name, d in sets.items():
            r = run_sampled(sm, tok, d.sequence.values, ml, a.bs, K, EPI)
            S[name] = S[name].assign(**{redo[0]: r["conf"], redo[1]: r["pred"], redo[2]: r["ent"], redo[3]: r["epi"]})
            meta = d.drop(columns=["sequence", "labels"], errors="ignore").reset_index(drop=True)
            pd.concat([meta, S[name]], axis=1).to_parquet(out / f"scores_{name}.parquet", index=False)
            print(f"  {pre} pass {name}: {len(d)}", flush=True)
        del sm
    else:
        m = base_model()
        _, Gv = run_base(m, tok, val.sequence.values, ml, a.bs)
        yv = torch.tensor([lab2id.get(int(l), -1) for l in label_of(val)])
        keep = yv >= 0
        T = fit_T(Gv[keep], yv[keep]); print(f"temperature (fit on {int(keep.sum())} val fragments) T = {T:.4f}", flush=True)
        R, _ = run_base(m, tok, ref.sequence.values, ml, a.bs)
        Rn = R.numpy().astype(np.float64); rlab = label_of(ref)
        cls = np.unique(rlab); MU = np.stack([Rn[rlab == c].mean(0) for c in cls])
        w, V = np.linalg.eigh(np.linalg.pinv(np.cov((Rn - MU[np.searchsorted(cls, rlab)]).T) + 1e-6 * np.eye(Rn.shape[1])))
        W = V * np.sqrt(np.clip(w, 0, None)); MUw = torch.from_numpy(MU @ W).float().to(DEV); Rd = R.to(DEV)

        S = {}
        todo = {TGT: sets[TGT]} if DM else sets
        for name, d in todo.items():
            E, G = run_base(m, tok, d.sequence.values, ml, a.bs)
            p = torch.softmax(G, -1); pT = torch.softmax(G / T, -1)
            df = pd.DataFrame({"base_conf": p.max(1).values.numpy(), "base_pred": p.argmax(1).numpy(),
                               "temp_conf": pT.max(1).values.numpy(),
                               "msp": -p.max(1).values.numpy(), "energy": -torch.logsumexp(G, 1).numpy(),
                               "entropy": compute_uncertainty(G)["normalized_total_uncertainty"].numpy()})
            Xw = torch.from_numpy(E.numpy().astype(np.float64) @ W).float().to(DEV)
            df["maha"] = np.concatenate([torch.cdist(Xw[i:i + 1024], MUw).pow(2).min(1).values.cpu().numpy() for i in range(0, len(Xw), 1024)])
            df["knn"] = np.concatenate([(-(E[i:i + 2048].to(DEV) @ Rd.T).max(1).values).cpu().numpy() for i in range(0, len(E), 2048)])
            S[name] = df
            print(f"  base pass {name}: {len(df)}", flush=True)
        if a.base_only:
            (out / "base_only").mkdir(exist_ok=True)
            for name, d in sets.items():
                meta = d.drop(columns=["sequence", "labels"], errors="ignore").reset_index(drop=True)
                pd.concat([meta, S[name]], axis=1).to_parquet(out / "base_only" / f"scores_{name}.parquet", index=False)
            json.dump({"T": T, "base": str(base_ck)}, open(out / "base_only" / "meta.json", "w"))
            return
        del m   # no empty_cache(): keep the freed memory reserved, or other users' jobs take it between phases

        mc = base_model(half=False); enable_mc_dropout(mc, p=0.1, exclude=MC_EXCLUDE.get(B, ()))
        for name, d in todo.items():
            r = run_sampled(mc, tok, d.sequence.values, ml, a.bs, 10, False)
            S[name] = S[name].assign(mc_conf=r["conf"], mc_pred=r["pred"], mc_entropy=r["ent"], mc_epistemic=r["epi"])
        del mc   # no empty_cache(): keep the freed memory reserved, or other users' jobs take it between phases
        bm = base_model(half=False)
        vs = len(tok) if B == "CARMANIA" else cfg.vocab_size   # as in each backbone's train_epinet.py
        wrap = EpinetWrapper(bm, FF, EpinetConfig(num_classes=cfg.num_labels, include_inputs=True,
                                                  vocab_size=vs), epinet=MLPEpinetWithConvPrior)
        em = HFEpinetSeqClassifier(wrap, k_train=8, k_eval=8).to(DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=AC):   # flash-attn needs fp16
            em(**{k: v.to(DEV) for k, v in tok("ACGT", padding="max_length", max_length=ml, return_tensors="pt").items()})
        em.load_state_dict(load_weights(epi_ck), strict=True); em.eval()
        for name, d in todo.items():
            r = run_sampled(em, tok, d.sequence.values, ml, a.bs, 16, True)
            S[name] = S[name].assign(epi_conf=r["conf"], epi_pred=r["pred"], epi_entropy=r["ent"], epi_epistemic=r["epi"])
            meta = d.drop(columns=["sequence", "labels"], errors="ignore").reset_index(drop=True)
            pd.concat([meta, S[name]], axis=1).to_parquet(out / f"scores_{name}.parquet", index=False)
        del em
        if DM:
            # Laplace + conformal for the new windows (same fit, prior selection and val threshold as --posthoc-only)
            posthoc_columns(a, B, base_model, tok, ml, todo, S, val, label_of, lab2id, cfg, GENE, ref_for_laplace(a, GENE, None))
            dmeta = sets[TGT].drop(columns=["sequence", "labels"], errors="ignore").reset_index(drop=True)
            pd.concat([dmeta, S[TGT]], axis=1).to_parquet(out / f"scores_{TGT}.parquet", index=False)
            for name, d in sets.items():          # every other set: stored scores, unchanged
                if name == TGT: continue
                old = pd.read_parquet(out / f"scores_{name}.parquet")
                meta = d.drop(columns=["sequence", "labels"], errors="ignore").reset_index(drop=True)
                assert len(old) == len(meta), f"{name}: {len(old)} stored rows vs {len(meta)} in the set"
                S[name] = old.drop(columns=list(meta.columns))

    # ---- calibration on in-vocabulary fragments
    rows = []
    for name in [k for k in sets if k != "darkmatter" and k not in NO_LABELS]:
        d, sc = sets[name], S[name]
        y = np.array([lab2id.get(int(l), -1) for l in label_of(d)]); iv = y >= 0
        if iv.mean() < 0.5: continue
        for meth, cc, pc in [m for m in [("softmax", "base_conf", "base_pred"), ("temperature", "temp_conf", "base_pred"),
                             ("mc_dropout", "mc_conf", "mc_pred"), ("epinet", "epi_conf", "epi_pred"),
                             ("laplace", "lap_conf", "lap_pred"), ("evidential", "evid_conf", "evid_pred")] if m[1] in sc]:
            corr = (sc[pc].values[iv] == y[iv]).astype(float); conf = sc[cc].values[iv]
            rows.append(dict(set=name, method=meth, n=int(iv.sum()), oov=float(1 - iv.mean()),
                             accuracy=corr.mean(), ece=ece(conf, corr)))
    pd.DataFrame(rows).to_csv(out / "calibration.csv", index=False)

    # ---- split conformal: coverage and set size per set (calibrated on val; alpha in meta.json)
    if "cp_size" in S["id_test"]:
        rows = []
        for name in [k for k in sets if k != "darkmatter"]:
            sc = S[name]; cov = sc["cp_covered"].values
            rows.append(dict(set=name, n=len(sc), mean_set_size=sc["cp_size"].mean(), median_set_size=sc["cp_size"].median(),
                             frac_singleton=(sc["cp_size"] == 1).mean(), frac_empty=(sc["cp_size"] == 0).mean(),
                             # coverage only where the labels are mostly in-vocabulary (same rule as calibration):
                             # e.g. the few non-bacterial reads whose label collides with a training class are not ID
                             coverage=np.nanmean(cov) if np.isfinite(cov).mean() >= 0.5 else np.nan,
                             n_labeled=int(np.isfinite(cov).sum())))
        pd.DataFrame(rows).to_csv(out / "conformal.csv", index=False)

    # ---- detection: id_test vs each tier (and matched error level)
    SC = [s for s in ["msp", "energy", "entropy", "mc_entropy", "mc_epistemic", "epi_entropy", "epi_epistemic",
                      "lap_entropy", "lap_epistemic", "cp_size", "evid_entropy", "evid_vacuity", "maha", "knn"]
          if s in S["id_test"]]
    rows = []
    for suf in ["", "_err0.5", "_err5"] if a.error_sweep else [""]:
        idn = f"id_test{suf}"
        for t in [k for k in sets if k not in ("id_test", "darkmatter") and "_err" not in k]:
            for s in SC:
                i_, o_ = S[idn][s].values, S[f"{t}{suf}"][s].values
                rows.append(dict(error=suf or "0", tier=t, score=s,
                                 auroc=roc_auc_score(np.r_[np.zeros(len(i_)), np.ones(len(o_))], np.r_[i_, o_]),
                                 fpr95=fpr95(i_, o_)))
    pd.DataFrame(rows).to_csv(out / "detection.csv", index=False)

    # ---- dark matter: per-sample trend over the 15 samples
    if "darkmatter" in S:
        dm = pd.concat([sets["darkmatter"][["up", "sample", "contig", "cov"]].reset_index(drop=True), S["darkmatter"]], axis=1)
        rows = []
        for s in SC:
            si = S["id_test"][s].values
            variants = {"mean": dm.assign(v=dm[s])}
            for f in FPRS:
                variants[f"flag@{f:.0%}"] = dm.assign(v=(dm[s] > np.quantile(si, 1 - f)).astype(float))
            for vname, x in variants.items():
                rep = x.groupby(["up", "sample"]).v.mean().reset_index()
                sl, _, r, p, _ = stats.linregress(rep.up, rep.v)
                lo, hi = rep[rep.up == rep.up.min()].v, rep[rep.up == rep.up.max()].v
                dd = (hi.mean() - lo.mean()) / np.sqrt((lo.var(ddof=1) + hi.var(ddof=1)) / 2)
                rows.append(dict(score=s, stat=vname, slope=sl, r2=r * r, cohen_d=dd, p=p,
                                 **{f"up{u}": v for u, v in rep.groupby("up").v.mean().items()}))
            rows.append(dict(score=s, stat="auroc_id_vs_up", **{
                f"up{u}": roc_auc_score(np.r_[np.zeros(len(si)), np.ones((dm.up == u).sum())], np.r_[si, dm[s][dm.up == u]])
                for u in sorted(dm.up.unique())}))
        pd.DataFrame(rows).to_csv(out / "darkmatter.csv", index=False)
    mp = out / "meta.json"; md = json.load(open(mp)) if mp.exists() else {}
    md.update({"T": T, "base": str(base_ck)})
    if epi_ck is not None and not (a.posthoc_only or a.evidential_only): md["epinet"] = str(epi_ck)
    md.update(EXTRA_META)
    json.dump(md, open(mp, "w"), indent=1)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
