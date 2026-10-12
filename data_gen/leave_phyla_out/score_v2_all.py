#!/usr/bin/env python3
"""Every method x every tier on the v2 (id_test-clean) leave-phyla-out set.

Models: fine-tuned GLM (post-hoc scores), a from-scratch 1D-CNN, and a k-mer
Random Forest. The latter two share no pretraining with the GLM, so they
separate "this task is hard" from "this representation is bad".

Tiers, per model (its 5 phyla define what counts as in-label):
  id_test    training species, unseen reads          -> in-distribution
  near_id    id_novel_genus, in-label phyla          -> novel genus
  near_ood   id_novel_genus, other bacterial phyla   -> novel phylum
  far_ood    ood_nonbacterial                        -> novel superkingdom

Comparisons reported (AUROC, >0.5 = correctly flags the righthand tier):
  idtest_v_nearid   can it see genus-level novelty?
  idtest_v_nearood  genus + phylum novelty vs in-distribution
  idtest_v_farood   the easiest possible case
  nearid_v_nearood  the phylum-novelty step alone (v1's headline)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "label_noise"))
import cnn_scratch
from cnn_scratch import SmallCNN

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
V2 = Path("data_gen/leave_phyla_out_v2")  # overridden by --root


def load_ckpt(d):
    cfg = transformers.AutoConfig.from_pretrained(str(d), trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                       trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(d) / "model.safetensors")), strict=True)
    return m.to(DEV).eval().half()


@torch.no_grad()
def fl(m, tok, seqs, L, bs=512):
    bs = max(4, int(bs * 48 / max(L, 48)))   # keep tokens-per-batch roughly constant
    fs, ls = [], []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        o = m(**b, output_hidden_states=True)
        h = o.hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
        fs.append(((h * mk).sum(1) / mk.sum(1)).float().cpu()); ls.append(o.logits.float().cpu())
    return torch.cat(fs).numpy(), torch.cat(ls).numpy()


def maha_fit(f, y):
    cls = np.unique(y); mus = np.stack([f[y == c].mean(0) for c in cls])
    cen = np.concatenate([f[y == c] - mus[k] for k, c in enumerate(cls)])
    return mus, np.linalg.pinv(np.cov(cen.T) + 1e-6 * np.eye(f.shape[1]))


def maha(mus, P, f):
    return np.stack([np.einsum("ij,jk,ik->i", f - m, P, f - m) for m in mus]).min(0)


def knn(tr, f, k=50):
    a = tr / (np.linalg.norm(tr, axis=1, keepdims=True) + 1e-9)
    b = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    o = np.empty(len(b))
    for i in range(0, len(b), 2048):
        o[i:i + 2048] = -np.partition(b[i:i + 2048] @ a.T, -k, axis=1)[:, -k]
    return o


def bald(P):
    pb = P.mean(0); ent = lambda q: -(q * np.log(q + 1e-12)).sum(-1)
    t = ent(pb); a = ent(P).mean(0); return t, t - a, pb


def auc(a, b):
    return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--n-train-feat", type=int, default=6000)
    ap.add_argument("--rf-train", type=int, default=20000)
    ap.add_argument("--cnn-epochs", type=int, default=12)
    ap.add_argument("--skip-glm", action="store_true")
    ap.add_argument("--root", type=Path, default=None, help="eval build dir (default v2)")
    ap.add_argument("--tier-len", default="150bp")
    ap.add_argument("--ckpt-root", type=Path, default=Path("checkpoints/lpo_v2"))
    ap.add_argument("--out", type=Path, default=Path("results/leave_phyla_out_v2.csv"))
    args = ap.parse_args()

    global V2
    if args.root is not None:
        V2 = args.root
    design = pd.read_csv(V2 / "design.csv")
    tiers = {t: pd.read_csv(V2 / f"eval_{t}_{args.tier_len}.csv") for t in ["id_test", "near", "far"]}
    for d in tiers.values():
        d["true_phylum"] = d.true_phylum.astype(str)
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token

    rows = []
    for i in range(10):
        phyla = design.loc[design.subset == i, "phyla"].iloc[0].split("|")
        tr = pd.read_csv(V2 / "subsets" / f"subset_{i}.csv")
        idt = tiers["id_test"][tiers["id_test"].true_phylum.isin(phyla)]
        near = tiers["near"]; is_in = near.true_phylum.isin(phyla).values
        far = tiers["far"]
        r = {"subset": i, "n_idtest": len(idt), "n_nearid": int(is_in.sum()),
             "n_nearood": int((~is_in).sum()), "n_far": len(far)}

        def record(tag, s_idt, s_near, s_far):
            r[f"{tag}__idtest_v_nearid"] = auc(s_idt, s_near[is_in])
            r[f"{tag}__idtest_v_nearood"] = auc(s_idt, s_near[~is_in])
            r[f"{tag}__idtest_v_farood"] = auc(s_idt, s_far)
            r[f"{tag}__nearid_v_nearood"] = auc(s_near[is_in], s_near[~is_in])

        # ---------- fine-tuned GLM ----------
        if not args.skip_glm:
            ck = sorted((args.ckpt_root / f"subset_{i}").glob("checkpoint-*"))[-1]
            m = load_ckpt(ck)
            trs = tr.sample(args.n_train_feat, random_state=1)
            trf, _ = fl(m, tok, trs.sequence.values, args.max_len)
            mus, P = maha_fit(trf, trs.labels.values)
            packs = {k: fl(m, tok, v.sequence.values, args.max_len)
                     for k, v in [("idt", idt), ("near", near), ("far", far)]}
            del m; torch.cuda.empty_cache()
            for tag, fn in [("glm_msp", lambda f, l: -l.max(1)),
                            ("glm_energy", lambda f, l: -torch.logsumexp(torch.tensor(l), 1).numpy()),
                            ("glm_maha", lambda f, l: maha(mus, P, f)),
                            ("glm_knn", lambda f, l: knn(trf, f))]:
                record(tag, fn(*packs["idt"]), fn(*packs["near"]), fn(*packs["far"]))

        # ---------- k-mer Random Forest ----------
        trs = tr.sample(min(len(tr), args.rf_train), random_state=1)
        vec = CountVectorizer(analyzer="char", ngram_range=(6, 6), lowercase=False)
        Xtr = vec.fit_transform(trs.sequence.values)
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=1).fit(Xtr, trs.labels.values)
        def rf_scores(df):
            X = vec.transform(df.sequence.values).toarray()
            P_ = np.stack([t.predict_proba(X) for t in rf.estimators_])
            tot, epi, pb = bald(P_)
            return -pb.max(1), tot, epi
        s_idt, s_near, s_far = rf_scores(idt), rf_scores(near), rf_scores(far)
        for j, tag in enumerate(["rf_msp", "rf_entropy", "rf_epistemic"]):
            record(tag, s_idt[j], s_near[j], s_far[j])

        # ---------- from-scratch CNN ----------
        # one_hot_encode allocates [N,4,SEQ_LEN]; must match the actual read length
        cnn_scratch.SEQ_LEN = len(tr.sequence.iloc[0])
        Xt = cnn_scratch.one_hot_encode(tr.sequence.values)
        yt = torch.tensor(tr.labels.values, dtype=torch.long)
        torch.manual_seed(1)
        net = SmallCNN(n_classes=5, dropout_p=0.3).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for ep in range(args.cnn_epochs):
            perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(1000 + ep))
            net.train()
            for s in range(0, len(Xt), 256):
                idx = perm[s:s + 256]; opt.zero_grad()
                F.cross_entropy(net(Xt[idx].to(DEV)), yt[idx].to(DEV)).backward(); opt.step()

        @torch.no_grad()
        def cnn_scores(df):
            X = cnn_scratch.one_hot_encode(df.sequence.values)
            def run(train_mode):
                net.train(train_mode); o = []
                for s in range(0, len(X), 512):
                    o.append(torch.softmax(net(X[s:s + 512].to(DEV)), -1).cpu().numpy())
                return np.concatenate(o)
            det = run(False); P_ = np.stack([run(True) for _ in range(10)])
            tot, epi, _ = bald(P_)
            return -det.max(1), tot, epi
        s_idt, s_near, s_far = cnn_scores(idt), cnn_scores(near), cnn_scores(far)
        for j, tag in enumerate(["cnn_msp", "cnn_mcdo_entropy", "cnn_mcdo_epistemic"]):
            record(tag, s_idt[j], s_near[j], s_far[j])
        del net; torch.cuda.empty_cache()

        rows.append(r); print(f"subset_{i} done", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows); mean = df.drop(columns=[c for c in df if c.startswith("n_") or c == "subset"]).mean()
    tags = sorted({c.split("__")[0] for c in mean.index})
    print("\n{:<20}{:>11}{:>12}{:>12}{:>12}".format("method", "ID/nearID", "ID/nearOOD", "ID/farOOD", "nearID/OOD"))
    for t in tags:
        print("{:<20}{:>11.3f}{:>12.3f}{:>12.3f}{:>12.3f}".format(
            t, mean[f"{t}__idtest_v_nearid"], mean[f"{t}__idtest_v_nearood"],
            mean[f"{t}__idtest_v_farood"], mean[f"{t}__nearid_v_nearood"]))


if __name__ == "__main__":
    main()
