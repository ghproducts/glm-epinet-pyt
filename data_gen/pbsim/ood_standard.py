#!/usr/bin/env python3
"""OOD detection on the STANDARD pbsim task, not the leave-phyla-out subsets.

The leave-phyla-out design (10 models, 5 phyla each) exists to make the label
space an independent variable, which was needed for the darkmatter question.
For "can a GLM detect novel organisms?" it is the wrong vehicle: the models are
artificial 5-way classifiers that appear nowhere else in the project. This uses
the original full-label-space checkpoint and the project's own taxonomic
splits, which already form a novelty ladder:

  id_test           training species, held-out reads      in-distribution
  id_novel_genus    novel genus, phylum known             near
  ood_novel_family  novel family, phylum still known      near
  ood_nonbacterial  archaea / eukaryota / viruses         far

Scores, all from one model and one forward pass per read:
  msp, energy      output space (classifier head)
  maha, knn        feature space (nearest-neighbour retrieval)

The retrieval reference is the FULL training set -- every training read, cut to
6kb -- rather than a sample, so reference size is not a free parameter. k is
swept rather than fixed, since k=1 and k=50 were giving different answers
elsewhere in this project.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
DATA = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")  # clean split
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# Leakage-free checkpoints. The old_models/* ones were trained on reads later
# carved into id_test, so any ID-tier comparison against them scored training
# data.
CKPT = {"phylum": "checkpoints/pbsim_clean/phylum/checkpoint-57734",
        "family": "checkpoints/pbsim_clean/family/checkpoint-57734"}


def load_enc(rank):
    cfg = transformers.AutoConfig.from_pretrained(CKPT[rank], trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                       trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CKPT[rank]) / "model.safetensors")), strict=True)
    return m.to(DEV).eval().half()


@torch.no_grad()
def fwd(m, tok, seqs, L, bs, tag=""):
    F, G = [], []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        o = m(**b, output_hidden_states=True)
        h = o.hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        F.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        G.append(o.logits.float().cpu())
        if tag and i % (bs * 400) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(F), torch.cat(G)


def cut(df, L, rng):
    out = []
    for s in df.sequence:
        out.append(s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", default="phylum", choices=["phylum", "family"])
    ap.add_argument("--read-len", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--ks", type=int, nargs="*", default=[50])
    ap.add_argument("--n-query", type=int, default=3000)
    ap.add_argument("--n-ref", type=int, default=0, help="0 = use every training read")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    L = args.read_len

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    m = load_enc(args.rank)

    print("reading training set for the retrieval reference ...")
    ref = pd.concat([c for c in pd.read_csv(DATA / "train.csv", usecols=["sequence", "taxid"],
                                            chunksize=50_000)], ignore_index=True)
    ref = ref[ref.sequence.str.len() >= L]
    # Rank labels for the class-conditional Mahalanobis fit, mapped the same way
    # nn_proj/common/datasets.load_local_dataset does it (species taxid -> rank taxid).
    lin = pd.read_csv(DATA / "full_basic_lineage.csv", index_col=0)
    s2t = dict(zip(lin["species"].astype("Int64"), lin[args.rank].astype("Int64")))
    ref["ranklab"] = ref.taxid.map(lambda t: int(s2t.get(t, -1)))
    n_un = int((ref.ranklab == -1).sum())
    if n_un:
        print(f"  dropping {n_un:,} reference reads with no {args.rank} in the lineage table")
        ref = ref[ref.ranklab != -1]
    if args.n_ref:
        ref = ref.sample(args.n_ref, random_state=1)
    print(f"  reference: {len(ref):,} training reads -> 6kb windows (no sampling)" if not args.n_ref
          else f"  reference: {len(ref):,} (sampled)")
    R, _ = fwd(m, tok, cut(ref, L, rng), args.max_len, args.bs, "ref")
    ref_lab = ref.ranklab.values

    tiers = {}
    for name, f in [("id_test", "id_test.csv"), ("near_genus", "id_novel_genus.csv"),
                    ("near_family", "ood_novel_family.csv"), ("far_nonbact", "ood_nonbacterial.csv")]:
        d = pd.read_csv(DATA / f, usecols=["sequence", "taxid"])
        d = d[d.sequence.str.len() >= L]
        if len(d) > args.n_query:
            d = d.sample(args.n_query, random_state=1)
        tiers[name] = fwd(m, tok, cut(d, L, rng), args.max_len, args.bs, name)
        print(f"  {name}: {len(d)} reads")
    del m; torch.cuda.empty_cache()

    Rd = R.to(DEV)
    def knn(Q, k):
        s = np.empty(len(Q))
        for i in range(0, len(Q), 64):
            sim = Q[i:i + 64].to(DEV) @ Rd.T
            s[i:i + 64] = (-sim.topk(k, 1).values[:, -1]).cpu().numpy()
        return s

    # Two Mahalanobis variants, reported separately. (An earlier comment here
    # claimed class means and pseudo-labels while the code below fit a single
    # global Gaussian -- that mismatch is what prompted adding the real
    # class-conditional form.)
    import numpy.linalg as la
    Rn = R.numpy().astype(np.float64)

    def whitener(P):
        """W with W @ W.T == P, so d' P d == ||d @ W||^2. eigh not cholesky:
        P comes from pinv and can land a hair outside PSD."""
        w, V = la.eigh(P)
        return V * np.sqrt(np.clip(w, 0, None))

    # `maha` = one global Gaussian. Kept under its original name so the phylum
    # numbers already on disk reproduce exactly.
    mu_g = Rn.mean(0, keepdims=True)
    W_g = whitener(la.pinv(np.cov((Rn - mu_g).T) + 1e-6 * np.eye(Rn.shape[1])))
    def maha(F):
        return (((F.numpy().astype(np.float64) - mu_g) @ W_g) ** 2).sum(1)

    # `maha_class` = per-class means + tied covariance, min distance over classes.
    # This is what Lee et al. 2018 actually specifies; on gene_taxa it beat the
    # global form on both shift axes (0.575 vs 0.562, 0.987 vs 0.976).
    classes = np.unique(ref_lab)
    MUc = np.stack([Rn[ref_lab == c].mean(0) for c in classes])
    resid = Rn - MUc[np.searchsorted(classes, ref_lab)]
    W_c = whitener(la.pinv(np.cov(resid.T) + 1e-6 * np.eye(Rn.shape[1])))
    MUw = torch.from_numpy(MUc @ W_c).to(DEV).float()
    def maha_class(F):
        Xw = torch.from_numpy(F.numpy().astype(np.float64) @ W_c).to(DEV).float()
        out = torch.empty(len(Xw), device=DEV)
        for i in range(0, len(Xw), 1024):
            out[i:i + 1024] = torch.cdist(Xw[i:i + 1024], MUw).pow(2).min(1).values
        return out.cpu().numpy()
    print(f"  class-conditional fit over {len(classes)} {args.rank} classes")

    scores = {}
    for name, (F, G) in tiers.items():
        scores[(name, "maha")] = maha(F)
        scores[(name, "maha_class")] = maha_class(F)
        scores[(name, "msp")] = -torch.softmax(G, -1).max(1).values.numpy()
        scores[(name, "energy")] = -torch.logsumexp(G, 1).numpy()
        for k in args.ks:
            scores[(name, f"knn_k{k}")] = knn(F, k)

    def auc(a, b):
        return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
    methods = ["msp", "energy", "maha", "maha_class"] + [f"knn_k{k}" for k in args.ks]
    rows = []
    for meth in methods:
        r = {"method": meth}
        for t in ["near_genus", "near_family", "far_nonbact"]:
            r[f"idtest_v_{t}"] = auc(scores[("id_test", meth)], scores[(t, meth)])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"results/ood_standard_{args.rank}.csv", index=False)
    print(f"\n=== standard pbsim {args.rank} task, AUROC (0.5 = no signal) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
