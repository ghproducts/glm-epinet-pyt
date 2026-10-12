#!/usr/bin/env python3
"""OOD detection on the Scorpio gene_taxa task (NT_transformer, 437 gene families).

Two novelty axes, orthogonal by construction -- this is what the task offers
that pbsim does not:

  test      ID            same gene families, same taxa            labels valid
  taxa_out  covariate     same gene families, HELD-OUT taxa        labels valid
  gene_out  semantic      gene families absent from the label      labels DUMMY
                          space entirely (every row carries a          (all =1)
                          placeholder label of 1)

Detection needs no labels, so all three are scored the same way. Scores:
  msp, energy    output space (classifier head)
  maha_global    one Gaussian over all reference embeddings
  maha_class     class means + tied covariance, min distance over classes
  knn_k*         cosine distance to the k-th nearest reference embedding

maha_global vs maha_class: data_gen/pbsim/ood_standard.py computes only the
global variant, though its comment describes the class-conditional one. Lee
et al. 2018 (arXiv:1807.03888) is class-conditional with a tied covariance, so
both are reported here to show what that choice is worth. gene_taxa has true
training labels, so the class-conditional fit is exact rather than pseudo-labelled.

Reference = training reads. Sequences are ~930bp median against a 3000bp
(500-token) window, so nothing is truncated and there is no windowing step --
unlike pbsim, where 10.7kb reads had to be cut to 6kb.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
SRC = Path("/scratch/home/glh52/glm-epinet/DATA/gene_taxa")
# Capped tier copies written by scripts/eval_gene_taxa.sh, so detection and
# calibration are computed over the same reads.
CAPPED = Path("temp/gene_taxa_eval")
CKPT = "old_models/trained_models_1/NT_transformer/gene_taxa/checkpoint-30800"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_enc():
    cfg = transformers.AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB, config=cfg, trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CKPT) / "model.safetensors")), strict=True)
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
        if tag and i % (bs * 200) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(F), torch.cat(G)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=500)   # 3000bp / 6 (NT is 6-mer)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 10, 50])
    ap.add_argument("--n-query", type=int, default=5000)
    ap.add_argument("--n-ref", type=int, default=150_000,
                    help="training reads used as the retrieval reference; 0 = all 547,523")
    args = ap.parse_args()

    tok = transformers.AutoTokenizer.from_pretrained(
        HUB, trust_remote_code=True, model_max_length=args.max_len,
        padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    m = load_enc()

    print("reading training set for the retrieval reference ...", flush=True)
    ref = pd.concat(list(pd.read_csv(SRC / "train.csv", usecols=["sequence", "labels"],
                                     chunksize=100_000)), ignore_index=True)
    n_all = len(ref)
    if args.n_ref and args.n_ref < n_all:
        ref = ref.sample(args.n_ref, random_state=1)
    print(f"  reference: {len(ref):,} of {n_all:,} training reads"
          f"{' (SUBSAMPLED)' if len(ref) < n_all else ' (all)'}", flush=True)
    R, _ = fwd(m, tok, ref.sequence.values, args.max_len, args.bs, "ref")
    ref_lab = ref.labels.values

    tiers = {}
    for name in ["test", "taxa_out", "gene_out"]:
        f = CAPPED / f"{name}.csv" if (CAPPED / f"{name}.csv").exists() else SRC / f"{name}.csv"
        d = pd.read_csv(f, usecols=["sequence"])
        if len(d) > args.n_query:
            d = d.sample(args.n_query, random_state=1)
        tiers[name] = fwd(m, tok, d.sequence.values, args.max_len, args.bs, name)
        print(f"  {name}: {len(d)} reads (from {f})", flush=True)
    del m; torch.cuda.empty_cache()

    Rd = R.to(DEV)
    def knn(Q, k):
        s = np.empty(len(Q))
        for i in range(0, len(Q), 64):
            sim = Q[i:i + 64].to(DEV) @ Rd.T
            s[i:i + 64] = (-sim.topk(k, 1).values[:, -1]).cpu().numpy()
        return s

    Rn = R.numpy().astype(np.float64)

    def whitener(P):
        """W with W @ W.T == P, so d' P d == ||d @ W||^2 and Mahalanobis
        distance becomes a plain squared euclidean one in whitened space.
        eigh rather than cholesky: P comes from pinv and can land a hair
        outside PSD, which cholesky rejects outright."""
        w, V = np.linalg.eigh(P)
        return V * np.sqrt(np.clip(w, 0, None))

    # Global: one Gaussian over the whole reference (what ood_standard.py does).
    mu_g = Rn.mean(0, keepdims=True)
    W_g = whitener(np.linalg.pinv(np.cov((Rn - mu_g).T) + 1e-6 * np.eye(Rn.shape[1])))
    def maha_global(F):
        return (((F.numpy().astype(np.float64) - mu_g) @ W_g) ** 2).sum(1)

    # Class-conditional with tied covariance (Lee et al. 2018): per-class means,
    # covariance pooled over within-class residuals, score = min_c distance.
    # Whitened, this is a nearest-centroid search, so cdist does it in one pass
    # instead of the O(b*C*D^2) einsum the direct form would need.
    classes = np.unique(ref_lab)
    MU = np.stack([Rn[ref_lab == c].mean(0) for c in classes])
    resid = Rn - MU[np.searchsorted(classes, ref_lab)]
    W_c = whitener(np.linalg.pinv(np.cov(resid.T) + 1e-6 * np.eye(Rn.shape[1])))
    MUw = torch.from_numpy(MU @ W_c).to(DEV).float()
    def maha_class(F):
        Xw = torch.from_numpy(F.numpy().astype(np.float64) @ W_c).to(DEV).float()
        out = torch.empty(len(Xw), device=DEV)
        for i in range(0, len(Xw), 1024):
            out[i:i + 1024] = torch.cdist(Xw[i:i + 1024], MUw).pow(2).min(1).values
        return out.cpu().numpy()

    scores = {}
    for name, (F, G) in tiers.items():
        scores[(name, "msp")] = -torch.softmax(G, -1).max(1).values.numpy()
        scores[(name, "energy")] = -torch.logsumexp(G, 1).numpy()
        scores[(name, "maha_global")] = maha_global(F)
        scores[(name, "maha_class")] = maha_class(F)
        for k in args.ks:
            scores[(name, f"knn_k{k}")] = knn(F, k)

    def auc(a, b):
        return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
    methods = ["msp", "energy", "maha_global", "maha_class"] + [f"knn_k{k}" for k in args.ks]
    rows = [{"method": mt,
             "test_v_taxa_out": auc(scores[("test", mt)], scores[("taxa_out", mt)]),
             "test_v_gene_out": auc(scores[("test", mt)], scores[("gene_out", mt)])}
            for mt in methods]
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/ood_gene_taxa.csv", index=False)
    np.savez("results/ood_gene_taxa_scores.npz",
             **{f"{n}__{mt}": v for (n, mt), v in scores.items()})
    print(f"\n=== gene_taxa AUROC (0.5 = no signal); reference {len(ref):,} train reads ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
