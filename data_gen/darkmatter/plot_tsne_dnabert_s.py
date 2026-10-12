#!/usr/bin/env python3
"""t-SNE of DNABERT-S embeddings: ID training reads vs dark matter, per `up`.

One t-SNE is fit on the COMBINED set (ID + all three `up` levels) and then split
into three panels. Fitting a separate t-SNE per panel would be wrong: t-SNE
coordinates are arbitrary up to rotation/scale and are not comparable between
fits, so three independent fits could not be read side by side.

Standard pipeline: L2-normalised 768-d embeddings -> PCA to 50 -> t-SNE to 2.
Subsampled because t-SNE is O(n^2)-ish; the full 180k dark matter reads are not
tractable and are not needed to see cluster structure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MODEL = "zhihan1996/DNABERT-S"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
UPS = [0.25, 0.75, 1.0]


@torch.no_grad()
def embed(m, tok, seqs, bs=256, tag=""):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=512)
        b = {k: v.to(DEV) for k, v in b.items()}
        o = m(**b)
        h = o[0] if isinstance(o, (tuple, list)) else o.last_hidden_state
        mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        if tag and i % (bs * 20) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--n-id", type=int, default=3000)
    ap.add_argument("--n-dm", type=int, default=3000, help="per `up` level")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=Path(".docs/latex/figs/darkmatter_tsne_dnabert_s.pdf"))
    args = ap.parse_args()
    rng = np.random.default_rng(1); L = 150
    cut = lambda s: s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]

    tr = pd.read_csv(D / "train.csv", usecols=["sequence"]).sample(args.n_id, random_state=1).sequence
    id_w = [cut(s) for s in tr]
    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    dm = dm[dm.up != 0.5]
    dm_s, dm_u = [], []
    for u in UPS:
        sub = dm[dm.up == u].sample(args.n_dm, random_state=1)
        dm_s += list(sub.sequence.values); dm_u += [u] * len(sub)
    dm_u = np.array(dm_u)
    print(f"ID training {len(id_w)} | dark matter {len(dm_s)} ({args.n_dm} per up)", flush=True)

    cfg = transformers.AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.attention_probs_dropout_prob = 0.1       # avoids Triton flash path; inert under eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    m = transformers.AutoModel.from_pretrained(MODEL, config=cfg,
                                               trust_remote_code=True).to(DEV).eval()
    Ei = embed(m, tok, id_w, tag="id"); Ed = embed(m, tok, dm_s, tag="dm")
    del m; torch.cuda.empty_cache()

    X = np.vstack([Ei, Ed])
    print(f"PCA 768 -> 50, then t-SNE on {len(X)} points ...", flush=True)
    Xp = PCA(n_components=50, random_state=1).fit_transform(X)
    Y = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
             random_state=1, max_iter=1000).fit_transform(Xp)
    Yi, Yd = Y[:len(Ei)], Y[len(Ei):]

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharex=True, sharey=True)
    for ax, u in zip(axes, UPS):
        ax.scatter(Yi[:, 0], Yi[:, 1], s=4, c="#bdbdbd", linewidths=0,
                   label="ID training", rasterized=True)
        m_ = dm_u == u
        ax.scatter(Yd[m_, 0], Yd[m_, 1], s=4, c="#08519c", linewidths=0,
                   label="dark matter", rasterized=True)
        ax.set_title(f"up = {u:.2f}", fontsize=11, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(markerscale=4, fontsize=9, loc="upper left", frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200); fig.savefig(str(args.out).replace(".pdf", ".png"), dpi=150)
    np.savez_compressed("results/darkmatter_tsne_dnabert_s.npz", Y=Y, n_id=len(Ei), dm_up=dm_u)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
