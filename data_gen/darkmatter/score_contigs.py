#!/usr/bin/env python3
"""Score assembled dark matter contigs for novelty, at MATCHED window length.

The length sweep showed NT has no detection signal below ~600bp and only
recovers above ~1200bp, so 150bp reads sit below its operating range. These are
the per-sample MEGAHIT contigs (data_gen/darkmatter/assemble.sh). Every contig,
every reference read and every id_test read is cut to ONE window of the same
length W, so read length cannot re-enter as a confound -- the earlier unwindowed
comparison showed it can dominate AUROC.

Unit of analysis is the contig. Assembly favours abundant organisms, so what
assembles differs across `up`; contig counts and a coverage-weighted variant
(MEGAHIT's multi= field) are reported so that selection is visible.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
ASM = Path("/scratch/home/glh52/darkmatter_assembly")
CKPT = {"nt-phylum": "checkpoints/pbsim_clean/phylum/checkpoint-57734",
        "nt-family": "checkpoints/pbsim_clean/family/checkpoint-57734"}
DEV = "cuda"
FPRS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def read_contigs(fa: Path):
    seqs, covs, cur, cov = [], [], [], 0.0
    for line in fa.open():
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); covs.append(cov)
            m = re.search(r"multi=([0-9.]+)", line); cov = float(m.group(1)) if m else 1.0
            cur = []
        else:
            cur.append(line.strip())
    if cur: seqs.append("".join(cur)); covs.append(cov)
    return seqs, np.array(covs)


def load_encoder(name, W):
    if name == "metagene1":
        tok = transformers.AutoTokenizer.from_pretrained("metagene-ai/METAGENE-1", trust_remote_code=True)
        tok.pad_token = tok.pad_token or tok.eos_token
        m = transformers.AutoModel.from_pretrained("metagene-ai/METAGENE-1", trust_remote_code=True,
                                                   torch_dtype=torch.float16).to(DEV).eval()
        return m, tok, 512, False
    ml = W // 6
    cfg = transformers.AutoConfig.from_pretrained(CKPT[name], trust_remote_code=True)
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True, model_max_length=ml,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg, trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CKPT[name]) / "model.safetensors")), strict=True)
    return m.to(DEV).eval().half(), tok, ml, True


@torch.no_grad()
def embed(m, tok, seqs, ml, bs, is_nt, tag=""):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True, truncation=True, max_length=ml)
        b = {k: v.to(DEV) for k, v in b.items()}
        if is_nt:
            h = m(**b, output_hidden_states=True).hidden_states[-1]
        else:
            o = m(**b); h = o[0] if isinstance(o, (tuple, list)) else o.last_hidden_state
        mk = b["attention_mask"].unsqueeze(-1).float()
        out.append(torch.nn.functional.normalize(((h * mk).sum(1) / mk.sum(1)).float(), dim=-1).cpu())
        if tag and i % (bs * 50) == 0: print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["nt-phylum", "nt-family", "metagene1"])
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--n-ref", type=int, default=40000)
    args = ap.parse_args(); W = args.window
    rng = np.random.default_rng(1)
    cut = lambda s: s[(st := int(rng.integers(0, len(s) - W + 1))):st + W]

    rows = []
    for d in sorted(ASM.glob("up*_s*")):
        fa = d / "asm" / "final.contigs.fa"
        if not fa.exists(): continue
        up = float(re.search(r"up([0-9.]+)_", d.name).group(1)); s = int(d.name.split("_s")[1])
        seqs, cov = read_contigs(fa)
        keep = [i for i, x in enumerate(seqs) if len(x) >= W]
        for i in keep: rows.append((up, s, cut(seqs[i]), cov[i], len(seqs[i])))
    ct = pd.DataFrame(rows, columns=["up", "sample", "seq", "cov", "len"])
    print(f"contigs >= {W}bp:\n{ct.groupby('up').size().to_string()}", flush=True)
    print(f"samples per up: {ct.groupby('up')['sample'].nunique().to_dict()}", flush=True)

    ref = pd.read_csv(D / "train.csv", usecols=["sequence"]).sequence
    ref = ref[ref.str.len() >= W].sample(args.n_ref, random_state=1)
    idt = pd.read_csv(D / "id_test.csv", usecols=["sequence"]).sequence
    idt = idt[idt.str.len() >= W]
    m, tok, ml, is_nt = load_encoder(args.encoder, W)
    bs = max(4, int(64 * 2000 / W)) if is_nt else 32
    R = embed(m, tok, [cut(x) for x in ref], ml, bs, is_nt, "ref").to(DEV)
    Ei = embed(m, tok, [cut(x) for x in idt], ml, bs, is_nt, "id")
    Ec = embed(m, tok, list(ct.seq), ml, bs, is_nt, "contigs")
    def knn(E):
        return np.concatenate([(-(E[i:i + 2048].to(DEV) @ R.T).max(1).values).cpu().numpy()
                               for i in range(0, len(E), 2048)])
    si, sc = knn(Ei), knn(Ec); ct["v"] = sc

    tag = f"{args.encoder}_W{W}"
    ct.drop(columns="seq").to_csv(f"results/contigs_{tag}.csv", index=False)
    np.save(f"results/contigs_{tag}_id.npy", si)
    print(f"\n=== {args.encoder}, {W}bp windows, kNN(k=1) ===")
    print(f"  AUROC id_test vs contigs: " + "  ".join(
        f"up={u}: {roc_auc_score(np.r_[np.zeros(len(si)),np.ones((ct.up==u).sum())], np.r_[si, ct.v[ct.up==u]]):.4f}"
        for u in sorted(ct.up.unique())))
    def trend(label, rep):
        lo, hi = rep[rep.up == rep.up.min()].v, rep[rep.up == rep.up.max()].v
        sl, _, r, p, _ = stats.linregress(rep.up, rep.v)
        d = (hi.mean() - lo.mean()) / np.sqrt((lo.var(ddof=1) + hi.var(ddof=1)) / 2)
        print(f"  {label:34s} slope {sl:+.4f}  R2 {r*r:.3f}  d {d:+6.2f}  p {p:.2e}  "
              f"means " + " ".join(f"{x:.4f}" for x in rep.groupby('up').v.mean()))
    trend("mean novelty (per sample)", ct.groupby(["up", "sample"]).v.mean().reset_index())
    w = ct.assign(wv=ct.v * ct["cov"]).groupby(["up", "sample"])   # ct.cov is DataFrame.cov()
    trend("coverage-weighted mean novelty", (w.wv.sum() / w["cov"].sum()).rename("v").reset_index())
    for f in FPRS:
        T = np.quantile(si, 1 - f)
        trend(f"flagged @ {f:.1%} FPR", ct.assign(v=(ct.v > T).astype(float))
              .groupby(["up", "sample"]).v.mean().reset_index())


if __name__ == "__main__":
    main()
