#!/usr/bin/env python3
"""Does the dark matter trend survive restricting the reference to a few phyla?

The retrieval reference has so far been the whole training set (all 36 phyla).
This restricts it to random 5-phylum subsets and repeats, asking two things:

  1. Does the sign of the `up` trend depend on WHICH phyla are in the reference?
     If it does -- and especially if it tracks the reference's GC -- then the
     trend is a property of reference composition rather than of taxonomic
     novelty. That is the sharpest available test of the GC confound, because
     `up` and GC are 94% collinear across samples and cannot be separated there.

  2. What unknown fraction does the model actually PREDICT? Treating `up` as a
     mixture proportion, perfect detection would flag FPR + (1-FPR)*up of reads,
     so inverting gives  up_hat = (flagged - FPR) / (1 - FPR).  Comparing up_hat
     to the true `up` says how far the estimate is from usable.

Embeddings are computed once; subsets are then taken in memory, so the cost is
one pass regardless of how many subsets are swept. The FPR threshold is
recalibrated per subset on id_test reads FROM THOSE SAME PHYLA -- otherwise the
threshold would be set by organisms the restricted reference does not cover.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
CKPT = "checkpoints/pbsim_clean/phylum/checkpoint-57734"
METAGENE = "metagene-ai/METAGENE-1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gc(s):
    s = s.upper(); return (s.count("G") + s.count("C")) / max(len(s), 1)


@torch.no_grad()
def embed(m, tok, seqs, L, bs, tag=""):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        o = m(**b, output_hidden_states=True) if hasattr(m, "classifier") else m(**b)
        h = o.hidden_states[-1] if hasattr(o, "hidden_states") and o.hidden_states is not None \
            else (o[0] if isinstance(o, (tuple, list)) else o.last_hidden_state)
        mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        if tag and i % (bs * 200) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--n-phyla", type=int, default=5)
    ap.add_argument("--n-subsets", type=int, default=15)
    ap.add_argument("--n-ref", type=int, default=80000)
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=384)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--encoder", default="nt", choices=["nt", "metagene1"])
    ap.add_argument("--tag", default="", help="suffix for the output CSV, so larger sweeps don't overwrite")
    args = ap.parse_args()
    if args.encoder == "metagene1":          # llama tokenizer, no 6-mer compression
        args.max_len, args.bs = 512, 128
    rng = np.random.default_rng(1); L = args.read_len
    cut = lambda s: s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]

    lin = pd.read_csv(D / "full_basic_lineage.csv", index_col=0)
    s2p = dict(zip(lin["species"].astype("Int64"), lin["phylum"].astype("Int64")))
    ref = pd.concat([c for c in pd.read_csv(D / "train.csv", usecols=["sequence", "taxid"],
                                            chunksize=50_000)], ignore_index=True)
    ref = ref.sample(args.n_ref, random_state=1)
    ref["phy"] = ref.taxid.map(lambda t: int(s2p.get(t, -1))); ref = ref[ref.phy != -1]
    idt = pd.read_csv(D / "id_test.csv", usecols=["sequence", "taxid"])
    idt["phy"] = idt.taxid.map(lambda t: int(s2p.get(t, -1))); idt = idt[idt.phy != -1]
    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    dm = dm[dm.up != 0.5].reset_index(drop=True)
    ref_w = [cut(s) for s in ref.sequence]; id_w = [cut(s) for s in idt.sequence]
    print(f"reference {len(ref_w):,} | id_test {len(id_w):,} | dark matter {len(dm):,}", flush=True)

    if args.encoder == "metagene1":
        tok = transformers.AutoTokenizer.from_pretrained(METAGENE, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = transformers.AutoModel.from_pretrained(METAGENE, trust_remote_code=True,
                                                   torch_dtype=torch.float16).to(DEV).eval()
    else:
        cfg = transformers.AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
        tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                         model_max_length=args.max_len,
                                                         padding_side="right", use_fast=True)
        tok.eos_token = tok.pad_token
        m = transformers.AutoModelForSequenceClassification.from_pretrained(
            HUB, config=cfg, trust_remote_code=True)
        m.load_state_dict(load_file(str(Path(CKPT) / "model.safetensors")), strict=True)
        m = m.to(DEV).eval().half()
    print(f"encoder: {args.encoder}", flush=True)
    R = embed(m, tok, ref_w, args.max_len, args.bs, "ref")
    Ei = embed(m, tok, id_w, args.max_len, args.bs, "id")
    Ed = embed(m, tok, dm.sequence.values, args.max_len, args.bs, "dm")
    del m; torch.cuda.empty_cache()
    ref_gc = np.array([gc(s) for s in ref_w]); ref_phy = ref.phy.values; id_phy = idt.phy.values
    Edg = Ed.to(DEV)

    def knn_against(Rsub):
        Rd = Rsub.to(DEV)
        def sc(E):
            s = np.empty(len(E))
            for i in range(0, len(E), 2048):
                s[i:i + 2048] = (-(E[i:i + 2048].to(DEV) @ Rd.T).max(1).values).cpu().numpy()
            return s
        a, b = sc(Ei), sc(Ed)
        del Rd; torch.cuda.empty_cache()
        return a, b

    ups = sorted(dm.up.unique())
    counts = pd.Series(ref_phy).value_counts()
    ok_phyla = [p for p in counts.index if counts[p] >= 200 and (id_phy == p).sum() >= 20]
    print(f"{len(ok_phyla)} phyla with enough reference and id_test reads", flush=True)

    rows = []
    specs = [("ALL", sorted(ok_phyla))] + [
        (f"S{i+1}", sorted(rng.choice(ok_phyla, args.n_phyla, replace=False).tolist()))
        for i in range(args.n_subsets)]
    for name, phy in specs:
        mr = np.isin(ref_phy, phy); mi = np.isin(id_phy, phy)
        si, sd = knn_against(R[mr])
        si = si[mi]                                   # threshold from the covered phyla only
        T = np.quantile(si, 1 - args.fpr)
        d2 = dm.assign(fl=(sd > T).astype(float))
        by = d2.groupby("up").fl.mean()
        rr = d2.groupby(["up", "sample"]).fl.mean().reset_index()
        sl, _, _, pv, _ = stats.linregress(rr.up, rr.fl)
        r = {"subset": name, "n_phyla": len(phy), "n_ref": int(mr.sum()), "n_id": int(mi.sum()),
             "ref_gc": float(ref_gc[mr].mean()), "slope": sl, "p": pv,
             "phyla": ";".join(map(str, phy))}
        for u in ups:
            r[f"flag_{u}"] = by[u]
            r[f"uphat_{u}"] = (by[u] - args.fpr) / (1 - args.fpr)   # mixture inversion
        rows.append(r)
        if len(specs) <= 40: print(f"  {name:4s} nphy={len(phy):2d} refGC={r['ref_gc']:.4f} "
              f"flag={[round(by[u],4) for u in ups]} slope={sl:+.4f} p={pv:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"results/taxa_subset_sweep_{args.encoder}{args.tag}.csv", index=False)
    sub = out[out.subset != "ALL"]
    print(f"\n=== {len(sub)} random {args.n_phyla}-phylum references, {args.fpr:.0%} FPR ===")
    print(f"  slopes: {(sub.slope>0).sum()} positive, {(sub.slope<0).sum()} negative; "
          f"significant: {((sub.slope>0)&(sub.p<0.05)).sum()} pos, {((sub.slope<0)&(sub.p<0.05)).sum()} neg")
    print(f"  slope range {sub.slope.min():+.4f} to {sub.slope.max():+.4f}")
    r_, p_ = stats.pearsonr(sub.ref_gc, sub.slope)
    print(f"  corr(reference GC, slope) = {r_:+.3f}  (p={p_:.4f})   <- compositional test")
    print(f"\n=== predicted unknown fraction (up_hat) vs truth ===")
    print(f"{'subset':8s}{'refGC':>8}" + "".join(f"{'up='+str(u):>12}" for u in ups))
    for _, r in out.iterrows():
        print(f"{r.subset:8s}{r.ref_gc:>8.4f}" + "".join(f"{r[f'uphat_{u}']:>12.4f}" for u in ups))
    print(f"\n  (true values are {ups}; up_hat assumes perfect detection, so the gap is the bias)")


if __name__ == "__main__":
    main()
