#!/usr/bin/env python3
"""Does the darkmatter `up` trend depend on WHICH phyla are in the reference?

Yesterday's run (10-phylum reference) showed flagged-fraction rising with `up`
at every threshold. Today's (all 43 phyla, leakage-free encoder, same k) does
not. Restoring k=1 did not bring it back, so k was not the cause -- leaving the
encoder or the reference.

The encoder should be nearly irrelevant here: contamination makes id_test reads
embed closer to the reference, which shifts the calibrated threshold, but a
constant offset cannot create a trend across `up`. The reference can. If it
covers only part of the phylogeny and darkmatter's unknown fraction is enriched
for the phyla it omits, then higher `up` mechanically means more reads with no
in-reference match -- an increasing trend produced by reference composition
rather than by novelty detection.

This holds the encoder fixed (clean checkpoint) and varies only the reference:
the embedding pool is computed once and different phylum subsets are selected
from it, so many configurations cost one embedding pass. If the slope tracks
reference composition, yesterday's trend was an artifact.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
CLEAN = "checkpoints/pbsim_clean/phylum/checkpoint-57734"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# the 10 phyla whose windows formed yesterday's reference
LPO10 = ["1224", "201174", "1239", "976", "1117", "200940", "203682", "2818505",
         "203691", "57723"]


@torch.no_grad()
def embed(m, tok, seqs, L, bs, tag=""):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        o = m(**b, output_hidden_states=True)
        h = o.hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        if tag and i % (bs * 200) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--per-phylum", type=int, default=2500, help="reference windows per phylum")
    ap.add_argument("--n-id", type=int, default=6000)
    ap.add_argument("--n-dm", type=int, default=16000)
    ap.add_argument("--n-random", type=int, default=5, help="random 10-phylum references")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    L = args.read_len

    lin = pd.read_csv(D / "full_basic_lineage.csv")
    sp2ph = dict(zip(lin.species.astype(str), lin.phylum.astype(str)))
    rows = []
    for ch in pd.read_csv(D / "train.csv", usecols=["sequence", "taxid"], chunksize=100_000):
        ch["ph"] = ch.taxid.astype(str).map(sp2ph)
        rows.append(ch.dropna(subset=["ph"]))
    tr = pd.concat(rows, ignore_index=True)

    # one window per read, capped per phylum, so every phylum is represented
    ref = []
    for p, g in tr.groupby("ph"):
        g = g.sample(min(len(g), args.per_phylum), random_state=1)
        for s in g.sequence:
            if len(s) > L:
                st = int(rng.integers(0, len(s) - L)); ref.append((s[st:st + L], p))
    ref = pd.DataFrame(ref, columns=["sequence", "ph"])
    print(f"reference pool: {len(ref):,} windows across {ref.ph.nunique()} phyla")

    idt = pd.read_csv(D / "id_test.csv", usecols=["sequence"]).sample(args.n_id, random_state=1)
    id_w = [s[(st := int(rng.integers(0, len(s) - L))):st + L] if len(s) > L else s
            for s in idt.sequence]
    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    per = args.n_dm // dm.up.nunique()
    dm = (dm.groupby("up", group_keys=False)
            .apply(lambda g: g.sample(n=min(per, len(g)), random_state=1)).reset_index(drop=True))

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    cfg = transformers.AutoConfig.from_pretrained(CLEAN, trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                        trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CLEAN) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()
    Eref = embed(m, tok, ref.sequence.values, args.max_len, args.bs, "ref")
    Eid = embed(m, tok, id_w, args.max_len, args.bs, "id")
    Edm = embed(m, tok, dm.sequence.values, args.max_len, args.bs, "dm")
    del m; torch.cuda.empty_cache()

    all_ph = sorted(ref.ph.unique())
    configs = [("all_43", all_ph), ("lpo_10", [p for p in LPO10 if p in all_ph])]
    for i in range(args.n_random):
        configs.append((f"random10_{i}", list(rng.choice(all_ph, size=10, replace=False))))
    for n in (5, 20):
        configs.append((f"random{n}", list(rng.choice(all_ph, size=n, replace=False))))

    out = []
    for name, phyla in configs:
        mask = ref.ph.isin(phyla).values
        R = Eref[mask].to(DEV)
        def sc(E):
            s = np.empty(len(E))
            for i in range(0, len(E), 512):
                sim = E[i:i + 512].to(DEV) @ R.T
                s[i:i + 512] = (-sim.topk(args.k, 1).values[:, -1]).cpu().numpy()
            return s
        s_id, s_dm = sc(Eid), sc(Edm)
        auc = roc_auc_score(np.r_[np.zeros(len(s_id)), np.ones(len(s_dm))], np.r_[s_id, s_dm])
        d = dm.assign(novelty=s_dm)
        line = {"reference": name, "n_phyla": len(phyla), "n_ref": int(mask.sum()), "auroc": auc}
        for fpr in (0.01, 0.05, 0.10):
            thr = np.quantile(s_id, 1 - fpr)
            d["flag"] = (d.novelty > thr).astype(float)
            rep = d[d.up != 0.5].groupby(["up", "sample"]).flag.mean().reset_index()
            sl, _, r, p_, se = stats.linregress(rep.up, rep.flag)
            line[f"slope_{int(fpr*100)}"] = sl
            line[f"p_{int(fpr*100)}"] = p_
        out.append(line)
        print(f"{name:<14} phyla={len(phyla):3d} n_ref={mask.sum():6d} AUROC={auc:.3f}  "
              f"slopes: 1%={line['slope_1']:+.4f}(p={line['p_1']:.3f}) "
              f"5%={line['slope_5']:+.4f}(p={line['p_5']:.3f}) "
              f"10%={line['slope_10']:+.4f}(p={line['p_10']:.3f})", flush=True)
        pd.DataFrame(out).to_csv("results/darkmatter_reference_composition.csv", index=False)


if __name__ == "__main__":
    main()
