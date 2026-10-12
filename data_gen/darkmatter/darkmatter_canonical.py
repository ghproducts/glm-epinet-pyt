#!/usr/bin/env python3
"""Canonical dark matter analysis -- ALL reads, one fixed configuration.

Earlier runs subsampled dark matter to ~16k of the 183k available reads. That
was an unforced choice: at 150bp the full set costs ~20 minutes to embed, and
the subsample left each replicate resting on ~800 reads, whose binomial noise
(~0.008 on a 5% flagged fraction) is the same order as the slopes being
measured. Two runs that differed only in their random draw disagreed on the
sign at loose thresholds. This uses every read.

Fixed configuration, no free parameters left floating:
  reads       all dark matter reads except up=0.5 (one file, no replicates)
  encoder     checkpoints/pbsim_clean/<rank>  (leakage-free); --rank selects
              phylum (36-way fine-tuning) or family (429-way). Only the encoder
              changes between runs -- reads, reference draw, k, seed and
              thresholds are identical -- so a difference in trend is
              attributable to the fine-tuning objective. The classifier head is
              unused either way; scores come from mean-pooled hidden states.
  reference   natural-abundance draw from the clean training set -- what a
              deployment would actually index; balancing by phylum was tested
              and changes nothing qualitatively
  k           1
  threshold   (1-FPR) quantile of id_test, swept; never calibrated on the test
              data itself

Reports threshold-swept and threshold-free measures together, since they
disagree and both are needed to describe the result honestly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
CKPT = {"phylum": "checkpoints/pbsim_clean/phylum/checkpoint-57734",
        "family": "checkpoints/pbsim_clean/family/checkpoint-57734"}
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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
        if tag and i % (bs * 400) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=384)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n-ref", type=int, default=80000)
    ap.add_argument("--rank", default="phylum", choices=["phylum", "family"])
    args = ap.parse_args()
    # phylum keeps the original unsuffixed filenames so the published figure and
    # writeup numbers stay where they are.
    sfx = "" if args.rank == "phylum" else f"_{args.rank}"
    CLEAN = CKPT[args.rank]
    print(f"encoder: {CLEAN}")
    rng = np.random.default_rng(1); L = args.read_len
    cut = lambda s: s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]

    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    dm = dm[dm.up != 0.5].reset_index(drop=True)
    print(f"dark matter: ALL {len(dm):,} reads  {dm.up.value_counts().sort_index().to_dict()}")

    ref = pd.concat([c for c in pd.read_csv(D / "train.csv", usecols=["sequence"],
                                            chunksize=50_000)], ignore_index=True)
    ref_w = [cut(s) for s in ref.sample(args.n_ref, random_state=1).sequence]
    idt = pd.read_csv(D / "id_test.csv", usecols=["sequence"])
    id_w = [cut(s) for s in idt.sequence]
    print(f"reference {len(ref_w):,} windows | id_test ALL {len(id_w):,} reads")

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    cfg = transformers.AutoConfig.from_pretrained(CLEAN, trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                        trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CLEAN) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()
    R = embed(m, tok, ref_w, args.max_len, args.bs, "ref").to(DEV)
    Ei = embed(m, tok, id_w, args.max_len, args.bs, "id")
    Ed = embed(m, tok, dm.sequence.values, args.max_len, args.bs, "dm")
    del m; torch.cuda.empty_cache()

    def sc(E):
        s = np.empty(len(E))
        for i in range(0, len(E), 512):
            sim = E[i:i + 512].to(DEV) @ R.T
            s[i:i + 512] = (-sim.topk(args.k, 1).values[:, -1]).cpu().numpy()
        return s
    si, sd = sc(Ei), sc(Ed)
    dm = dm.assign(novelty=sd)
    np.save(f"results/dm_canon_id{sfx}.npy", si)
    dm.to_csv(f"results/darkmatter_canonical_scores{sfx}.csv", index=False)

    ups = sorted(dm.up.unique())
    print("\n=== threshold-free ===")
    rep = dm.groupby(["up", "sample"]).novelty.mean().reset_index()
    sl, _, _, p, se = stats.linregress(rep.up, rep.novelty)
    print(f"  mean score vs up           slope={sl:+.5f} +/- {se:.5f}  p={p:.4f}")
    a, b = dm[dm.up == ups[0]].novelty.values, dm[dm.up == ups[-1]].novelty.values
    print(f"  AUROC up={ups[0]} vs up={ups[-1]}     "
          f"{roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]):.4f}   (0.5 = no trend)")
    print("  AUROC id_test vs each up:  " + "  ".join(
        f"up={u}:{roc_auc_score(np.r_[np.zeros(len(si)), np.ones((dm.up==u).sum())], np.r_[si, dm[dm.up==u].novelty]):.4f}"
        for u in ups))

    print("\n=== threshold sweep (FPR calibrated on id_test) ===")
    print(f"{'FPR':>7}{'flagged @up':>34}{'slope':>10}{'p':>9}")
    rows = []
    for f in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        T = np.quantile(si, 1 - f)
        d2 = dm.assign(flag=(dm.novelty > T).astype(float))
        by = d2.groupby("up").flag.mean()
        r = d2.groupby(["up", "sample"]).flag.mean().reset_index()
        s2, _, _, p2, se2 = stats.linregress(r.up, r.flag)
        rows.append(dict(fpr=f, slope=s2, p=p2, se=se2,
                         **{f"flag_{u}": by[u] for u in ups}))
        print(f"{f:>7.1%}" + "  ".join(f"{u:.2f}:{by[u]:.4f}" for u in ups).rjust(32)
              + f"{s2:>10.4f}{p2:>9.4f}" + ("*" if p2 < 0.05 else ""))
    pd.DataFrame(rows).to_csv(f"results/darkmatter_canonical_sweep{sfx}.csv", index=False)


if __name__ == "__main__":
    main()
