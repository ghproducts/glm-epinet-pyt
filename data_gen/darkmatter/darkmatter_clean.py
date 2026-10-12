#!/usr/bin/env python3
"""Darkmatter novelty detection against the LEAKAGE-FREE checkpoint.

Supersedes retrieval_darkmatter.py, which used (a) the old pbsim_phylum
checkpoint -- trained on reads later carved into id_test, so its ID reference
was contaminated -- and (b) a reference built from the leave-phyla-out subsets,
a scaffold that has since been dropped.

Here the encoder is checkpoints/pbsim_clean/phylum and the reference is drawn
from pbsim_id_test/train.csv, the same leakage-free training set that model saw.

The question is the one that matters for a mixture: `up` is the fraction of the
community drawn from outside the reference, so the test is not whether
darkmatter as a whole looks novel, but whether the FRACTION of reads flagged
novel rises with `up` -- ideally tracking it one-for-one. The threshold is
calibrated on in-distribution reads at a fixed false-positive rate, so
"flagged" means "more novel than (1-fpr) of known reads".

Everything runs at darkmatter's native ~150bp. The encoder was trained at 6kb,
so it is operating well below its native length; that is a property of the
data, not a choice, and it is why these numbers should be read as a floor.
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


def windows(seqs, L, n_per, rng):
    out = []
    for s in seqs:
        if len(s) <= L:
            out.append(s); continue
        for st in rng.integers(0, len(s) - L, size=n_per):
            out.append(s[int(st):int(st) + L])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n-ref-reads", type=int, default=40000, help="training reads -> windows")
    ap.add_argument("--win-per-read", type=int, default=2)
    ap.add_argument("--n-id", type=int, default=6000)
    ap.add_argument("--n-dm", type=int, default=20000)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    L = args.read_len

    print("building reference from the clean training set ...")
    ref = pd.concat([c for c in pd.read_csv(D / "train.csv", usecols=["sequence"],
                                            chunksize=50_000)], ignore_index=True)
    ref = ref.sample(min(len(ref), args.n_ref_reads), random_state=1)
    ref_w = windows(ref.sequence.values, L, args.win_per_read, rng)
    print(f"  {len(ref_w):,} reference windows of {L}bp")

    idt = pd.read_csv(D / "id_test.csv", usecols=["sequence"]).sample(args.n_id, random_state=1)
    id_w = windows(idt.sequence.values, L, 1, rng)

    dm = pd.read_csv(here / "csv_data" / "darkmatter_reads_all.csv")
    per = args.n_dm // dm.up.nunique()
    dm = (dm.groupby("up", group_keys=False)
            .apply(lambda g: g.sample(n=min(per, len(g)), random_state=1))
            .reset_index(drop=True))
    print(f"  id_test windows {len(id_w):,} | darkmatter reads {len(dm):,}")

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
    Qi = embed(m, tok, id_w, args.max_len, args.bs, "id")
    Qd = embed(m, tok, dm.sequence.values, args.max_len, args.bs, "dm")
    del m; torch.cuda.empty_cache()

    def score(E):
        s = np.empty(len(E))
        for i in range(0, len(E), 512):
            sim = E[i:i + 512].to(DEV) @ R.T
            s[i:i + 512] = (-sim.topk(args.k, 1).values[:, -1]).cpu().numpy()
        return s
    s_id, s_dm = score(Qi), score(Qd)
    dm = dm.assign(novelty=s_dm)

    y = np.r_[np.zeros(len(s_id)), np.ones(len(s_dm))]
    print(f"\n=== 1. darkmatter vs in-distribution ===")
    print(f"  mean novelty: id_test={s_id.mean():.4f}  darkmatter={s_dm.mean():.4f}")
    print(f"  AUROC = {roc_auc_score(y, np.r_[s_id, s_dm]):.3f}")

    print(f"\n=== 2. does the FLAGGED FRACTION track `up`? ===")
    out = []
    for fpr in (0.01, 0.05, 0.10):
        thr = np.quantile(s_id, 1 - fpr)
        dm["flagged"] = (dm.novelty > thr).astype(float)
        by = dm.groupby("up").flagged.mean()
        rep = dm[dm.up != 0.5].groupby(["up", "sample"]).flagged.mean().reset_index()
        sl, ic, r, p_, se = stats.linregress(rep.up, rep.flagged)
        print(f"\n  threshold at {fpr:.0%} FPR on id_test")
        print("    flagged: " + "  ".join(f"up={u:.2f}:{v:.3f}" for u, v in by.items()))
        print("    perfect: " + "  ".join(f"up={u:.2f}:{fpr+(1-fpr)*u:.3f}" for u in by.index))
        print(f"    slope={sl:+.4f} +/- {se:.4f}  (perfect {1-fpr:+.2f})  r={r:+.3f}  p={p_:.4f}")
        for u, v in by.items():
            out.append(dict(fpr=fpr, up=u, flagged=v, expected=fpr + (1 - fpr) * u, slope=sl, p=p_))
    pd.DataFrame(out).to_csv("results/darkmatter_clean_flagged.csv", index=False)
    dm.to_csv("results/darkmatter_clean_scores.csv", index=False)
    print("\nwrote results/darkmatter_clean_flagged.csv")


if __name__ == "__main__":
    main()
