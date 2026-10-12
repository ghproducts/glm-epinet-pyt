#!/usr/bin/env python3
"""Retrieval-based novelty scoring on the darkmatter reads.

The original darkmatter analysis asked whether uncertainty tracks `up` (the
fraction of the community drawn from outside the reference) and found nothing:
every classifier-derived UQ score sat at chance, and the apparent trends turned
out to be label-space and GC artifacts. Retrieval is the one method that has
worked on this data family -- 0.84/0.88 AUROC separating known from novel
phyla, against ~0.55 for Kraken2 on the identical reference -- so it is worth
re-asking the darkmatter question with it.

Setup mirrors the working retrieval experiment, at darkmatter's read length:
  reference  150bp windows from the pbsim TRAINING reads (the same data the
             models were trained on), all of them, no sampling
  queries    darkmatter reads, plus the id_test / near tiers at 150bp so the
             score has a known scale to be read against
  score      negative cosine similarity to the nearest reference embedding

Two questions:
  1. Are darkmatter reads scored as more novel than in-distribution reads?
  2. Does that novelty score increase with `up`, as it should if `up` is the
     fraction of genuinely unknown organisms?
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
FINETUNED = "old_models/trained_models_1/NT_transformer/pbsim_phylum/checkpoint-60708"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--v2", type=Path, default=here.parent / "leave_phyla_out_v2")
    ap.add_argument("--dm", type=Path, default=here / "csv_data" / "darkmatter_reads_all.csv")
    ap.add_argument("--n-dm", type=int, default=20000, help="darkmatter reads, stratified by up")
    ap.add_argument("--max-len", type=int, default=48)   # 150bp / 6-mer ~ 25 tokens
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()
    rng = np.random.default_rng(1)

    # reference: every distinct 150bp training window, all 10 phyla
    pool = set()
    for i in range(10):
        pool |= set(pd.read_csv(args.v2 / "subsets" / f"subset_{i}.csv").sequence)
    pool = sorted(pool)
    print(f"reference: {len(pool):,} training windows (150bp)")

    idt = pd.read_csv(args.v2 / "eval_id_test_150bp.csv")
    near = pd.read_csv(args.v2 / "eval_near_150bp.csv")
    dm = pd.read_csv(args.dm)
    # stratify by `up` without dropping the grouping column (include_groups=False
    # removes it, which is what broke the downstream trend analysis)
    # up=0.5 is a single file (3000 reads), far fewer than the other levels,
    # so cap the per-group draw at whatever that group actually has
    per = args.n_dm // dm.up.nunique()
    dm = (dm.groupby("up", group_keys=False)
            .apply(lambda g: g.sample(n=min(per, len(g)), random_state=1))
            .reset_index(drop=True))
    print(f"queries: id_test={len(idt)} near={len(near)} darkmatter={len(dm)}")

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    cfg = transformers.AutoConfig.from_pretrained(FINETUNED, trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                       trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(FINETUNED) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()

    R = embed(m, tok, pool, args.max_len, args.bs, "ref").to(DEV)
    Q = {"id_test": embed(m, tok, idt.sequence.values, args.max_len, args.bs, "idt"),
         "near": embed(m, tok, near.sequence.values, args.max_len, args.bs, "near"),
         "darkmatter": embed(m, tok, dm.sequence.values, args.max_len, args.bs, "dm")}
    del m; torch.cuda.empty_cache()

    def score(E):
        s = np.empty(len(E))
        for i in range(0, len(E), 512):
            sim = E[i:i + 512].to(DEV) @ R.T
            s[i:i + 512] = (-sim.topk(args.k, 1).values[:, -1]).cpu().numpy()
        return s
    S = {k: score(v) for k, v in Q.items()}

    def auc(a, b):
        return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
    print("\n=== 1. is darkmatter scored as novel? ===")
    print(f"  mean novelty score: id_test={S['id_test'].mean():.4f} "
          f"near={S['near'].mean():.4f} darkmatter={S['darkmatter'].mean():.4f}")
    print(f"  AUROC id_test vs darkmatter : {auc(S['id_test'], S['darkmatter']):.3f}")
    print(f"  AUROC id_test vs near       : {auc(S['id_test'], S['near']):.3f}")
    print(f"  AUROC near   vs darkmatter  : {auc(S['near'], S['darkmatter']):.3f}")

    dm = dm.assign(novelty=S["darkmatter"])

    # ---- the question that actually matters ----------------------------
    # `up` is a MIXTURE PROPORTION: at up=0.25 roughly a quarter of the reads
    # come from organisms outside the reference, at up=1.0 all of them do. So
    # the test is not whether darkmatter as a whole scores more novel than ID
    # (a single AUROC), but whether the FRACTION of reads flagged novel rises
    # with `up` -- and ideally tracks it one-for-one.
    #
    # The threshold is calibrated on the in-distribution tier at a fixed false
    # positive rate, so "flagged" means "more novel than (1-fpr) of known
    # reads". Under perfect detection the flagged fraction would be
    # approximately fpr + (1-fpr)*up, i.e. slope ~1 against `up`.
    print("\n=== 2. does the FLAGGED FRACTION track `up`? ===")
    out = []
    for fpr in (0.01, 0.05, 0.10):
        thr = np.quantile(S["id_test"], 1 - fpr)
        dm["flagged"] = (dm.novelty > thr).astype(float)
        by_up = dm.groupby("up").flagged.mean()
        d2 = dm[dm.up != 0.5]
        rep = d2.groupby(["up", "sample"]).flagged.mean().reset_index()
        sl, ic, r, p_, se = stats.linregress(rep.up, rep.flagged)
        print(f"\n  threshold at {fpr:.0%} FPR on ID (score > {thr:.4f})")
        print("    flagged fraction by up: " +
              "  ".join(f"up={u:.2f}:{v:.3f}" for u, v in by_up.items()))
        print(f"    expected if detection were perfect: " +
              "  ".join(f"up={u:.2f}:{fpr + (1-fpr)*u:.3f}" for u in by_up.index))
        print(f"    slope vs up = {sl:+.4f} +/- {se:.4f}  (perfect = {1-fpr:+.2f})  "
              f"r={r:+.3f}  p={p_:.4f}")
        for u, v in by_up.items():
            out.append(dict(fpr=fpr, threshold=thr, up=u, flagged=v,
                            expected_perfect=fpr + (1 - fpr) * u))
        out.append(dict(fpr=fpr, threshold=thr, up=np.nan, flagged=np.nan,
                        expected_perfect=np.nan, slope=sl, slope_se=se, p=p_))
    pd.DataFrame(out).to_csv("results/darkmatter_flagged_fraction.csv", index=False)

    # mean-score trend kept for continuity with the earlier UQ analyses
    d2 = dm[dm.up != 0.5]
    rep = d2.groupby(["up", "sample"]).novelty.mean().reset_index()
    sl, ic, r, p_, se = stats.linregress(rep.up, rep.novelty)
    print(f"\n  (mean-score trend, for comparison with the UQ results: "
          f"slope={sl:+.5f} +/- {se:.5f}  r={r:+.3f}  p={p_:.4f})")

    dm.to_csv("results/darkmatter_retrieval_scores.csv", index=False)
    np.save("results/darkmatter_idtest_scores.npy", S["id_test"])


if __name__ == "__main__":
    main()
