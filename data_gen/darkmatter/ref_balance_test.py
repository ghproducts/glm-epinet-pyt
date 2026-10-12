#!/usr/bin/env python3
"""Does phylum-balancing the retrieval reference change the dark matter trend?

Two earlier runs disagreed at loose thresholds: a phylum-balanced reference
(2500 windows per phylum) gave a positive slope at 10% FPR, while a
natural-abundance reference (random training reads) gave a clearly negative one.
Everything else about them also differed, so this isolates the reference:
identical dark matter reads, identical id_test, identical k, equal reference
SIZE, varying only whether phyla are weighted evenly or by their natural
abundance in the training set.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n-ref", type=int, default=50000, help="same size for both references")
    args = ap.parse_args()
    rng = np.random.default_rng(1); L = args.read_len
    cut1 = lambda s: s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]

    lin = pd.read_csv(D / "full_basic_lineage.csv")
    sp2ph = dict(zip(lin.species.astype(str), lin.phylum.astype(str)))
    tr = pd.concat([c.assign(ph=c.taxid.astype(str).map(sp2ph))
                    for c in pd.read_csv(D / "train.csv", usecols=["sequence", "taxid"],
                                         chunksize=100_000)], ignore_index=True).dropna(subset=["ph"])

    # (a) natural abundance: plain random draw
    nat = tr.sample(args.n_ref, random_state=1)
    # (b) phylum-balanced: equal quota per phylum, same total
    per = args.n_ref // tr.ph.nunique() + 1
    bal = (tr.groupby("ph", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), per), random_state=1))
             .sample(frac=1, random_state=1).head(args.n_ref))
    print(f"natural-abundance ref: {len(nat):,} windows, top phylum share "
          f"{nat.ph.value_counts(normalize=True).iloc[0]:.1%}")
    print(f"phylum-balanced   ref: {len(bal):,} windows, top phylum share "
          f"{bal.ph.value_counts(normalize=True).iloc[0]:.1%}")

    idt = [cut1(s) for s in pd.read_csv(D / "id_test.csv", usecols=["sequence"])
                              .sample(6000, random_state=1).sequence]
    dm = pd.read_csv("results/darkmatter_all_methods_reads.csv")   # identical reads to the k-sweep

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    cfg = transformers.AutoConfig.from_pretrained(CLEAN, trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                        trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CLEAN) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()
    Ei = embed(m, tok, idt, args.max_len, args.bs, "id")
    Ed = embed(m, tok, dm.sequence.values, args.max_len, args.bs, "dm")
    refs = {}
    for name, df in [("natural", nat), ("balanced", bal)]:
        refs[name] = embed(m, tok, [cut1(s) for s in df.sequence], args.max_len, args.bs, name)
    del m; torch.cuda.empty_cache()

    fprs = [0.01, 0.05, 0.10, 0.20, 0.50]
    print(f"\n{'reference':<12}{'AUROC id/dm':>13}{'AUROC .25v1':>13}" + "".join(f"{f:>9.0%}" for f in fprs))
    rows = []
    for name, R in refs.items():
        Rd = R.to(DEV)
        def sc(E):
            s = np.empty(len(E))
            for i in range(0, len(E), 512):
                sim = E[i:i + 512].to(DEV) @ Rd.T
                s[i:i + 512] = (-sim.topk(args.k, 1).values[:, -1]).cpu().numpy()
            return s
        si, sd = sc(Ei), sc(Ed)
        d = dm.assign(score=sd)
        auc = roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(sd))], np.r_[si, sd])
        a, b = d[d.up == 0.25].score.values, d[d.up == 1.0].score.values
        aup = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
        line = f"{name:<12}{auc:>13.3f}{aup:>13.3f}"; rec = {"reference": name, "auroc": auc, "auroc_up": aup}
        for f in fprs:
            T = np.quantile(si, 1 - f)
            rep = (d.assign(flag=(d.score > T).astype(float))
                     .groupby(["up", "sample"]).flag.mean().reset_index())
            sl, _, _, p, _ = stats.linregress(rep.up, rep.flag)
            rec[f"slope_{f}"] = sl; rec[f"p_{f}"] = p
            line += f"{sl:>+8.4f}{'*' if p < 0.05 else ' '}"
        rows.append(rec); print(line, flush=True)
    pd.DataFrame(rows).to_csv("results/darkmatter_ref_balance.csv", index=False)


if __name__ == "__main__":
    main()
