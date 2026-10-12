#!/usr/bin/env python3
"""How much does read length cost the GLM's novelty detection?

Dark matter is 150bp; the encoders were fine-tuned on 6kb. Every dark matter
result is therefore reported in a regime the model was never trained for, and
the natural question is whether 150bp is simply too short for this class of
method. That cannot be answered on dark matter itself, because its true novelty
composition is unknown -- but it CAN be answered on the pbsim tiers, where the
taxonomic relationship of every read to the training set is known exactly.

So: hold everything fixed and sweep only the window length, scoring the same
novelty ladder (novel genus / novel family / non-bacterial vs id_test) at each.
If AUROC climbs steeply with length, 150bp is the binding constraint and
assembling the dark matter reads first is worth the effort. If it is flat, the
problem is elsewhere and assembly will not rescue it.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
CKPT = {"phylum": "checkpoints/pbsim_clean/phylum/checkpoint-57734",
        "family": "checkpoints/pbsim_clean/family/checkpoint-57734"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TIERS = [("id_novel_genus", "novel genus"), ("ood_novel_family", "novel family"),
         ("ood_nonbacterial", "non-bacterial")]


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
        if tag and i % (bs * 100) == 0:
            print(f"      {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", default="phylum", choices=["phylum", "family"])
    ap.add_argument("--lengths", type=int, nargs="*", default=[150, 300, 600, 1200, 2400, 6000])
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--n-query", type=int, default=3000)
    args = ap.parse_args()

    cfg = transformers.AutoConfig.from_pretrained(CKPT[args.rank], trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB, config=cfg, trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(CKPT[args.rank]) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()

    def take(fn, n):           # tiers differ in size (novel_family has only 1940 reads)
        d = pd.read_csv(D / fn, usecols=["sequence"])
        return d.sample(min(n, len(d)), random_state=1).sequence
    ref_src = take("train.csv", args.n_ref)
    idt_src = take("id_test.csv", args.n_query)
    tier_src = {t: take(f"{t}.csv", args.n_query) for t, _ in TIERS}
    print("query sizes: id_test %d, %s" % (len(idt_src),
          ", ".join(f"{t} {len(v)}" for t, v in tier_src.items())), flush=True)

    rows = []
    for L in args.lengths:
        rng = np.random.default_rng(1)                       # same windows start per length
        cut = lambda s: s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]
        ml = max(16, L // 6)                                 # NT is 6-mer
        bs = max(4, int(256 * 150 / L))
        print(f"\n[L={L}bp, {ml} tokens, bs={bs}]", flush=True)
        tok_l = transformers.AutoTokenizer.from_pretrained(
            HUB, trust_remote_code=True, model_max_length=ml, padding_side="right", use_fast=True)
        tok_l.eos_token = tok_l.pad_token
        R = embed(m, tok_l, [cut(s) for s in ref_src], ml, bs, "ref")
        Ei = embed(m, tok_l, [cut(s) for s in idt_src], ml, bs, "id")
        Rd = R.to(DEV)
        def knn(E, k=1):
            s = np.empty(len(E))
            for i in range(0, len(E), 512):
                sim = E[i:i + 512].to(DEV) @ Rd.T
                s[i:i + 512] = (-sim.topk(k, 1).values[:, -1]).cpu().numpy()
            return s
        si = knn(Ei)
        r = {"length": L, "tokens": ml, "id_mean_nn": float(-si.mean())}
        for t, lab in TIERS:
            Et = embed(m, tok_l, [cut(s) for s in tier_src[t]], ml, bs, t[:10])
            st = knn(Et)
            r[lab] = roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(st))], np.r_[si, st])
        del Rd; torch.cuda.empty_cache()
        rows.append(r)
        print(f"  L={L}: " + "  ".join(f"{lab} {r[lab]:.4f}" for _, lab in TIERS), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"results/length_sweep_{args.rank}.csv", index=False)
    print(f"\n=== k-NN(k=1) detection AUROC vs read length ({args.rank} encoder) ===")
    print(f"{'length':>8}{'tokens':>8}" + "".join(f"{lab:>15}" for _, lab in TIERS)
          + f"{'mean NN sim (ID)':>19}")
    for _, r in out.iterrows():
        print(f"{int(r.length):>8}{int(r.tokens):>8}"
              + "".join(f"{r[lab]:>15.4f}" for _, lab in TIERS) + f"{r.id_mean_nn:>19.4f}")


if __name__ == "__main__":
    main()
