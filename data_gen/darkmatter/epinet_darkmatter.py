#!/usr/bin/env python3
"""Epinet scores on the dark matter set, matching all_methods_darkmatter.py.

Epinet needs the EpinetWrapper/HFEpinetSeqClassifier stack rather than a plain
forward pass, so it is computed here and merged into the same comparison table.
Construction mirrors nn_proj/models/NT_transformer/inference.py exactly (conv
prior, include_inputs, a dummy forward to build internals before loading
weights), against the leakage-free epinet head.

K index samples give a [K, N, C] logit stack, from which predictive entropy and
the BALD epistemic term are taken -- the same two quantities MC dropout
contributes to the table.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nn_proj.models.epinet import (EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier,
                                   MLPEpinetWithConvPrior)
from nn_proj.models.epinet.feature_fns import NT_feature_fn

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
BASE = "checkpoints/pbsim_clean/phylum/checkpoint-57734"
EPI = "checkpoints/pbsim_clean_epinet/phylum/checkpoint-57734"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build(tok, max_len, num_labels):
    cfg = transformers.AutoConfig.from_pretrained(BASE, trust_remote_code=True)
    base = transformers.AutoModelForSequenceClassification.from_pretrained(
        HUB, config=cfg, trust_remote_code=True)
    epi_cfg = EpinetConfig(num_classes=num_labels, include_inputs=True,
                           vocab_size=cfg.vocab_size)
    wrapper = EpinetWrapper(base, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=8).to(DEV)
    with torch.no_grad():                      # build lazily-created internals
        d = tok("ACGT", truncation=True, padding="max_length", max_length=max_len,
                return_tensors="pt")
        model(**{k: v.to(DEV) for k, v in d.items()})
    model.load_state_dict(load_file(str(Path(EPI) / "model.safetensors")), strict=True)
    return model.eval()


@torch.no_grad()
def logits_k(model, tok, seqs, L, bs, k, tag=""):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        # HFEpinetSeqClassifier.forward returns {"loss","logits"} with logits
        # already averaged over k_eval samples. The per-sample stack needed for
        # the BALD decomposition comes from the wrapper directly.
        batch = {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
        out.append(model.wrapper(batch, n_index_samples=k, return_all=True).float().cpu())
        if tag and i % (bs * 100) == 0:
            print(f"    {tag} {i}/{len(seqs)}", flush=True)
    return torch.cat(out, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read-len", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--num-labels", type=int, default=36)
    args = ap.parse_args()
    rng = np.random.default_rng(1); L = args.read_len
    cut = lambda ss: [s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]
                      for s in ss]

    # exactly the same query sets as all_methods_darkmatter.py
    idt = cut(pd.read_csv(D / "id_test.csv", usecols=["sequence"])
                .sample(6000, random_state=1).sequence.values)
    dm = pd.read_csv("results/darkmatter_all_methods_reads.csv")
    print(f"id {len(idt):,} | dm {len(dm):,}")

    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    model = build(tok, args.max_len, args.num_labels)
    Gi = logits_k(model, tok, idt, args.max_len, args.bs, args.k, "id")
    Gd = logits_k(model, tok, dm.sequence.values, args.max_len, args.bs, args.k, "dm")
    del model; torch.cuda.empty_cache()

    def bald(G):
        p = torch.softmax(G, -1); pb = p.mean(0)
        ent = lambda q: -(q * torch.log(q + 1e-12)).sum(-1)
        return ent(pb).numpy(), (ent(pb) - ent(p).mean(0)).numpy(), (-pb.max(1).values).numpy()
    ei, epi_i, mi = bald(Gi)
    ed, epi_d, md = bald(Gd)

    rows = []
    for name, si, sd in [("epinet_msp", mi, md), ("epinet_entropy", ei, ed),
                         ("epinet_epistemic", epi_i, epi_d)]:
        auc = roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(sd))], np.r_[si, sd])
        r = {"method": name, "auroc_dm_vs_id": auc}
        d2 = dm.assign(score=sd)
        for fpr in (0.01, 0.05, 0.10, 0.20):
            thr = np.quantile(si, 1 - fpr)
            rep = (d2.assign(flag=(d2.score > thr).astype(float))
                     .groupby(["up", "sample"]).flag.mean().reset_index())
            sl, _, _, pp, _ = stats.linregress(rep.up, rep.flag)
            r[f"slope_{int(fpr*100)}"] = sl; r[f"p_{int(fpr*100)}"] = pp
        rows.append(r)
        np.save(f"results/dm_scores_{name}_id.npy", si)
        np.save(f"results/dm_scores_{name}_dm.npy", sd)
        print(f"  {name:18s} AUROC={auc:.3f}  " +
              "  ".join(f"{f}%:{r[f'slope_{f}']:+.4f}(p={r[f'p_{f}']:.3f})" for f in (1,5,10,20)),
              flush=True)
    pd.DataFrame(rows).to_csv("results/darkmatter_epinet_methods.csv", index=False)


if __name__ == "__main__":
    main()
