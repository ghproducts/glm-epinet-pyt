#!/usr/bin/env python3
"""Is the dark matter failure a sequencing-technology shift?

Everything the NT encoder was fine-tuned on -- and every read in the kNN
reference and every pbsim evaluation tier -- is PBSIM3 QSHMM-RSII output: PacBio
RS II CLR reads at ~85% accuracy, errors mostly indels. Dark matter is ART
Illumina (~0.1% substitutions) and its contigs are consensus, i.e. near
error-free. So dark matter differs from everything the model has seen in error
profile, independently of taxonomy.

Test on genomes whose taxonomy is known: cut ERROR-FREE windows straight from
the raw genome FASTAs of the training species and of each novelty tier, and
score all of them against the usual noisy-read reference.
  - id_test (noisy) vs clean TRAINING-species windows: pure technology shift,
    zero taxonomic novelty. ~0.5 = no shift.
  - clean train vs clean novel family / non-bacterial: taxonomic signal with the
    error profile held equal (both clean).
"""
from __future__ import annotations
import argparse, gzip
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test")
G = Path("data_gen/pbsim/raw_sequences"); S = Path("data_gen/pbsim/splits")
CKPT = "checkpoints/pbsim_clean/phylum/checkpoint-57734"


def genome_windows(species, n, W, rng):
    out, sp = [], list(species)
    while len(out) < n:
        files = list((G / str(rng.choice(sp))).glob("*.fna*"))
        if not files: continue
        f = files[rng.integers(len(files))]
        op = gzip.open if f.suffix == ".gz" else open
        seqs, cur = [], []
        with op(f, "rt") as h:
            for l in h:
                if l.startswith(">"):
                    if cur: seqs.append("".join(cur)); cur = []
                else: cur.append(l.strip().upper())
        if cur: seqs.append("".join(cur))
        seqs = [s for s in seqs if len(s) >= W]
        if not seqs: continue
        s = seqs[rng.integers(len(seqs))]; st = int(rng.integers(0, len(s) - W + 1)); w = s[st:st + W]
        if w.count("N") < 0.01 * W: out.append(w)
    return out


@torch.no_grad()
def embed(m, tok, seqs, ml, bs):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True, truncation=True, max_length=ml).to("cuda")
        h = m(**b, output_hidden_states=True).hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
        out.append(torch.nn.functional.normalize(((h * mk).sum(1) / mk.sum(1)).float(), dim=-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=6000)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-ref", type=int, default=40000)
    a = ap.parse_args(); W = a.window; ml = W // 6; bs = max(4, int(64 * 2000 / W))
    rng = np.random.default_rng(1)
    cut = lambda s: s[(st := int(rng.integers(0, len(s) - W + 1))):st + W]
    sp = {k: [x.strip() for x in open(S / f) if x.strip()] for k, f in
          [("train", "train_species.txt"), ("novel genus", "id_novel_genus_species.txt"),
           ("novel family", "ood_novel_family_species.txt"), ("non-bacterial", "ood_nonbacterial_species.txt")]}
    cfg = transformers.AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True, model_max_length=ml,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg, trust_remote_code=True)
    m.load_state_dict(load_file(f"{CKPT}/model.safetensors"), strict=True); m = m.cuda().eval().half()
    ref = pd.read_csv(D / "train.csv", usecols=["sequence"]).sequence
    R = embed(m, tok, [cut(x) for x in ref[ref.str.len() >= W].sample(a.n_ref, random_state=1)], ml, bs).cuda()
    knn = lambda E: np.concatenate([(-(E[i:i+2048].cuda() @ R.T).max(1).values).cpu().numpy() for i in range(0, len(E), 2048)])
    idt = pd.read_csv(D / "id_test.csv", usecols=["sequence"]).sequence
    S_ = {"id_test (noisy)": knn(embed(m, tok, [cut(x) for x in idt[idt.str.len() >= W].sample(a.n, random_state=1)], ml, bs))}
    nf = pd.read_csv(D / "ood_novel_family.csv", usecols=["sequence"]).sequence
    S_["novel family (noisy)"] = knn(embed(m, tok, [cut(x) for x in nf[nf.str.len() >= W]], ml, bs))
    for k, v in sp.items():
        S_[f"{k} (CLEAN genome)"] = knn(embed(m, tok, genome_windows(v, a.n, W, rng), ml, bs))
        print(f"  embedded clean {k}", flush=True)
    auc = lambda x, y: roc_auc_score(np.r_[np.zeros(len(S_[x])), np.ones(len(S_[y]))], np.r_[S_[x], S_[y]])
    print(f"\n=== NT phylum, {W}bp windows, kNN(k=1) vs the noisy-read reference ===")
    print(f"  mean nearest-neighbour similarity:")
    for k, v in S_.items(): print(f"    {k:28s} {-v.mean():.4f}")
    print(f"\n  technology shift alone (no taxonomic novelty):")
    print(f"    id_test noisy  vs train CLEAN          AUROC {auc('id_test (noisy)', 'train (CLEAN genome)'):.4f}")
    print(f"  taxonomic signal, both sides noisy (the original pbsim result):")
    print(f"    id_test noisy  vs novel family noisy   AUROC {auc('id_test (noisy)', 'novel family (noisy)'):.4f}")
    print(f"  taxonomic signal, both sides CLEAN:")
    for k in ["novel genus", "novel family", "non-bacterial"]:
        print(f"    train CLEAN vs {k+' CLEAN':22s} AUROC {auc('train (CLEAN genome)', k+' (CLEAN genome)'):.4f}")


if __name__ == "__main__":
    main()
