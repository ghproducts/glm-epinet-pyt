#!/usr/bin/env python3
"""GLM-as-retrieval: nearest-neighbour novelty detection against an embedded
reference genome database.

Rationale. Kraken2 detects novelty by failing to find exact k-mer matches, so
it needs comprehensive reference coverage and is brittle to divergence. A GLM
generalises across loci and organisms -- which is why it classifies unseen
species at ~85% -- but that same generalisation is why its softmax carries no
novelty signal. Retrieval combines the two: use the GLM only as an encoder, and
get the "is this known?" signal from distance to a densely-covered reference,
the way Kraken2 gets it from k-mer presence.

This is the experiment the earlier kNN/Mahalanobis scores could not be. Those
used each model's own 40,000 training windows as the reference -- 0.18% locus
coverage, i.e. retrieval against an essentially empty index. Here the reference
is chunked training GENOMES, the same data Kraken2's database was built from,
so the comparison isolates exact k-mer matching vs learned embedding distance.

Encoders compared:
  finetuned  the 36-way phylum checkpoint (trained at 6kb on these genomes)
  pretrained raw NT, no fine-tuning -- shows whether task fine-tuning helps or
             hurts the embedding space for retrieval
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
FINETUNED = "old_models/trained_models_1/NT_transformer/pbsim_phylum/checkpoint-60708"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(kind: str):
    if kind == "pretrained":
        m = transformers.AutoModelForSequenceClassification.from_pretrained(
            HUB, num_labels=2, trust_remote_code=True)
    else:
        cfg = transformers.AutoConfig.from_pretrained(FINETUNED, trust_remote_code=True)
        m = transformers.AutoModelForSequenceClassification.from_pretrained(
            HUB, config=cfg, trust_remote_code=True)
        m.load_state_dict(load_file(str(Path(FINETUNED) / "model.safetensors")), strict=True)
    return m.to(DEV).eval().half()


@torch.no_grad()
def embed(m, tok, seqs, L, bs=16):
    out = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        o = m(**b, output_hidden_states=True)
        h = o.hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        if i % (bs * 200) == 0:
            print(f"    {i}/{len(seqs)}", flush=True)
    return torch.cat(out)


def build_reference(n_chunks: int, L: int, rng):
    """Chunk the training-species reference genomes -- the same source Kraken2
    indexed, so the two methods see identical reference information."""
    R = Path("data_gen/pbsim/raw_sequences")
    sp = [s for s in open("data_gen/pbsim/splits/train_species.txt").read().split()]
    rng.shuffle(sp)
    chunks, per = [], max(1, n_chunks // len(sp) + 1)
    for s in sp:
        for fna in (R / s).glob("*.fna"):
            seq = "".join(l.strip() for l in fna.open() if not l.startswith(">")).upper()
            if len(seq) < L:
                continue
            for st in rng.integers(0, len(seq) - L, size=min(per, 8)):
                chunks.append(seq[int(st):int(st) + L])
            break
        if len(chunks) >= n_chunks:
            break
    rng.shuffle(chunks)
    return chunks[:n_chunks]


def knn_dist(ref: torch.Tensor, q: torch.Tensor, k: int):
    """Negative similarity to the k-th nearest reference embedding."""
    out = np.empty(len(q))
    ref = ref.to(DEV)
    for i in range(0, len(q), 256):
        sim = (q[i:i + 256].to(DEV) @ ref.T)
        out[i:i + 256] = (-sim.topk(k, dim=1).values[:, -1]).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--n-ref", type=int, default=60000)
    ap.add_argument("--read-len", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--encoders", nargs="*", default=["finetuned", "pretrained"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    root = here.parent / "leave_phyla_out_v3"
    tiers = {t: pd.read_csv(root / f"eval_{t}_{args.read_len}bp.csv")
             for t in ["id_test", "near", "far"]}
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token

    print(f"building reference: {args.n_ref} chunks of {args.read_len}bp from training genomes")
    ref_seqs = build_reference(args.n_ref, args.read_len, rng)
    print(f"  got {len(ref_seqs)} chunks")

    rows = []
    for kind in args.encoders:
        print(f"\n=== encoder: {kind} ===")
        m = load_encoder(kind)
        print("  embedding reference ...")
        R = embed(m, tok, ref_seqs, args.max_len)
        scores = {}
        for t, d in tiers.items():
            print(f"  embedding {t} ...")
            Q = embed(m, tok, d.sequence.values, args.max_len)
            scores[t] = knn_dist(R, Q, args.k)
        del m; torch.cuda.empty_cache()

        def auc(a, b):
            return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
        r = dict(encoder=kind,
                 idtest_v_near=auc(scores["id_test"], scores["near"]),
                 idtest_v_far=auc(scores["id_test"], scores["far"]),
                 near_v_far=auc(scores["near"], scores["far"]))
        rows.append(r)
        print(f"  AUROC  ID/near={r['idtest_v_near']:.3f}  ID/far={r['idtest_v_far']:.3f}  "
              f"near/far={r['near_v_far']:.3f}")
        pd.DataFrame(rows).to_csv("results/glm_retrieval.csv", index=False)

    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nreference: Kraken2 on the same genomes at 6kb -> ID/near 0.975, ID/far 0.988")


if __name__ == "__main__":
    main()
