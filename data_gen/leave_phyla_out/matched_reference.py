#!/usr/bin/env python3
"""Kraken2 vs GLM retrieval over IDENTICAL reference sets, at several coverage levels.

Every comparison so far gave the two methods different reference information:
Kraken2 got complete genomes (5.7GB) while the GLM scores were computed against
either the model's own training windows (0.18% locus coverage) or a sparse
genome sample. Kraken2's results turned out to be essentially a function of
coverage --

    0.18% coverage -> AUROC 0.500
    3.6%  coverage -> AUROC ~0.55
    100%  coverage -> AUROC 0.975 / 0.988

-- so any comparison that does not hold coverage fixed is measuring the
database, not the method.

Here one set of genome chunks is built, persisted, and handed to both methods:
Kraken2 indexes exactly those sequences, the GLM embeds exactly those
sequences. Repeating across coverage levels gives a dose-response curve for
each, which is the actual question: exact k-mer matching needs the specific
locus, while a learned embedding should generalise across loci -- if so, the
GLM curve should stay above Kraken2's as coverage falls.
"""
from __future__ import annotations
import argparse, subprocess, shutil
from pathlib import Path
import numpy as np, pandas as pd, torch, transformers
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
FINETUNED = "old_models/trained_models_1/NT_transformer/pbsim_phylum/checkpoint-60708"
KRAKEN = "/home/glh52/miniconda3/envs/kraken2/bin"
TAXO = Path("/scratch/home/glh52/kraken2_db_train/taxonomy")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GENOME_MB = 5710.0          # total training-genome sequence, for the coverage axis


def build_chunks(n, L, rng, per_species_cap):
    R = Path("data_gen/pbsim/raw_sequences")
    lin = pd.read_csv("/scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv")
    sp2ph = dict(zip(lin.species.astype(str), lin.phylum.astype(str)))
    sp = open("data_gen/pbsim/splits/train_species.txt").read().split()
    rng.shuffle(sp)
    rows = []
    for s in sp:
        if len(rows) >= n:
            break
        for fna in (R / s).glob("*.fna"):
            seq = "".join(l.strip() for l in fna.open() if not l.startswith(">")).upper()
            if len(seq) < L:
                continue
            for st in rng.integers(0, len(seq) - L, size=per_species_cap):
                rows.append({"sequence": seq[int(st):int(st) + L],
                             "taxid": s, "phylum": sp2ph.get(s, "1")})
            break
    rng.shuffle(rows)
    return pd.DataFrame(rows[:n])


def kraken_eval(ref: pd.DataFrame, tiers, work: Path, threads):
    db = work / "db"
    if db.exists():
        shutil.rmtree(db)
    (db / "taxonomy").mkdir(parents=True)
    for f in ("nodes.dmp", "names.dmp"):
        shutil.copy2(TAXO / f, db / "taxonomy" / f)
    lib = work / "lib.fna"
    with lib.open("w") as f:
        for j, (s, t) in enumerate(zip(ref.sequence, ref.taxid)):
            f.write(f">c{j}|kraken:taxid|{t}\n{s}\n")
    for cmd in (["--add-to-library", str(lib)], ["--build"]):
        subprocess.run([f"{KRAKEN}/kraken2-build", *cmd, "--db", str(db),
                        "--threads", str(threads)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = {}
    for t, d in tiers.items():
        fa = work / f"{t}.fa"
        with fa.open("w") as f:
            for i, s in enumerate(d.sequence):
                f.write(f">q{i}\n{s}\n")
        res = work / f"{t}.out"
        subprocess.run([f"{KRAKEN}/kraken2", "--db", str(db), "--threads", str(threads),
                        "--output", str(res), "--report", str(work / f"{t}.rep"), str(fa)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fr = []
        for line in res.open():
            p = line.rstrip("\n").split("\t")
            hit = tot = 0
            for ch in p[4].split():
                a, b = ch.rsplit(":", 1)
                if b.isdigit():
                    tot += int(b)
                    if a not in ("0", "A"):
                        hit += int(b)
            fr.append(hit / tot if tot else 0.0)
        out[t] = -np.array(fr)     # low k-mer hit fraction => novel
    shutil.rmtree(db, ignore_errors=True)
    return out


@torch.no_grad()
def embed(m, tok, seqs, L, bs=16):
    o = []
    for i in range(0, len(seqs), bs):
        b = tok(list(seqs[i:i + bs]), return_tensors="pt", padding=True,
                truncation=True, max_length=L).to(DEV)
        h = m(**b, output_hidden_states=True).hidden_states[-1]
        mk = b["attention_mask"].unsqueeze(-1).float()
        e = (h * mk).sum(1) / mk.sum(1)
        o.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
    return torch.cat(o)


def glm_eval(ref: pd.DataFrame, tiers, tok, m, max_len, k=50):
    R = embed(m, tok, ref.sequence.values, max_len).to(DEV)
    out = {}
    for t, d in tiers.items():
        Q = embed(m, tok, d.sequence.values, max_len)
        s = np.empty(len(Q))
        for i in range(0, len(Q), 256):
            sim = Q[i:i + 256].to(DEV) @ R.T
            s[i:i + 256] = (-sim.topk(k, 1).values[:, -1]).cpu().numpy()
        out[t] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--sizes", type=int, nargs="*", default=[5000, 20000, 80000])
    ap.add_argument("--read-len", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--work", type=Path, default=Path("/scratch/home/glh52/matched_ref"))
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    root = here.parent / "leave_phyla_out_v3"
    tiers = {t: pd.read_csv(root / f"eval_{t}_{args.read_len}bp.csv")
             for t in ["id_test", "near", "far"]}
    tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True,
                                                     model_max_length=args.max_len,
                                                     padding_side="right", use_fast=True)
    tok.eos_token = tok.pad_token
    cfg = transformers.AutoConfig.from_pretrained(FINETUNED, trust_remote_code=True)
    m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg,
                                                                       trust_remote_code=True)
    m.load_state_dict(load_file(str(Path(FINETUNED) / "model.safetensors")), strict=True)
    m = m.to(DEV).eval().half()

    def auc(a, b):
        return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])

    rows = []
    for n in args.sizes:
        cap = max(1, min(40, n // 400))
        ref = build_chunks(n, args.read_len, rng, cap)
        ref.to_csv(args.work / f"ref_{n}.csv", index=False)
        cov = len(ref) * args.read_len / 1e6 / GENOME_MB * 100
        print(f"\n=== reference {len(ref)} chunks, {len(ref)*args.read_len/1e6:.0f} Mb, "
              f"{cov:.2f}% genome coverage ===", flush=True)
        ks = kraken_eval(ref, tiers, args.work, args.threads)
        gs = glm_eval(ref, tiers, tok, m, args.max_len)
        for name, sc in [("kraken2", ks), ("glm_retrieval", gs)]:
            r = dict(method=name, n_ref=len(ref), coverage_pct=cov,
                     idtest_v_near=auc(sc["id_test"], sc["near"]),
                     idtest_v_far=auc(sc["id_test"], sc["far"]))
            rows.append(r)
            print(f"  {name:14s} ID/near={r['idtest_v_near']:.3f}  ID/far={r['idtest_v_far']:.3f}",
                  flush=True)
        pd.DataFrame(rows).to_csv("results/matched_reference.csv", index=False)

    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
