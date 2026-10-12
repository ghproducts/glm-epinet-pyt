#!/usr/bin/env python3
"""Does the NT encoder's novelty score depend on GC within known-novelty data?

The dark matter GC confound rests on reads closer to the training set's GC
looking more familiar. That was only ever measured on dark matter, where GC is
tied to UP. Here it is tested on PBsim tiers, where novelty is known: within
id_test every read is from a training species, so any GC dependence there is a
bias of the model/reference, not novelty. Same setup as the dark matter runs
(NT phylum, 150bp windows, kNN k=1 to an 80k training-window reference).
"""
import numpy as np, pandas as pd, torch, transformers
from pathlib import Path
from safetensors.torch import load_file
from scipy import stats
HUB = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
D = Path("/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test"); CK = "checkpoints/pbsim_clean/phylum/checkpoint-57734"
L, ML, BS = 150, 48, 384
rng = np.random.default_rng(1)
cut = lambda s: s if len(s) <= L else s[(st := int(rng.integers(0, len(s) - L))):st + L]
gc = lambda s: (s.count("G") + s.count("C")) / max(len(s), 1)
cfg = transformers.AutoConfig.from_pretrained(CK, trust_remote_code=True)
tok = transformers.AutoTokenizer.from_pretrained(HUB, trust_remote_code=True, model_max_length=ML, padding_side="right", use_fast=True)
tok.eos_token = tok.pad_token
m = transformers.AutoModelForSequenceClassification.from_pretrained(HUB, config=cfg, trust_remote_code=True)
m.load_state_dict(load_file(f"{CK}/model.safetensors"), strict=True); m = m.cuda().eval().half()
@torch.no_grad()
def emb(seqs):
    out = []
    for i in range(0, len(seqs), BS):
        b = tok(list(seqs[i:i + BS]), return_tensors="pt", padding=True, truncation=True, max_length=ML).to("cuda")
        h = m(**b, output_hidden_states=True).hidden_states[-1]; mk = b["attention_mask"].unsqueeze(-1).float()
        out.append(torch.nn.functional.normalize(((h * mk).sum(1) / mk.sum(1)).float(), dim=-1))
    return torch.cat(out)
ref = [cut(s) for s in pd.read_csv(D / "train.csv", usecols=["sequence"]).sample(80000, random_state=1).sequence]
R = emb(ref); ref_gc = np.array([gc(s) for s in ref]); mu = ref_gc.mean()
print(f"reference GC mean {mu:.4f}")
rows = []
for t in ["id_test", "id_novel_genus", "ood_novel_family", "ood_nonbacterial"]:
    w = [cut(s) for s in pd.read_csv(D / f"{t}.csv", usecols=["sequence"]).sequence]
    E = emb(w); v = np.concatenate([(-(E[i:i + 4096] @ R.T).max(1).values).cpu().numpy() for i in range(0, len(E), 4096)])
    g = np.array([gc(s) for s in w])
    r1, p1 = stats.spearmanr(g, v); r2, p2 = stats.spearmanr(np.abs(g - mu), v)
    q = pd.qcut(g, 5, labels=False, duplicates="drop")
    dec = [(g[q == k].mean(), v[q == k].mean()) for k in range(q.max() + 1)]
    print(f"\n{t} (n={len(w)})")
    print(f"  spearman(novelty, GC)              {r1:+.3f}  p={p1:.1e}")
    print(f"  spearman(novelty, |GC - ref mean|) {r2:+.3f}  p={p2:.1e}")
    print("  GC quintile mean GC : " + " ".join(f"{a:.3f}" for a, _ in dec))
    print("  mean novelty        : " + " ".join(f"{b:+.4f}" for _, b in dec))
    rows.append(dict(tier=t, n=len(w), rho_gc=r1, rho_absdev=r2))
pd.DataFrame(rows).to_csv("results/gc_bias_check_pbsim.csv", index=False)
