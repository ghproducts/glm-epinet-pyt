#!/usr/bin/env python3
"""Write the TRAINING regions of the clean1500 benchmark as a Kraken2 library FASTA.

Replays build_fragments.regions() with the same seed and species order (regions
is the only RNG consumer during that loop, so the replay is exact), then checks
it: every train fragment must lie inside a replayed train region and no val/
id_test fragment may. Output excludes the held-out val/test blocks, so the
database contains the same genome regions the models trained on and nothing
from the ID test.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fragments import read_records, regions, SPLITS
D = Path("/scratch/home/glh52/glm-epinet/DATA/clean1500")
out = Path("/scratch/home/glh52/kraken2_db_clean1500/library"); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
sp = pd.read_csv(SPLITS); sp = sp[sp.has_sequence]; tr = sp[sp.split == "train"]
segs = {}
with open(out / "train_regions.fna", "w") as f:
    for s in tr.species.astype(int):
        recs = read_records(str(s))
        if not recs: continue
        rg = regions(recs, rng); segs[s] = (recs, rg)
        for k, (i, a, b) in enumerate(rg["train"]):
            f.write(f">s{s}_{k}|kraken:taxid|{s}\n{recs[i][a:b]}\n")
# verification on samples
def inside(s, frag, key):
    recs, rg = segs[s]
    return any(frag in recs[i][a:b] for i, a, b in rg[key])
for name, key, want in [("train", "train", True), ("val", "train", False), ("id_test", "train", False),
                        ("val", "val", True), ("id_test", "test", True)]:
    d = pd.read_csv(D / f"{name}.csv").sample(300, random_state=0)
    ok = np.mean([inside(s, x, key) == want for s, x in zip(d.taxid, d.sequence)])
    print(f"  {name:8s} fragments {'inside' if want else 'outside'} replayed {key} regions: {ok:.1%}")
print("wrote", out / "train_regions.fna")
