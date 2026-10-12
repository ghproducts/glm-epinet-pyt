#!/usr/bin/env python3
"""Cut every dark matter contig >= 1.5kb into non-overlapping 1.5kb windows.

Matches the clean1500 benchmark's unit exactly. Keeps contig id, window index
and MEGAHIT coverage (multi=) so results can be aggregated per contig or
coverage-weighted. 150bp reads are out of the models' range and not used.
"""
import re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_contigs import read_contigs, ASM
L = 1500
rows = []
for d in sorted(ASM.glob("up*_s*")):
    up = float(re.search(r"up([0-9.]+)_", d.name).group(1)); s = int(d.name.split("_s")[1])
    seqs, cov = read_contigs(d / "asm" / "final.contigs.fa")
    for c, (x, v) in enumerate(zip(seqs, cov)):
        for w in range(len(x) // L):
            win = x[w * L:(w + 1) * L]
            if set(win) <= set("ACGT"):
                rows.append((up, s, c, w, v, len(x), win))
df = pd.DataFrame(rows, columns=["up", "sample", "contig", "window", "cov", "contig_len", "sequence"])
out = Path("/scratch/home/glh52/glm-epinet/DATA/clean1500/darkmatter_contig_windows.csv")
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df):,} windows from {df.groupby(['up','sample']).contig.nunique().sum():,} contigs")
print(df.groupby("up").agg(windows=("sequence", "size"), contigs=("contig", "nunique")).to_string())
