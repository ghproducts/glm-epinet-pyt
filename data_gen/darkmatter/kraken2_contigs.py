#!/usr/bin/env python3
"""Kraken2 (read-built DB) on the assembled contigs >= 2kb: fraction unclassified per sample."""
import re, subprocess, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
K="/home/glh52/miniconda3/envs/kraken2/bin/kraken2"; DB="/scratch/home/glh52/kraken2_db_reads"
ASM=Path("/scratch/home/glh52/darkmatter_assembly"); TMP=Path("/scratch/home/glh52/kraken2_tmp_reads")
rows=[]
for d in sorted(ASM.glob("up*_s*")):
    fa=d/"asm"/"final.contigs.fa"; up=float(re.search(r"up([0-9.]+)_",d.name).group(1)); s=int(d.name.split("_s")[1])
    seqs,cur=[],[]
    for l in fa.open():
        if l.startswith(">"):
            if cur: seqs.append("".join(cur)); cur=[]
        else: cur.append(l.strip())
    if cur: seqs.append("".join(cur))
    seqs=[x for x in seqs if len(x)>=2000]
    q=TMP/f"contigs_{d.name}.fa"; q.write_text("".join(f">c{i}\n{x}\n" for i,x in enumerate(seqs)))
    out=TMP/f"contigs_{d.name}.kraken"
    subprocess.run([K,"--db",DB,"--threads","32","--output",str(out),str(q)],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    uc=np.mean([l.startswith("U") for l in out.open()])
    rows.append((up,s,len(seqs),uc))
df=pd.DataFrame(rows,columns=["up","sample","n","v"]); df.to_csv("results/kraken2_contigs.csv",index=False)
sl,_,r,p,_=stats.linregress(df.up,df.v)
lo,hi=df[df.up==0.25].v,df[df.up==1.0].v
d_=(hi.mean()-lo.mean())/np.sqrt((lo.var(ddof=1)+hi.var(ddof=1))/2)
print("=== Kraken2 on contigs >=2kb: fraction UNclassified ===")
print(df.groupby("up").agg(n=("n","sum"),unclassified=("v","mean")).to_string())
print(f"  slope {sl:+.4f}  R2 {r*r:.3f}  d {d_:+.2f}  p {p:.2e}")
