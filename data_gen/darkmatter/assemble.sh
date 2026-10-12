#!/usr/bin/env bash
# Per-sample MEGAHIT assembly of the dark matter reads (15 samples: up 0.25/0.75/1.0 x 5).
#
# The paired_1/paired_2 files hold the same read IDs but in different ORDER (a
# lexicographic-sort discrepancy), so they are re-paired by ID with `seqkit pair`
# before assembly; passing them to megahit positionally would assemble chimeras.
# unmatched_1/_2 go in as single-end reads. Each sample is assembled separately
# so `up` is preserved; contigs < 1000bp are discarded.
set -uo pipefail
ENV=/home/glh52/miniconda3/envs/assembly/bin
SRC=/scratch/home/glh52/glm-epinet-pyt/data_gen/darkmatter
OUT=/scratch/home/glh52/darkmatter_assembly
THREADS=${THREADS:-32}; PAR=${PAR:-5}
mkdir -p "$OUT"
run_one () {
  up=$1; s=$2; p="$SRC/profile.n.300.size.7.5.k.100.sigma.1.0.up.${up}.mut_rate.0.0_sample_${s}"
  d="$OUT/up${up}_s${s}"
  [ -s "$d/asm/final.contigs.fa" ] && { echo "skip $d"; return; }
  mkdir -p "$d"; echo "[$(date +%T)] start up=$up s=$s"
  $ENV/seqkit pair -j 8 -1 "${p}_paired_1.fastq.gz" -2 "${p}_paired_2.fastq.gz" -O "$d/paired" > "$d/pair.log" 2>&1 || { echo "FAILED pair up=$up s=$s"; return; }
  r1=$(ls "$d"/paired/*paired_1*); r2=$(ls "$d"/paired/*paired_2*)
  rm -rf "$d/asm"
  $ENV/megahit -1 "$r1" -2 "$r2" -r "${p}_unmatched_1.fastq.gz,${p}_unmatched_2.fastq.gz" \
     --min-contig-len 1000 -t $THREADS -o "$d/asm" > "$d/megahit.log" 2>&1 \
     && echo "[$(date +%T)] DONE up=$up s=$s $(grep -c '>' $d/asm/final.contigs.fa) contigs" \
     || echo "FAILED megahit up=$up s=$s"
  rm -rf "$d/paired" "$d/asm/intermediate_contigs"
}
export -f run_one; export ENV SRC OUT THREADS
for up in 0.25 0.75 1.0; do for s in 0 1 2 3 4; do echo "$up $s"; done; done \
  | xargs -P $PAR -n 2 bash -c 'run_one "$0" "$1"'
echo "ALL ASSEMBLIES FINISHED"
