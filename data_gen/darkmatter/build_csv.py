#!/usr/bin/env python3
"""Convert the darkmatter FASTQ reads into an inference-ready CSV.

This dataset (``data_gen/darkmatter/*.fastq.gz``) is a raw drop with no
generation script, data card, or documentation anywhere in this repo -- see
the exploratory notes in the plan this script was built from. Filenames look
like::

    profile.n.300.size.7.5.k.100.sigma.1.0.up.0.25.mut_rate.0.0_sample_0_unmatched_2.fastq.gz

``up`` is inferred (not confirmed -- nothing in the repo defines it) to mean
"fraction of the simulated community drawn from taxa outside the reference/
label space", i.e. percent unknown/"dark matter". ``sample`` is a replicate
index. FASTQ headers themselves (``@S0R<n>/2``) carry no taxonomy at all, so
there is no per-read ground truth to attach -- this script only extracts
sequences and the provenance implied by the filename.

Output columns match what ``nn_proj.common.datasets.load_local_dataset``
requires (``sequence`` + ``label``), plus metadata that
``nn_proj.models.<BACKBONE>.inference`` auto-detects and carries through to
the output ``inference_uncertainty.csv`` (any column beyond
``sequence``/``label``/``labels``):

    label        constant placeholder (0) -- there is no real label; never
                 used for scoring, just satisfies the required-column check
    sequence     the read, uppercased with non-ACGT bases replaced by N
    up           parsed from the filename (float)
    sample       parsed from the filename (int)
    read_type    "paired" or "unmatched", parsed from the filename (str)
    mate         mate index 1 or 2, parsed from the filename (int)
    source_file  originating filename, for provenance

On 2026-09-14 the remaining 46 files of this drop were added alongside the 15
``unmatched_2`` files that were here originally (their mtimes were preserved
from generation, so only ctime shows them as new). The filename regex used to
accept only ``_unmatched_<mate>``, silently skipping every ``paired_*`` file;
it now accepts both and records which is which in ``read_type``/``mate`` so
downstream analysis can stratify or pool. Note the only ``up=0.5`` file in the
drop is ``sample_4_paired_1`` -- a single file, no replicates and no unmatched
counterpart, so that point is not comparable to the other ``up`` values.

Deliberately do NOT emit a ``taxid`` column and do NOT pass --taxa_rank/
--taxa_df at inference time: load_local_dataset's taxid->rank remap drops
every row whose taxid isn't in the lineage table, which for reads outside the
trained label space is every row -- the opposite of what this analysis needs.

Reservoir sampling (reused from data_gen/pbsim/reads_to_csv.py's pattern) so
memory stays flat regardless of how many reads are in each ~50MB gzipped file
(~500k reads each, verified directly).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import re
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Optional

FILENAME_RE = re.compile(
    r"profile\.n\.(?P<n>[\d.]+)\.size\.(?P<size>[\d.]+)\.k\.(?P<k>[\d.]+)"
    r"\.sigma\.(?P<sigma>[\d.]+)\.up\.(?P<up>[\d.]+)\.mut_rate\.(?P<mut_rate>[\d.]+)"
    r"_sample_(?P<sample>\d+)_(?P<read_type>paired|unmatched)_(?P<mate>\d+)\.fastq\.gz$"
)

NON_ACGT = re.compile(r"[^ATCG]")


def iter_fastq(path: Path) -> Iterator[str]:
    """Yield sequences from a gzipped FASTQ file. Same shape as pbsim's reads_to_csv.py."""
    with gzip.open(path, "rt") as f:
        while True:
            header = f.readline()
            if not header.strip():
                return
            seq = f.readline().rstrip()
            f.readline()          # '+' separator
            if not f.readline():  # quality line; truncated record if absent
                return
            yield seq


def clean(seq: str) -> str:
    """Uppercase and replace every non-ACGT character with N."""
    return NON_ACGT.sub("N", seq.upper())


def sample_reads(path: Path, target: Optional[int], rng: random.Random) -> List[str]:
    """Up to `target` reads from one file via reservoir sampling, or all of them if None.

    Pure-Python fallback, kept for environments without zcat/awk. Correct but
    slow: it pushes every line of every file through the interpreter, which is
    fine for the ~50MB unmatched_2 files but not for the ~2GB paired_1 ones
    (see sample_reads_fast).
    """
    if target is None:
        return [clean(s) for s in iter_fastq(path)]

    reservoir: List[str] = []
    seen = 0
    for raw in iter_fastq(path):
        seen += 1
        if len(reservoir) < target:
            reservoir.append(clean(raw))
        else:
            j = rng.randrange(seen)
            if j < target:
                reservoir[j] = clean(raw)
    return reservoir


# Algorithm R (the same reservoir sampling sample_reads does) but executed by
# awk instead of Python, fed by zcat. The full drop is ~70GB gzipped / ~245GB
# decompressed and reservoir sampling has to read all of it to stay uniform;
# in Python that is hours, here it runs at ~170MB/s of decompressed text.
# Uppercasing and the non-ACGT -> N substitution happen here too, matching
# clean().
_AWK_RESERVOIR = r"""
BEGIN { srand(seed) }
NR % 4 == 2 {
    n++
    s = toupper($0)
    gsub(/[^ATCG]/, "N", s)
    if (m < k) { r[m++] = s }
    else { j = int(rand() * n); if (j < k) r[j] = s }
}
END { for (i = 0; i < m; i++) print r[i] }
"""


def sample_reads_fast(path: Path, target: Optional[int], seed: int) -> List[str]:
    """sample_reads via a `zcat | awk` pipeline. `seed` is per-file, so each
    file's sample is reproducible independently of how many other files are in
    the run (unlike the shared random.Random stream sample_reads uses, where
    adding a file to the directory reshuffles every later file's sample)."""
    if target is None:
        cmd = ["awk", "NR % 4 == 2 { s = toupper($0); gsub(/[^ATCG]/, \"N\", s); print s }"]
    else:
        cmd = ["awk", "-v", f"k={target}", "-v", f"seed={seed}", _AWK_RESERVOIR]

    with subprocess.Popen(["zcat", str(path)], stdout=subprocess.PIPE) as zcat:
        out = subprocess.run(cmd, stdin=zcat.stdout, stdout=subprocess.PIPE,
                             check=True, text=True).stdout
        zcat.stdout.close()
        if zcat.wait() != 0:
            raise RuntimeError(f"zcat failed on {path}")
    return out.splitlines()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--raw-dir", type=Path, default=here,
                     help="Directory containing the darkmatter *.fastq.gz files")
    ap.add_argument("--outdir", type=Path, default=here / "csv_data")
    ap.add_argument("--out-name", default="darkmatter_reads.csv")
    ap.add_argument("--reads-per-file", type=int, default=3000,
                     help="Reservoir-sample this many reads per file (default 3000). "
                          "Use a small value (e.g. 50) for a quick sanity check first.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=8,
                     help="Files sampled concurrently (default 8). Each worker is a "
                          "zcat|awk pipeline, so this is bounded by disk read throughput.")
    ap.add_argument("--no-fast", action="store_true",
                     help="Use the pure-Python reservoir sampler instead of zcat|awk. "
                          "Correct but far slower on the multi-GB paired_* files.")
    args = ap.parse_args()

    files = sorted(args.raw_dir.glob("*.fastq.gz"))
    if not files:
        raise SystemExit(f"No .fastq.gz files found under {args.raw_dir}")

    parsed = []
    n_unparsed = 0
    for path in files:
        m = FILENAME_RE.match(path.name)
        if not m:
            print(f"[warn] filename doesn't match expected pattern, skipping: {path.name}")
            n_unparsed += 1
            continue
        parsed.append((path, m))

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / args.out_name

    # Sample every file up front so the slow part runs concurrently, then write
    # in sorted-filename order regardless of which worker finished first -- the
    # CSV stays byte-identical across runs for a given seed.
    if args.no_fast:
        rng = random.Random(args.seed)
        samples = [sample_reads(path, args.reads_per_file, rng) for path, _ in parsed]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            samples = list(pool.map(
                lambda iv: sample_reads_fast(iv[1][0], args.reads_per_file, args.seed + iv[0]),
                enumerate(parsed),
            ))

    n_written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "sequence", "up", "sample", "read_type", "mate",
                          "n", "size", "k", "sigma", "mut_rate", "source_file"])

        for (path, m), seqs in zip(parsed, samples):
            for seq in seqs:
                writer.writerow([
                    0, seq,
                    float(m.group("up")), int(m.group("sample")),
                    m.group("read_type"), int(m.group("mate")),
                    float(m.group("n")), float(m.group("size")), float(m.group("k")),
                    float(m.group("sigma")), float(m.group("mut_rate")),
                    path.name,
                ])
            n_written += len(seqs)
            print(f"{path.name}: sampled {len(seqs)} reads "
                  f"(up={m.group('up')}, sample={m.group('sample')}, "
                  f"{m.group('read_type')}_{m.group('mate')})")

    if n_unparsed:
        print(f"\n[warn] {n_unparsed} file(s) skipped, filename pattern mismatch")
    print(f"\nWrote {n_written} rows from {len(parsed)} file(s) -> {out_path}")


if __name__ == "__main__":
    main()
