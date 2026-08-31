#!/usr/bin/env python3
"""Extract the 6 selected JASPAR motifs from the full CORE vertebrates file.

Step 2 of the promoter-motif pipeline (see README.md), run after the JASPAR
download and the AME enrichment scan documented there. The six IDs below were
chosen by hand from the top of AME's enrichment ranking, for enrichment *and*
diversity — many of the highest-ranked motifs are CpG-content artifacts
rather than a specific regulatory signal (see README.md for why).

Extracting the motif blocks from the ~1000-motif JASPAR file is a pure text
operation and is deterministic given the input file, which is why it's a
committed script rather than a documented shell step like the download.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# JASPAR motif ID -> short name used everywhere downstream (annotate_motifs.py,
# make_splits.py, and the motif indicator columns in the final CSVs).
MOTIF_NAME_MAP = {
    "MA0516.3": "SP2",
    "MA2328.1": "ZBED4",
    "MA0162.5": "EGR1",
    "MA0759.3": "ELK3",
    "MA1122.2": "TFDP1",
    "MA2546.1": "ZNF131",
}


def extract(meme_text: list[str], selected_ids: set[str]) -> tuple[list[str], list[list[str]]]:
    """Split a MEME motif file into a header and a list of MOTIF blocks."""
    header_lines: list[str] = []
    motif_blocks: list[list[str]] = []
    current_block: list[str] = []
    in_motif = False

    for line in meme_text:
        if line.startswith("MOTIF "):
            if current_block:
                motif_blocks.append(current_block)
            current_block, in_motif = [line], True
        elif in_motif:
            current_block.append(line)
        else:
            header_lines.append(line)
    if current_block:
        motif_blocks.append(current_block)

    selected = [b for b in motif_blocks if b[0].split()[1] in selected_ids]
    return header_lines, selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--jaspar-meme", type=Path,
                     default=here / "motifs" / "jaspar_core_vertebrates.meme",
                     help="Full JASPAR CORE vertebrates MEME-format file")
    ap.add_argument("--out", type=Path,
                     default=here / "motifs" / "selected_promoter_motifs.meme")
    args = ap.parse_args()

    text = args.jaspar_meme.read_text().splitlines()
    header, blocks = extract(text, set(MOTIF_NAME_MAP))

    found = {b[0].split()[1] for b in blocks}
    missing = set(MOTIF_NAME_MAP) - found
    if missing:
        raise SystemExit(f"Motif ID(s) not found in {args.jaspar_meme}: {sorted(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(header).rstrip() + "\n\n")
        for block in blocks:
            f.write("\n".join(block).rstrip() + "\n\n")

    print(f"Wrote {len(blocks)} motifs to {args.out}:")
    for motif_id, name in MOTIF_NAME_MAP.items():
        print(f"  {motif_id}  {name}")


if __name__ == "__main__":
    main()
