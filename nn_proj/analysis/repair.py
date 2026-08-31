"""Detection and repair of malformed prediction CSVs.

Some early inference runs wrote a multi-line array repr into the output
stream, interleaving lines like ``        0, 0, 0])"`` between genuine
prediction rows. Pandas reads the affected columns as object dtype, and every
downstream metric then either raises or silently computes on strings.

The genuine rows survive intact, so the damage is recoverable: a prediction
row has the full column count, an integer label, and floating-point
confidence and uncertainty values. Everything else is discarded.

Usage
-----
    # report only
    python -m nn_proj.analysis.repair <results_root>

    # rewrite affected files, keeping a .corrupt backup of each
    python -m nn_proj.analysis.repair <results_root> --write
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Columns that must parse as floats in a genuine prediction row.
FLOAT_COLS = ("max_confidence", "U_total", "U_aleatoric", "U_epistemic")
INT_COLS = ("labels",)


def is_clean(df: pd.DataFrame) -> bool:
    """Whether every numeric column of a loaded frame parsed as numeric."""
    for col in FLOAT_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            return False
    return True


def read_predictions(path: Path | str, warn: bool = True) -> Tuple[pd.DataFrame, int]:
    """Read a prediction CSV, filtering interleaved junk rows if present.

    Returns the frame and the number of rows discarded. A clean file costs one
    ordinary ``read_csv`` and discards nothing.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if is_clean(df):
        return df, 0

    n_before = len(df)
    float_cols = [c for c in FLOAT_COLS if c in df.columns]
    int_cols = [c for c in INT_COLS if c in df.columns]

    # A genuine row is identified by its confidence: the top-class softmax
    # probability is bounded below by 1/C, so it is always strictly positive
    # and at most 1. The interleaved filler rows carry 0 in every field, and
    # the wrapped repr fragments do not parse at all. Both fail this test.
    #
    # The other uncertainty columns are deliberately *not* range-checked.
    # U_epistemic is a difference of two entropy estimates and lands a few
    # units in the last place either side of zero for deterministic models,
    # so a >= 0 bound there would discard valid rows.
    keep = pd.Series(True, index=df.index)
    if "max_confidence" in df.columns:
        conf = pd.to_numeric(df["max_confidence"], errors="coerce")
        keep &= conf.notna() & (conf > 0.0) & (conf <= 1.0)

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in int_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        keep &= vals.notna()
        df[col] = vals

    cleaned = df[keep].reset_index(drop=True)
    for col in int_cols:
        cleaned[col] = cleaned[col].astype("int64")

    dropped = n_before - len(cleaned)
    if warn and dropped:
        print(f"[repair] {path}: dropped {dropped} malformed row(s), kept {len(cleaned)}")
    return cleaned, dropped


def scan(root: Path | str, csv_name: str = "inference_uncertainty.csv") -> pd.DataFrame:
    """Find every malformed prediction CSV under ``root``."""
    rows = []
    for path in sorted(Path(root).rglob(csv_name)):
        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            rows.append({"path": str(path), "status": "unreadable",
                         "rows_raw": 0, "rows_kept": 0, "detail": str(exc)[:80]})
            continue
        if is_clean(raw):
            continue
        cleaned, dropped = read_predictions(path, warn=False)
        rows.append({"path": str(path), "status": "malformed",
                     "rows_raw": len(raw), "rows_kept": len(cleaned),
                     "detail": f"dropped {dropped}"})
    return pd.DataFrame(rows)


def repair_file(path: Path, backup_suffix: str = ".corrupt") -> Optional[int]:
    """Rewrite one file in place, preserving the original alongside it.

    Returns the number of rows dropped, or None if the file was already clean.
    """
    raw = pd.read_csv(path)
    if is_clean(raw):
        return None
    cleaned, dropped = read_predictions(path, warn=False)
    backup = path.with_suffix(path.suffix + backup_suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    cleaned.to_csv(path, index=False)
    return dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Directory to scan recursively")
    ap.add_argument("--csv-name", default="inference_uncertainty.csv")
    ap.add_argument("--write", action="store_true",
                    help="Rewrite affected files (originals kept as *.csv.corrupt)")
    args = ap.parse_args()

    report = scan(args.root, args.csv_name)
    if report.empty:
        print(f"No malformed prediction files under {args.root}")
        return

    print(report.to_string(index=False))
    print(f"\n{len(report)} malformed file(s)")

    if args.write:
        print()
        for path in report.path:
            dropped = repair_file(Path(path))
            print(f"repaired {path} (dropped {dropped} rows)")
    else:
        print("\nRe-run with --write to repair them in place.")


if __name__ == "__main__":
    main()
