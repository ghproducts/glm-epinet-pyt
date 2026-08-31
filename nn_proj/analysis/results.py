"""Loading the prediction grid written by the inference runs.

Inference writes one CSV per (seed, backbone, method, train task, test set):

    <results_root>/
      inference_results_<seed>/
        <backbone>/
          <method>/
            <train_task>/
              <test_set>/
                inference_uncertainty.csv

``ResultsIndex`` walks that tree once, records what exists, and hands back
prediction frames on demand. It replaces the per-directory loader that each
analysis notebook used to construct by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import pandas as pd

from .metrics import correctness
from .repair import read_predictions
from .tasks import TaskPair, TaskRegistry, load_registry

SEED_DIR_RE = re.compile(r"^inference_results_(\d+)$")
DEFAULT_CSV_NAME = "inference_uncertainty.csv"


@dataclass(frozen=True)
class RunKey:
    """Identifies one prediction file."""

    seed: int
    backbone: str
    method: str
    train: str
    test: str

    def __str__(self) -> str:
        return f"seed{self.seed}/{self.backbone}/{self.method}/{self.train}->{self.test}"


class ResultsIndex:
    """An index over a tree of inference outputs.

    Parameters
    ----------
    root:
        Directory containing the ``inference_results_<seed>`` folders.
    registry:
        Task registry; defaults to ``configs/tasks.yaml``.
    csv_name:
        Prediction filename to look for. Override for ablation sweeps that
        write e.g. ``inference_uncertainty_epoch3.csv``.
    canonical_only:
        Restrict to the seeds, backbones, and methods listed under ``grid`` in
        the task config. On by default so that superseded or exploratory runs
        sitting in the same tree cannot leak into a reported number. Pass
        False to index everything present.
    """

    def __init__(
        self,
        root: Path | str,
        registry: Optional[TaskRegistry] = None,
        csv_name: str = DEFAULT_CSV_NAME,
        canonical_only: bool = True,
    ):
        self.root = Path(root)
        self.registry = registry or load_registry()
        self.csv_name = csv_name
        self.canonical_only = canonical_only
        self._paths: Dict[RunKey, Path] = {}
        self._cache: Dict[RunKey, pd.DataFrame] = {}
        # key -> rows discarded while loading, for malformed files
        self.repaired: Dict[RunKey, int] = {}
        self._scan()

    def _allowed(self, field: str, value) -> bool:
        """Whether a value passes the canonical-grid allowlist."""
        if not self.canonical_only:
            return True
        allowed = self.registry.grid.get(field)
        return allowed is None or value in allowed

    def _scan(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Results root does not exist: {self.root}")

        for seed_dir in sorted(self.root.iterdir()):
            m = SEED_DIR_RE.match(seed_dir.name)
            if not (seed_dir.is_dir() and m):
                continue
            seed = int(m.group(1))
            if not self._allowed("seeds", seed):
                continue
            for backbone_dir in sorted(seed_dir.iterdir()):
                if not backbone_dir.is_dir() or backbone_dir.name.startswith("."):
                    continue
                if not self._allowed("backbones", backbone_dir.name):
                    continue
                for method_dir in sorted(backbone_dir.iterdir()):
                    if not method_dir.is_dir() or method_dir.name.startswith("."):
                        continue
                    if not self._allowed("methods", method_dir.name):
                        continue
                    for train_dir in sorted(method_dir.iterdir()):
                        if not train_dir.is_dir() or train_dir.name.startswith("."):
                            continue
                        for test_dir in sorted(train_dir.iterdir()):
                            csv = test_dir / self.csv_name
                            if not csv.is_file():
                                continue
                            key = RunKey(seed, backbone_dir.name, method_dir.name,
                                         train_dir.name, test_dir.name)
                            self._paths[key] = csv

        if not self._paths:
            raise FileNotFoundError(
                f"No {self.csv_name} files found under {self.root}. Expected "
                f"{self.root}/inference_results_<seed>/<backbone>/<method>/<train>/<test>/"
            )

    # -- inventory ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[RunKey]:
        return iter(sorted(self._paths, key=lambda k: (k.backbone, k.method, k.train, k.test, k.seed)))

    def _values(self, attr: str) -> List:
        return sorted({getattr(k, attr) for k in self._paths})

    @property
    def seeds(self) -> List[int]:
        return self._values("seed")

    @property
    def backbones(self) -> List[str]:
        return self._values("backbone")

    @property
    def methods(self) -> List[str]:
        return self._values("method")

    def select(self, **constraints) -> List[RunKey]:
        """Keys matching every given field. Values may be scalars or sequences."""
        def matches(key: RunKey) -> bool:
            for field, wanted in constraints.items():
                if wanted is None:
                    continue
                actual = getattr(key, field)
                if isinstance(wanted, (list, tuple, set)):
                    if actual not in wanted:
                        return False
                elif actual != wanted:
                    return False
            return True
        return [k for k in self if matches(k)]

    def has(self, **constraints) -> bool:
        return bool(self.select(**constraints))

    # -- access ------------------------------------------------------------

    def path(self, key: RunKey) -> Path:
        return self._paths[key]

    def __contains__(self, key: RunKey) -> bool:
        return key in self._paths

    def frame(self, key: RunKey) -> pd.DataFrame:
        """Prediction frame for one run, with a ``correct`` column added.

        Frames are cached; the grid is read many times over when building
        cross-method tables.
        """
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        df, dropped = read_predictions(self._paths[key], warn=False)
        if dropped:
            self.repaired[key] = dropped
        absent = {"labels", "pred"} - set(df.columns)
        if absent:
            raise ValueError(f"{self._paths[key]}: missing column(s) {sorted(absent)}")
        df["correct"] = correctness(df)
        self._cache[key] = df
        return df

    def get(self, seed: int, backbone: str, method: str, train: str, test: str) -> pd.DataFrame:
        return self.frame(RunKey(seed, backbone, method, train, test))

    # -- coverage checking -------------------------------------------------

    def coverage(self) -> pd.DataFrame:
        """Which registered pairs are present, per backbone/method/seed.

        Columns: backbone, method, train, test, n_seeds, seeds, registered.
        ``registered`` flags whether the pair appears in ``configs/tasks.yaml``
        — an unregistered pair on disk means the registry is out of date.
        """
        registered = {(p.train, p.test) for p in self.registry.pairs}
        rows = {}
        for k in self:
            entry = rows.setdefault((k.backbone, k.method, k.train, k.test), [])
            entry.append(k.seed)
        return pd.DataFrame([{
            "backbone": bb, "method": me, "train": tr, "test": te,
            "n_seeds": len(seeds), "seeds": ",".join(map(str, sorted(seeds))),
            "registered": (tr, te) in registered,
        } for (bb, me, tr, te), seeds in sorted(rows.items())])

    def row_count_check(self, tolerance: int = 0) -> pd.DataFrame:
        """Test sets whose row count disagrees across seeds or methods.

        Every run over a given (train, test) pair scores the same test file, so
        the row counts must agree exactly. A disagreement means some run lost
        examples — a truncated write, a crashed job, or rows destroyed by the
        interleaved-repr corruption that ``nn_proj.analysis.repair`` handles.

        Returns the offending runs with their count and the modal count for
        that pair, so a partially-recovered file cannot quietly contribute a
        biased mean to a reported cell.
        """
        counts = {}
        for k in self:
            counts[k] = len(self.frame(k))

        by_pair: Dict[tuple, List[RunKey]] = {}
        for k in counts:
            by_pair.setdefault((k.train, k.test), []).append(k)

        rows = []
        for pair, keys in sorted(by_pair.items()):
            values = [counts[k] for k in keys]
            expected = max(set(values), key=values.count)  # modal count
            for k in keys:
                if abs(counts[k] - expected) > tolerance:
                    rows.append({
                        "seed": k.seed, "backbone": k.backbone, "method": k.method,
                        "train": k.train, "test": k.test,
                        "n_rows": counts[k], "expected": expected,
                        "shortfall": expected - counts[k],
                        "pct_lost": 100.0 * (expected - counts[k]) / max(expected, 1),
                    })
        return pd.DataFrame(rows)

    def missing(self) -> pd.DataFrame:
        """Registered (backbone, method, pair, seed) combinations with no file.

        Run this before generating figures: a silently missing cell otherwise
        shows up as a gap in a heatmap with no indication of why.
        """
        rows = []
        for bb in self.backbones:
            for me in self.methods:
                for pair in self.registry.pairs:
                    for seed in self.seeds:
                        if not self.has(seed=seed, backbone=bb, method=me,
                                        train=pair.train, test=pair.test):
                            rows.append({"seed": seed, "backbone": bb, "method": me,
                                         "train": pair.train, "test": pair.test})
        return pd.DataFrame(rows)

    def summary(self) -> str:
        cov = self.coverage()
        unregistered = cov[~cov.registered]
        lines = [
            f"{len(self)} prediction files under {self.root}",
            f"  seeds     : {self.seeds}",
            f"  backbones : {self.backbones}",
            f"  methods   : {self.methods}",
            f"  pairs     : {len(cov)} distinct (backbone, method, train, test) combinations",
        ]
        if len(unregistered):
            lines.append(f"  WARNING: {len(unregistered)} pair(s) on disk are not in configs/tasks.yaml:")
            for _, r in unregistered.drop_duplicates(["train", "test"]).iterrows():
                lines.append(f"    {r.train} -> {r.test}")
        miss = self.missing()
        if len(miss):
            lines.append(f"  {len(miss)} registered (seed, backbone, method, pair) cells have no file")
        if self.repaired:
            lines.append(f"  {len(self.repaired)} file(s) needed row repair on load; "
                         f"run `python -m nn_proj.analysis.repair {self.root} --write` to fix on disk")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    idx = ResultsIndex(root)
    print(idx.summary())
