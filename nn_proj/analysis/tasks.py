"""Loader for the train/test pair registry in ``configs/tasks.yaml``.

The registry is the single source of truth for which evaluations exist, what
each predicts, and how its shift category was assigned. Nothing downstream
should hard-code a task list.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "tasks.yaml"

CATEGORIES = ("ID", "Near-ID", "Near-OOD", "OOD")


@dataclass(frozen=True)
class TaskPair:
    """One train/test evaluation."""

    train: str          # training task directory name, e.g. "pbsim_family"
    test: str           # test set directory name, e.g. "ood_novel_family_family"
    train_task: str     # registry key, e.g. "pbsim" (rank stripped)
    rank: Optional[str] # taxonomic rank, or None
    category: str       # ID | Near-ID | Near-OOD | OOD
    target: str
    target_shift: bool
    held_out: str
    supports: Sequence[str]
    note: str = ""

    @property
    def supports_calibration(self) -> bool:
        """Whether error rate and ECE are interpretable for this pair."""
        return "calibration" in self.supports

    @property
    def supports_novelty(self) -> bool:
        """Whether the pair is valid as ID/OOD inputs for AUROC."""
        return "novelty" in self.supports

    @property
    def is_id(self) -> bool:
        return self.category == "ID"

    def __str__(self) -> str:
        return f"{self.train}->{self.test}"


@dataclass
class TaskRegistry:
    pairs: List[TaskPair]
    ood_anchors: Dict[str, str]
    methods: Dict[str, dict]
    scores: List[str]
    grid: Dict[str, list]
    excluded_runs: List[dict] = field(default_factory=list)

    # -- lookup ------------------------------------------------------------

    def for_train(self, train: str) -> List[TaskPair]:
        """Every pair sharing a training task directory."""
        return [p for p in self.pairs if p.train == train]

    def get(self, train: str, test: str) -> TaskPair:
        for p in self.pairs:
            if p.train == train and p.test == test:
                return p
        raise KeyError(f"No registered pair {train}->{test}")

    @property
    def train_tasks(self) -> List[str]:
        seen = []
        for p in self.pairs:
            if p.train not in seen:
                seen.append(p.train)
        return seen

    def anchor_for(self, pair: TaskPair) -> Optional[str]:
        """Test-set name of the ID anchor used when scoring ``pair`` as OOD."""
        anchor = self.ood_anchors.get(pair.train_task)
        if anchor is None:
            return None
        return f"{anchor}_{pair.rank}" if pair.rank and anchor.startswith(("id_", "ood_")) else anchor

    def ood_pairs(self) -> List[tuple]:
        """All (id_pair, ood_pair) combinations for OOD detection.

        The ID anchor is excluded from acting as its own OOD set.
        """
        out = []
        for train in self.train_tasks:
            pairs = self.for_train(train)
            anchor_name = self.ood_anchors.get(pairs[0].train_task)
            if anchor_name is None:
                continue
            anchors = [p for p in pairs if p.test == anchor_name or
                       (p.rank and p.test == f"{anchor_name}_{p.rank}")]
            if not anchors:
                continue
            anchor = anchors[0]
            for p in pairs:
                if p.test != anchor.test and p.supports_novelty:
                    out.append((anchor, p))
        return out

    # -- export ------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Registry as a table, for the supplement and for sanity checks."""
        return pd.DataFrame([{
            "train": p.train,
            "test": p.test,
            "rank": p.rank or "",
            "category": p.category,
            "target": p.target,
            "target_shift": p.target_shift,
            "held_out": p.held_out,
            "supports": "+".join(p.supports),
            "note": " ".join(p.note.split()),
        } for p in self.pairs])


def _expand(entry: dict) -> List[TaskPair]:
    """Expand one registry entry, fanning out over ranks when present."""
    train_task = entry["train"]
    ranks = entry.get("ranks", [None])
    default_target = entry.get("target", "")

    pairs = []
    for rank in ranks:
        train_dir = f"{train_task}_{rank}" if rank else train_task
        for t in entry["tests"]:
            test_dir = f"{t['test']}_{rank}" if rank else t["test"]
            pairs.append(TaskPair(
                train=train_dir,
                test=test_dir,
                train_task=train_task,
                rank=rank,
                category=t["category"],
                target=t.get("target", default_target),
                target_shift=bool(t.get("target_shift", False)),
                held_out=t.get("held_out", ""),
                supports=tuple(t.get("supports", [])),
                note=t.get("note", ""),
            ))
    return pairs


@functools.lru_cache(maxsize=None)
def load_registry(path: Path | str = CONFIG_PATH) -> TaskRegistry:
    """Load and validate ``configs/tasks.yaml``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    pairs: List[TaskPair] = []
    for entry in cfg["datasets"]:
        pairs.extend(_expand(entry))

    for p in pairs:
        if p.category not in CATEGORIES:
            raise ValueError(f"{p}: unknown category {p.category!r}, expected one of {CATEGORIES}")
        if not p.supports:
            raise ValueError(f"{p}: must support at least one of calibration/novelty")
        if p.target_shift and p.supports_calibration:
            raise ValueError(
                f"{p}: marked target_shift but claims to support calibration. A pair that "
                "changes the prediction target cannot have interpretable error or ECE."
            )

    return TaskRegistry(
        pairs=pairs,
        ood_anchors=cfg["ood_anchors"],
        methods=cfg["methods"],
        scores=list(cfg["scores"]),
        grid=cfg.get("grid", {}),
        excluded_runs=cfg.get("excluded_runs", []),
    )


if __name__ == "__main__":
    reg = load_registry()
    df = reg.to_frame()
    pd.set_option("display.max_colwidth", 40, "display.width", 200)
    print(df.drop(columns=["note"]).to_string(index=False))
    print(f"\n{len(reg.pairs)} registered pairs across {len(reg.train_tasks)} training tasks")
    print(f"{sum(p.supports_calibration for p in reg.pairs)} support calibration, "
          f"{sum(p.supports_novelty for p in reg.pairs)} support novelty detection")
