#!/usr/bin/env python3
"""Emit or run every command in the experimental grid.

The four shell templates in this directory each encode one cell of the grid and
have to be edited between runs. This script expands `configs/experiments.yaml`
into the whole set, so the experiment is specified once rather than
reconstructed by hand each time.

Stages run in dependency order:

    train      fine-tune each backbone on each training task
    epinet     train an epinet head on each frozen base checkpoint
    temperature fit a temperature on each task's validation split
    inference  run every (method, test set) combination

Examples
--------
    # See what would run, without running it
    python scripts/run_grid.py --dry-run

    # One backbone, one seed, just the inference stage
    python scripts/run_grid.py --stage inference --backbone DNABERT2 --seed 1

    # Write a job list for a scheduler instead of running locally
    python scripts/run_grid.py --dry-run --format plain > jobs.txt

    # Actually run, skipping cells whose output already exists
    python scripts/run_grid.py --stage all --skip-existing --run

Nothing runs without `--run`; the default is to print.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["train", "epinet", "temperature", "inference"]

# Inference method -> (uncertainty_method flag, whether it needs the epinet
# checkpoint, whether it needs a fitted temperature).
METHODS = {
    "base": ("base", False, False),
    "base_scaled": ("base", False, True),
    "mc_dropout": ("mc_dropout", False, False),
    "conv_epinet": ("epinet", True, False),
}


@dataclass
class Job:
    stage: str
    name: str
    argv: List[str]
    outdir: Optional[Path] = None
    output_marker: Optional[Path] = None

    def exists(self) -> bool:
        return self.output_marker is not None and self.output_marker.exists()


class Grid:
    def __init__(self, config_path: Path):
        self.cfg = yaml.safe_load(config_path.read_text())
        self.paths = self.cfg["paths"]

    # -- helpers -----------------------------------------------------------

    def max_length(self, backbone: str, task: dict) -> int:
        """model_max_length for a backbone/task, from the tokenizer's compression."""
        per_base = self.cfg["backbones"][backbone]["tokens_per_base"]
        return max(1, math.ceil(task["seq_length"] * per_base))

    def task_variants(self, name: str, task: dict) -> Iterator[tuple]:
        """Yield (task_dir, rank) for a training task, fanning out over ranks."""
        for rank in task.get("ranks", [None]):
            yield (f"{name}_{rank}" if rank else name), rank

    def _fmt(self, key: str, **kw) -> Path:
        return REPO_ROOT / self.paths[key].format(**kw)

    def _data_args(self, task: dict, rank: Optional[str], data_path: str) -> List[str]:
        args = ["--data_path", data_path]
        if rank:
            args += ["--taxa_rank", rank, "--taxa_df", task["taxa_df"]]
        if task.get("num_labels"):
            args += ["--num_labels", str(task["num_labels"])]
        return args

    # -- stages ------------------------------------------------------------

    def jobs(self, stages: List[str], backbones: List[str], seeds: List[int],
             tasks: Optional[List[str]] = None) -> Iterator[Job]:
        train_cfg = self.cfg["training"]
        epi_cfg = self.cfg["epinet"]
        inf_cfg = self.cfg["inference"]

        for seed in seeds:
            for backbone in backbones:
                spec = self.cfg["backbones"][backbone]
                model = spec["model_name_or_path"]

                for task_name, task in self.cfg["training_tasks"].items():
                    if tasks and task_name not in tasks:
                        continue
                    for task_dir, rank in self.task_variants(task_name, task):
                        max_len = self.max_length(backbone, task)
                        base_ckpt = self._fmt("base_checkpoint", seed=seed,
                                              backbone=backbone, task=task_dir)
                        epi_ckpt = self._fmt("epinet_checkpoint", seed=seed,
                                             backbone=backbone, task=task_dir)
                        common = [
                            "--model_name_or_path", model,
                            "--model_max_length", str(max_len),
                            "--seed", str(seed), "--data_seed", str(seed),
                        ]
                        data = self._data_args(task, rank, task["data_path"])

                        if "train" in stages:
                            yield Job(
                                stage="train",
                                name=f"seed{seed}/{backbone}/{task_dir}/base",
                                outdir=base_ckpt,
                                output_marker=base_ckpt,
                                argv=[sys.executable, "-m", f"nn_proj.models.{backbone}.train_base",
                                      *data, *common,
                                      "--output_dir", str(base_ckpt),
                                      "--run_name", f"{backbone}_{task_dir}_s{seed}",
                                      "--learning_rate", str(train_cfg["learning_rate"]),
                                      "--num_train_epochs", str(train_cfg["num_train_epochs"]),
                                      "--per_device_train_batch_size",
                                      str(train_cfg["per_device_train_batch_size"]),
                                      "--per_device_eval_batch_size",
                                      str(train_cfg["per_device_eval_batch_size"]),
                                      "--warmup_steps", str(train_cfg["warmup_steps"]),
                                      "--weight_decay", str(train_cfg["weight_decay"]),
                                      *(["--fp16"] if train_cfg["fp16"] else []),
                                      "--overwrite_output_dir", "True"],
                            )

                        if "epinet" in stages:
                            yield Job(
                                stage="epinet",
                                name=f"seed{seed}/{backbone}/{task_dir}/epinet",
                                outdir=epi_ckpt,
                                output_marker=epi_ckpt,
                                argv=[sys.executable, "-m", f"nn_proj.models.{backbone}.train_epinet",
                                      *data, *common,
                                      "--checkpoint", str(base_ckpt),
                                      "--output_dir", str(epi_ckpt),
                                      "--run_name", f"{backbone}_{task_dir}_epinet_s{seed}",
                                      "--learning_rate", str(epi_cfg["learning_rate"]),
                                      "--num_train_epochs", str(epi_cfg["num_train_epochs"]),
                                      "--per_device_train_batch_size",
                                      str(train_cfg["per_device_train_batch_size"]),
                                      "--per_device_eval_batch_size",
                                      str(train_cfg["per_device_eval_batch_size"]),
                                      *(["--fp16"] if train_cfg["fp16"] else []),
                                      "--overwrite_output_dir", "True"],
                            )

                        if "temperature" in stages:
                            yield Job(
                                stage="temperature",
                                name=f"seed{seed}/{backbone}/{task_dir}/temperature",
                                argv=[sys.executable, "-m", f"nn_proj.models.{backbone}.scaling",
                                      *data, *common,
                                      "--checkpoint", str(base_ckpt),
                                      "--run_name", f"{backbone}_{task_dir}_temp_s{seed}",
                                      "--per_device_eval_batch_size",
                                      str(inf_cfg["per_device_eval_batch_size"])],
                            )

                        if "inference" in stages:
                            temps = self.load_temperatures()
                            for method, (flag, needs_epi, needs_temp) in METHODS.items():
                                for test_name, test_path in task["tests"].items():
                                    test_dir = f"{test_name}_{rank}" if rank else test_name
                                    out = self._fmt("predictions", seed=seed, backbone=backbone,
                                                    method=method, task=task_dir, test=test_dir)
                                    ckpt = epi_ckpt if needs_epi else base_ckpt
                                    temp = 1.0
                                    if needs_temp:
                                        temp = temps.get(f"{seed}/{backbone}/{task_dir}", 1.0)
                                    yield Job(
                                        stage="inference",
                                        name=f"seed{seed}/{backbone}/{method}/{task_dir}/{test_dir}",
                                        outdir=out,
                                        output_marker=out / "inference_uncertainty.csv",
                                        argv=[sys.executable, "-m",
                                              f"nn_proj.models.{backbone}.inference",
                                              *self._data_args(task, rank, test_path), *common,
                                              "--checkpoint", str(ckpt),
                                              "--output_dir", str(out),
                                              "--run_name", f"{backbone}_{method}_{test_dir}_s{seed}",
                                              "--uncertainty_method", flag,
                                              "--temperature", str(temp),
                                              "--num_samples", str(inf_cfg["k_samples"]),
                                              "--per_device_eval_batch_size",
                                              str(inf_cfg["per_device_eval_batch_size"])],
                                    )

    # -- temperatures ------------------------------------------------------

    def temperature_path(self) -> Path:
        return REPO_ROOT / self.paths["temperatures"]

    def load_temperatures(self) -> Dict[str, float]:
        """Fitted temperatures, keyed 'seed/backbone/task'.

        The temperature stage prints ``T = <value>``; `record_temperature`
        below persists it. Without this file the base_scaled runs silently fall
        back to T = 1.0, which is identical to the unscaled base model — the
        failure mode that makes a temperature-scaling row look like a no-op.
        """
        path = self.temperature_path()
        return json.loads(path.read_text()) if path.exists() else {}

    def record_temperature(self, seed: int, backbone: str, task: str, value: float) -> None:
        path = self.temperature_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temps = self.load_temperatures()
        temps[f"{seed}/{backbone}/{task}"] = value
        path.write_text(json.dumps(temps, indent=2, sort_keys=True))


def parse_temperature(stdout: str) -> Optional[float]:
    """Pull the fitted T out of the scaling script's output."""
    for line in reversed(stdout.splitlines()):
        if line.strip().startswith("T ="):
            try:
                return float(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "experiments.yaml")
    ap.add_argument("--stage", nargs="*", default=["all"],
                    help=f"One or more of {STAGES}, or 'all'")
    ap.add_argument("--backbone", nargs="*", default=None)
    ap.add_argument("--task", nargs="*", default=None)
    ap.add_argument("--seed", nargs="*", type=int, default=None)
    ap.add_argument("--run", action="store_true", help="Execute (default is to print)")
    ap.add_argument("--dry-run", action="store_true", help="Print only; the default")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip cells whose output already exists")
    ap.add_argument("--format", choices=["pretty", "plain"], default="pretty",
                    help="'plain' prints bare commands, one per line, for a scheduler")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going when a job fails, and report failures at the end")
    args = ap.parse_args()

    grid = Grid(args.config)
    stages = STAGES if "all" in args.stage else args.stage
    unknown = set(stages) - set(STAGES)
    if unknown:
        ap.error(f"unknown stage(s) {sorted(unknown)}; choose from {STAGES}")

    backbones = args.backbone or list(grid.cfg["backbones"])
    seeds = args.seed or grid.cfg["seeds"]

    jobs = list(grid.jobs(stages, backbones, seeds, args.task))
    if args.skip_existing:
        before = len(jobs)
        jobs = [j for j in jobs if not j.exists()]
        print(f"# {before - len(jobs)} of {before} jobs already have output; "
              f"{len(jobs)} remaining\n", file=sys.stderr)

    if not (args.run and not args.dry_run):
        by_stage: Dict[str, int] = {}
        for job in jobs:
            by_stage[job.stage] = by_stage.get(job.stage, 0) + 1
            if args.format == "plain":
                print(" ".join(job.argv))
            else:
                print(f"[{job.stage}] {job.name}\n    {' '.join(job.argv)}\n")
        print("# " + ", ".join(f"{k}: {v}" for k, v in by_stage.items()) +
              f"  (total {len(jobs)})", file=sys.stderr)
        print("# nothing was executed; pass --run to execute", file=sys.stderr)
        return

    failures = []
    for i, job in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {job.stage}: {job.name}", flush=True)
        if job.outdir:
            job.outdir.mkdir(parents=True, exist_ok=True)
        capture = job.stage == "temperature"
        proc = subprocess.run(job.argv, cwd=REPO_ROOT, text=True,
                              capture_output=capture)
        if capture and proc.stdout:
            print(proc.stdout)
        if proc.returncode != 0:
            failures.append(job.name)
            print(f"!!! failed with exit code {proc.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                sys.exit(proc.returncode)
            continue

        if job.stage == "temperature":
            value = parse_temperature(proc.stdout or "")
            seed, backbone, task, _ = job.name.split("/")
            if value is None:
                print(f"!!! could not parse a temperature from the output of {job.name}; "
                      "base_scaled will fall back to T=1.0", file=sys.stderr)
                failures.append(job.name + " (no temperature parsed)")
            else:
                grid.record_temperature(int(seed.removeprefix("seed")), backbone, task, value)
                print(f"    recorded T = {value:.6f}")

    if failures:
        print(f"\n{len(failures)} job(s) failed:", file=sys.stderr)
        for name in failures:
            print(f"  {name}", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(jobs)} job(s) completed.")


if __name__ == "__main__":
    main()
