"""Smoke tests for the per-backbone entry points and the experiment grid.

These do not train anything. They check the contract the grid driver depends
on: every entry point imports, exposes its callable, and accepts the arguments
`scripts/run_grid.py` generates for it. That is enough to catch the failure
mode this suite exists for — a refactor that silently changes or drops a
command-line flag, which would otherwise only surface hours into a run.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BACKBONES = ["NT_transformer", "DNABERT2", "hyenaDNA", "CARMANIA"]
ENTRYPOINTS = {
    "train_base": "train",
    "train_epinet": "train",
    "scaling": "get_scalilng_params",
    "inference": "evaluate",
}


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("module,func", ENTRYPOINTS.items())
def test_entrypoint_imports_and_exposes_callable(backbone, module, func):
    mod = importlib.import_module(f"nn_proj.models.{backbone}.{module}")
    assert callable(getattr(mod, func)), f"{backbone}.{module}.{func} is not callable"


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("module", list(ENTRYPOINTS))
def test_argument_parser_accepts_grid_arguments(backbone, module):
    """The config dataclasses must accept every flag the grid driver emits."""
    import transformers

    config = importlib.import_module(f"nn_proj.models.{backbone}.config")
    parser = transformers.HfArgumentParser(
        (config.ModelArguments, config.DataArguments, config.TrainingArguments)
    )
    argv = [
        "--data_path", "dummy/path",
        "--model_name_or_path", "dummy/model",
        "--checkpoint", "dummy/ckpt",
        "--model_max_length", "75",
        "--seed", "1", "--data_seed", "1",
        "--output_dir", "dummy/out",
        "--run_name", "test",
        "--num_labels", "2",
        "--taxa_rank", "family",
        "--taxa_df", "dummy/lineage.csv",
        "--uncertainty_method", "epinet",
        "--temperature", "1.5",
        "--num_samples", "10",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "2",
        "--per_device_train_batch_size", "32",
        "--per_device_eval_batch_size", "16",
        "--warmup_steps", "50",
        "--weight_decay", "0.01",
    ]
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(argv)

    assert data_args.data_path == "dummy/path"
    assert data_args.taxa_rank == "family"
    assert model_args.uncertainty_method == "epinet"
    assert model_args.num_samples == 10
    assert model_args.temperature == pytest.approx(1.5)
    assert training_args.model_max_length == 75
    assert training_args.seed == 1


# ---------------------------------------------------------------------------
# Grid driver
# ---------------------------------------------------------------------------

def test_grid_expands_to_expected_shape():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_grid import Grid, METHODS

    grid = Grid(REPO_ROOT / "configs" / "experiments.yaml")
    seeds = grid.cfg["seeds"]
    backbones = list(grid.cfg["backbones"])

    task_dirs = [d for name, task in grid.cfg["training_tasks"].items()
                 for d, _ in grid.task_variants(name, task)]
    jobs = list(grid.jobs(["train"], backbones, seeds))
    assert len(jobs) == len(seeds) * len(backbones) * len(task_dirs)

    inference = list(grid.jobs(["inference"], backbones, seeds))
    expected = sum(
        len(seeds) * len(backbones) * len(METHODS) * len(task["tests"])
        * len(task.get("ranks", [None]))
        for task in grid.cfg["training_tasks"].values()
    )
    assert len(inference) == expected


def test_grid_max_length_matches_shell_templates():
    """The tokens-per-base rule must reproduce the hand-written constants."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_grid import Grid

    grid = Grid(REPO_ROOT / "configs" / "experiments.yaml")
    promoter = grid.cfg["training_tasks"]["promoter_all"]
    # scripts/test_model.sh hard-codes MAX_LENGTH=75 for DNABERT2 on promoter_all
    assert grid.max_length("DNABERT2", promoter) == 75
    # Single-character tokenizers use the raw sequence length
    assert grid.max_length("hyenaDNA", promoter) == promoter["seq_length"]
    assert grid.max_length("CARMANIA", promoter) == promoter["seq_length"]


def test_grid_inference_targets_epinet_checkpoint_for_epinet_method():
    """conv_epinet must load the epinet checkpoint, not the base one.

    Pointing it at the base checkpoint is a silent failure: the run completes
    and writes predictions that are simply the base model's.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_grid import Grid

    grid = Grid(REPO_ROOT / "configs" / "experiments.yaml")
    for job in grid.jobs(["inference"], ["DNABERT2"], [1], ["promoter_all"]):
        ckpt = job.argv[job.argv.index("--checkpoint") + 1]
        if "/conv_epinet/" in job.name:
            assert ckpt.endswith("/epinet"), f"{job.name} points at {ckpt}"
        else:
            assert ckpt.endswith("/base"), f"{job.name} points at {ckpt}"


def test_grid_dry_run_executes_nothing(tmp_path):
    """The driver must not run anything without --run."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_grid.py"),
         "--dry-run", "--stage", "train", "--backbone", "DNABERT2", "--seed", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "nothing was executed" in proc.stderr
    assert not (REPO_ROOT / "checkpoints").exists()
