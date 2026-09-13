#!/usr/bin/env python3
"""Shared backbone/dataset/IO layer for the UQ sweep.

One place that knows how each backbone loads and how each evaluation dataset
is built, so `run_sweep.py` holds only the UQ methods themselves. Replaces the
per-run hardcoded scripts under `promoter_alisim/uncertainty_eval/` and
`aleatoric_boundary/`.

Every method writes the same per-example schema, and unlike the earlier
scripts that schema includes the predicted **probability** vector -- without it
ECE/NLL/Brier are not computable and have to be reconstructed by inverting
entropy, which only works for binary tasks.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd
import torch
import transformers
from safetensors.torch import load_file

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer  # noqa: E402

def _default_ckpt_root() -> str:
    """Checkpoints are large and gitignored, so they live in the main checkout
    even when this runs from a git worktree. Prefer $GLM_CKPT_ROOT, then this
    tree, then the enclosing main checkout."""
    env = os.environ.get("GLM_CKPT_ROOT")
    if env:
        return env
    local = os.path.join(REPO, "checkpoints")
    if os.path.isdir(local):
        return local
    marker = os.path.join(".claude", "worktrees")
    if marker in REPO:
        main_checkout = REPO.split(marker)[0].rstrip(os.sep)
        candidate = os.path.join(main_checkout, "checkpoints")
        if os.path.isdir(candidate):
            return candidate
    return local


def _find_data_file(rel: str) -> str:
    """Some generated data dirs are gitignored, so they exist only in the main
    checkout even when running from a worktree (same situation as checkpoints)."""
    local = os.path.join(REPO, rel)
    if os.path.exists(local):
        return local
    marker = os.path.join(".claude", "worktrees")
    if marker in REPO:
        candidate = os.path.join(REPO.split(marker)[0].rstrip(os.sep), rel)
        if os.path.exists(candidate):
            return candidate
    return local


CKPT_ROOT = _default_ckpt_root()
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 300  # promoter_all


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Backbone:
    name: str
    hub_id: str
    tokens_per_base: float
    feature_fn: str          # nn_proj.models.epinet.feature_fns attribute
    custom_class: bool = False   # CARMANIA isn't a native AutoModelForSequenceClassification
    # manuscript ID promoter_all error % (mean, sd) for the `base` strategy
    manuscript_error: tuple[float, float] = (float("nan"), float("nan"))

    @property
    def max_length(self) -> int:
        return max(1, int(round(SEQ_LEN * self.tokens_per_base)))


BACKBONES: dict[str, Backbone] = {
    b.name: b for b in [
        Backbone("NT_transformer", "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
                 0.1667, "NT_feature_fn", manuscript_error=(13.4, 1.2)),
        Backbone("DNABERT2", "zhihan1996/DNABERT-2-117M",
                 0.25, "NT_feature_fn", manuscript_error=(13.9, 1.1)),
        Backbone("hyenaDNA", "LongSafari/hyenadna-small-32k-seqlen-hf",
                 1.0, "hyenaDNA_feature_fn", manuscript_error=(15.0, 0.2)),
        Backbone("CARMANIA", "MsAlEhR/carmania-160k-seqlen-human",
                 1.0, "NT_feature_fn", custom_class=True, manuscript_error=(14.0, 0.2)),
    ]
}


def ckpt_path(backbone: str, seed: int | str, kind: str, task_dir: str = "label_noise_r00") -> str:
    """`kind` is the checkpoint flavour: base / epinet / epinet_zfix / evidential."""
    return os.path.join(CKPT_ROOT, f"seed_{seed}", backbone, task_dir, kind)


def load_checkpoint_weights(ckpt_dir: str) -> dict:
    """Four save layouts exist in this repo, in preference order:

    - `model.safetensors` (DNABERT2/CARMANIA base, and the first epinet_zfix)
    - `pytorch_model.bin` (hyenaDNA base)
    - `model_epinet.pt`   (what train_epinet.py --save_model writes: a dict
                           with the state_dict under a "state_dict" key, plus
                           the EpinetConfig used)
    - `checkpoint-*/pytorch_model.bin` (HF Trainer per-epoch dirs, used as a
                           last resort when no top-level save happened; the
                           highest step is taken)
    """
    st = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(st):
        return load_file(st)
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    epi = os.path.join(ckpt_dir, "model_epinet.pt")
    if os.path.exists(epi):
        blob = torch.load(epi, map_location="cpu", weights_only=False)
        return blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    # Trainer ran with load_best_model_at_end, so the highest-step checkpoint is
    # NOT necessarily the best one -- read trainer_state.json rather than
    # assuming, or we silently score non-best weights.
    import glob
    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(state_path):
        import json as _json
        best = _json.load(open(state_path)).get("best_model_checkpoint")
        if best:
            for cand in (os.path.join(best, "pytorch_model.bin"),
                         os.path.join(best, "model.safetensors"),
                         os.path.join(ckpt_dir, os.path.basename(best), "pytorch_model.bin"),
                         os.path.join(ckpt_dir, os.path.basename(best), "model.safetensors")):
                if os.path.exists(cand):
                    return (load_file(cand) if cand.endswith(".safetensors")
                            else torch.load(cand, map_location="cpu", weights_only=True))
    inner = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint-*", "pytorch_model.bin")),
                   key=lambda p: int(p.split("checkpoint-")[1].split(os.sep)[0]))
    if inner:
        return torch.load(inner[-1], map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"no model.safetensors / pytorch_model.bin / model_epinet.pt / checkpoint-*/ "
        f"weights in {ckpt_dir}"
    )


def load_epinet_saved_config(ckpt_dir: str) -> Optional[dict]:
    """EpinetConfig fields recorded by train_epinet.py --save_model, so the
    wrapper is rebuilt with the config it was trained under rather than
    re-derived defaults."""
    epi = os.path.join(ckpt_dir, "model_epinet.pt")
    if not os.path.exists(epi):
        return None
    blob = torch.load(epi, map_location="cpu", weights_only=False)
    return blob.get("epinet_cfg") if isinstance(blob, dict) else None


def load_config(bb: Backbone, num_labels: int = 2, ckpt_dir: Optional[str] = None):
    """From the hub id, never the checkpoint dir: DNABERT2's remote code
    resolves differently for a local path and raises
    `BertConfig has no attribute pad_token_id` on newer transformers.

    CARMANIA is the exception -- its config *must* come from the checkpoint,
    whose config.json carries the `encoder_name` its constructor requires.
    """
    if bb.custom_class:
        from nn_proj.models.CARMANIA.model import CarmaniaSequenceClassificationConfig
        if ckpt_dir is None:
            raise ValueError(f"{bb.name} needs ckpt_dir to load its config (encoder_name lives there)")
        return CarmaniaSequenceClassificationConfig.from_pretrained(ckpt_dir)
    return transformers.AutoConfig.from_pretrained(
        bb.hub_id, num_labels=num_labels, trust_remote_code=True
    )


def load_tokenizer(bb: Backbone):
    """From the hub id, not the checkpoint: some saved `tokenizer.json` files
    were written by a newer `tokenizers` than the pinned runtime can parse."""
    tok = transformers.AutoTokenizer.from_pretrained(
        bb.hub_id, model_max_length=bb.max_length, padding_side="right",
        use_fast=True, trust_remote_code=True,
    )
    if getattr(tok, "eos_token", None) is None and getattr(tok, "pad_token", None) is not None:
        tok.eos_token = tok.pad_token
    return tok


def build_base_arch(bb: Backbone, config=None, num_labels: int = 2,
                    ckpt_dir: Optional[str] = None):
    """Randomly-initialized-head architecture, before checkpoint weights."""
    config = config if config is not None else load_config(bb, num_labels, ckpt_dir)
    if bb.custom_class:
        from nn_proj.models.CARMANIA.model import CarmaniaForSequenceClassification
        return CarmaniaForSequenceClassification(config)
    return transformers.AutoModelForSequenceClassification.from_pretrained(
        bb.hub_id, config=config, trust_remote_code=True
    )


def load_finetuned(bb: Backbone, ckpt_dir: str, config=None, num_labels: int = 2,
                   strict: bool = True):
    """Architecture from hub, weights from checkpoint. Raises on key mismatch
    unless `strict=False` -- a silent mismatch means evaluating a
    partly-randomly-initialized model, which looks like a bad result rather
    than a bug."""
    model = build_base_arch(bb, config, num_labels, ckpt_dir=ckpt_dir)
    missing, unexpected = model.load_state_dict(load_checkpoint_weights(ckpt_dir), strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"{bb.name} {ckpt_dir}: state_dict mismatch "
            f"(missing={list(missing)[:6]}, unexpected={list(unexpected)[:6]})"
        )
    return model.to(DEVICE).eval()


def get_feature_fn(bb: Backbone) -> Callable:
    from nn_proj.models.epinet import feature_fns
    return getattr(feature_fns, bb.feature_fn)


def find_classifier_attr(model, num_labels: int = 2) -> str:
    """Dotted name of the final classification `nn.Linear`, for Laplace's
    `classifier_attr`.

    Not hardcodable: DNABERT2/CARMANIA expose `classifier` as a single Linear,
    but NT's `EsmForSequenceClassification.classifier` is an
    `EsmClassificationHead` (dense + out_proj), so the last Linear is
    `classifier.out_proj`. Picking the wrong module silently fits the
    posterior over the wrong parameters. Returns the *last* Linear whose
    out_features == num_labels.
    """
    candidates = [name for name, mod in model.named_modules()
                  if isinstance(mod, torch.nn.Linear) and mod.out_features == num_labels]
    if not candidates:
        raise RuntimeError(
            f"no nn.Linear with out_features=={num_labels} found; "
            f"cannot locate the classification head"
        )
    return candidates[-1]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@dataclass
class EvalDataset:
    name: str
    tokenized: object          # HF dataset incl. metadata columns
    collate_ds: object         # same rows, model-input columns only
    collator: object
    metadata_cols: list[str]
    labels: list[int]
    tokenizer: object = None

    def loader(self, batch_size: int):
        return torch.utils.data.DataLoader(
            self.collate_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collator
        )

    def metadata_rows(self, start: int, end: int) -> list[dict]:
        rows = []
        for i in range(start, end):
            row = {}
            for c in self.metadata_cols:
                v = self.tokenized[c][i]
                row[c] = v.item() if isinstance(v, torch.Tensor) else v
            rows.append(row)
        return rows


ALISIM_CSV = _find_data_file("data_gen/promoter_alisim/csv_data/promoter_alisim_dense.csv")
ALISIM_EXTENDED_CSV = _find_data_file("data_gen/promoter_alisim/csv_data/promoter_alisim_extended.csv")
BOUNDARY_MARGINS = _find_data_file("data_gen/aleatoric_boundary/csv_data/margin_scores.csv")

ALISIM_METADATA = ["anchor_id", "branch_length", "realized_identity_to_anchor", "original_label"]


def build_alisim(bb: Backbone, tokenizer, csv_path: str = ALISIM_CSV) -> EvalDataset:
    ds = load_local_dataset(csv_path, encode_labels=True)
    meta = [c for c in ALISIM_METADATA if c in ds.column_names]
    tokenized, collator = prep_for_trainer(ds, tokenizer, max_length=bb.max_length, metadata_cols=meta)
    collate_ds = tokenized.remove_columns(["sequence"] + meta)
    return EvalDataset("alisim", tokenized, collate_ds, collator, meta, list(ds["labels"]), tokenizer)


def build_boundary(bb: Backbone, tokenizer) -> EvalDataset:
    """promoter_all test split in its original, unshuffled row order, with the
    precomputed margin quartile joined on by row index. Row order must match
    margin_scores.csv exactly -- that file's `idx` is positional."""
    ds = load_NT_tasks(task="promoter_all", split="test")
    margins = pd.read_csv(BOUNDARY_MARGINS)
    if len(margins) != len(ds):
        raise RuntimeError(f"margin_scores rows ({len(margins)}) != promoter_all test rows ({len(ds)})")
    ds = ds.add_column("margin", margins["margin"].tolist())
    ds = ds.add_column("quartile", margins["quartile"].tolist())
    labels = list(ds["labels"])
    if labels != margins["label"].tolist():
        raise RuntimeError("row order mismatch: promoter_all labels != margin_scores labels")
    meta = ["margin", "quartile"]
    tokenized, collator = prep_for_trainer(ds, tokenizer, max_length=bb.max_length, metadata_cols=meta)
    collate_ds = tokenized.remove_columns(["sequence"] + meta)
    return EvalDataset("boundary", tokenized, collate_ds, collator, meta, labels, tokenizer)


LABEL_NOISE_TRAIN = _find_data_file("data_gen/label_noise/csv_data_r00/train.csv")
VAL_FRACTION = 0.1  # matches the manuscript's 90/10 train/val split


def build_train_subset(bb: Backbone, tokenizer, max_examples: Optional[int] = None) -> EvalDataset:
    """The held-out tail of the training split (the manuscript's D_val), used to
    fit post-hoc calibrators -- temperature, the Laplace posterior, and the
    conformal threshold. Deliberately NOT the evaluation set: fitting a
    calibrator on the data you then score makes the resulting ECE/coverage
    meaningless.

    `label_noise_r00` is byte-identical to clean promoter_all train, and is
    verified disjoint from the promoter_all test split.
    """
    if not os.path.exists(LABEL_NOISE_TRAIN):
        raise FileNotFoundError(
            f"{LABEL_NOISE_TRAIN} not found; it holds the training split needed to fit "
            f"post-hoc calibrators (it is gitignored -- see data_gen/label_noise/)"
        )
    df = pd.read_csv(LABEL_NOISE_TRAIN)
    n_val = int(round(len(df) * VAL_FRACTION))
    df = df.iloc[len(df) - n_val:].reset_index(drop=True)  # tail == the 10% val slice
    if max_examples:
        df = df.iloc[:max_examples].reset_index(drop=True)
    from datasets import Dataset
    ds = Dataset.from_dict({"sequence": df["sequence"].tolist(),
                            "labels": [int(x) for x in df["label"].tolist()]})
    tokenized, collator = prep_for_trainer(ds, tokenizer, max_length=bb.max_length, metadata_cols=())
    collate_ds = tokenized.remove_columns(["sequence"])
    return EvalDataset("train_val", tokenized, collate_ds, collator, [], list(ds["labels"]), tokenizer)


DATASET_BUILDERS = {
    "alisim": build_alisim,
    "alisim_extended": lambda bb, tok: build_alisim(bb, tok, ALISIM_EXTENDED_CSV),
    "boundary": build_boundary,
}


# ---------------------------------------------------------------------------
# Result IO
# ---------------------------------------------------------------------------

def cell_path(dataset: str, backbone: str, method: str, seed: int | str) -> str:
    d = os.path.join(RESULTS_DIR, dataset, backbone)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{method}__seed{seed}.csv")


def write_cell(rows: list[dict], dataset: str, backbone: str, method: str, seed: int | str) -> str:
    df = pd.DataFrame(rows)
    df.insert(0, "backbone", backbone)
    df.insert(1, "dataset", dataset)
    df.insert(2, "seed", str(seed))
    path = cell_path(dataset, backbone, method, seed)
    df.to_csv(path, index=False)
    return path


def probs_to_row_fields(probs: torch.Tensor) -> dict:
    """Persist the full mean predictive distribution so ECE/NLL/Brier are
    directly computable later (the older scripts saved only derived entropies,
    which made proper scoring rules unrecoverable except by entropy inversion
    on binary tasks)."""
    p = probs.tolist()
    out = {f"prob_{i}": float(v) for i, v in enumerate(p)}
    out["prob_max"] = float(max(p))
    return out
