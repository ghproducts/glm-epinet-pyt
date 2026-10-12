# file to run split conformal prediction on a trained DNABERT2 base checkpoint
"""
Split conformal prediction for DNABERT2 (Vovk et al.; Romano, Sesia &
Candès 2020 for the "aps" score -- see nn_proj/common/conformal.py for the
method itself and its documented caveats). Post-hoc: this script never
touches a single fine-tuned weight, it only reads logits off the frozen
base checkpoint.

IMPORTANT -- what "the test set" means for this method specifically:
split conformal's marginal coverage guarantee requires the calibration and
evaluation points to be *exchangeable*. An earlier version of this script
calibrated on a 90/10 split of the *training* data (reusing `scaling.py`'s
convention) and evaluated on the dataset's separate official `test` split.
That measurably violates exchangeability for this checkpoint (86.8% vs
85.8% accuracy, and a notably different nonconformity-score distribution
specifically at the 80th percentile) and produced a reproducible ~1-3
point *under*-coverage at `target_coverage=0.80` across 5 independent
calibration seeds -- see `.docs/NEW_UQ_METHODS_FIXES.md` for the full
before/after numbers.

The fix: calibration and evaluation are now two independent, stratified
random draws from the *same* pool -- the task's official `test` split
(or `--test_data_path` for local datasets). `--conformal_cal_frac`
(default 0.5) controls what fraction of that pool goes to calibration; the
remaining `1 - conformal_cal_frac` fraction is what `empirical_coverage`
is computed against and what's written to the output CSV.

**Consequence, stated plainly**: unlike every other UQ method in this
repo (`base`, `base_scaled`, `mc_dropout`, `epinet`), which all evaluate
against the *full* official test split, this method's reported numbers
(coverage, mean set size, the output CSV) only cover a `1 -
conformal_cal_frac` random subset of it -- the other half (by default) was
spent on calibration and never scored. This is a necessary trade-off of
split conformal, not a bug: there is no way to get a valid marginal
coverage guarantee without holding out *some* exchangeable calibration
data, and reusing train-derived data for that (the old behavior) is what
caused the coverage violation above. Don't directly compare this method's
per-example counts/accuracy against the other methods' full-test-split
numbers without accounting for this.

Two data passes:

  1. Calibration -- fit a scalar threshold `qhat` for the requested
     `--target_coverage` on a stratified random `conformal_cal_frac`
     fraction of the test split.
  2. Evaluation -- apply that threshold to the complementary
     `1 - conformal_cal_frac` fraction of the test split and report, per
     example, the prediction set and its size, plus the two aggregate
     numbers that matter for conformal prediction: empirical coverage and
     mean set size (see .docs/NEW_UQ_METHODS.md for the numbers and why
     those two, not e.g. accuracy, are what's reported).

Reuses `collect_logits_labels` from `scaling.py` for the forward passes
(rather than reimplementing a forward loop a second time), but -- unlike
that reuse used to imply -- metadata columns (e.g. taxid/split for pbsim)
are captured separately before `collect_logits_labels` strips them, and
reattached to the output CSV by row order, the same `metadata_cols`
mechanism `nn_proj.models.epinet.epinet.predict`/`predict_laplace`/
`predict_evidential` already use. See `run_conformal` below.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
import pandas as pd
from datasets import ClassLabel
from safetensors.torch import load_file
from transformers import set_seed

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.common.conformal import (
    score_nonconformity,
    aps_nonconformity,
    fit_conformal_threshold,
    predict_sets,
)
from .config import ModelArguments, DataArguments, TrainingArguments
from .scaling import collect_logits_labels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ConformalArguments:
    target_coverage: float = field(
        default=0.9, metadata={"help": "Desired marginal coverage, 1 - alpha."}
    )
    conformal_method: str = field(
        default="score",
        metadata={"help": "Nonconformity score: 'score' (1 - softmax(true)) or 'aps' (Romano et al. 2020)."},
    )
    conformal_cal_frac: float = field(
        default=0.5,
        metadata={"help": "Fraction of the test split (stratified by label) reserved for calibration; "
                           "the remaining (1 - conformal_cal_frac) fraction is what empirical_coverage is "
                           "computed against and what's written to the output CSV. See module docstring: "
                           "this makes 'the test set' mean something narrower for this method than for the "
                           "other UQ methods in this repo, by necessity."},
    )
    test_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Held-out test file for local (non-HF-hub) datasets, from which BOTH the "
                           "calibration and evaluation subsets are drawn (see conformal_cal_frac). "
                           "Required unless --data_path is an InstaDeepAI NT task, which already has a "
                           "'test' split. --data_path itself is only used to decide whether this is an NT "
                           "hub task; it is no longer used as a source of calibration data."},
    )


def run_conformal():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ConformalArguments)
    )
    model_args, data_args, training_args, conformal_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    config = transformers.AutoConfig.from_pretrained(
        model_args.checkpoint, cache_dir=training_args.cache_dir, trust_remote_code=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    if data_args.data_path is None:
        raise ValueError("data_path must be specified.")

    if not (0.0 < conformal_args.conformal_cal_frac < 1.0):
        raise ValueError(
            f"--conformal_cal_frac must be in (0, 1); got {conformal_args.conformal_cal_frac}"
        )

    is_nt_task = data_args.data_path.startswith(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
    )
    if is_nt_task:
        task = data_args.data_path.split("/")[-1]
        test_pool = load_NT_tasks(task=task, split="test", encode_labels=False)
    else:
        if conformal_args.test_data_path is None:
            raise ValueError(
                "For local datasets, --test_data_path must point at the held-out test file that "
                "BOTH the calibration and evaluation subsets are drawn from (see module docstring)."
            )
        test_pool = load_local_dataset(
            path=conformal_args.test_data_path, encode_labels=False, rank=data_args.taxa_rank, taxa_df=data_args.taxa_df
        )

    # Calibration and evaluation must be exchangeable draws from the same
    # pool -- see module docstring for why this is no longer a split of the
    # *training* data. `train_test_split(stratify_by_column=...)` requires
    # a ClassLabel feature; the test pool's labels are plain ints (loaded
    # with encode_labels=False to avoid re-deriving a label<->id mapping),
    # so encode them for the purposes of this split only.
    strat_pool = test_pool if isinstance(test_pool.features["labels"], ClassLabel) else test_pool.class_encode_column("labels")
    cal_eval_split = strat_pool.train_test_split(
        test_size=1.0 - conformal_args.conformal_cal_frac,
        seed=training_args.data_seed,
        stratify_by_column="labels",
    )
    cal_dataset, eval_dataset = cal_eval_split["train"], cal_eval_split["test"]

    cal_dataset_tok, cal_collator = prep_for_trainer(
        cal_dataset, tokenizer, max_length=training_args.model_max_length
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        trust_remote_code=True,
    )
    state_path = os.path.join(model_args.checkpoint, "model.safetensors")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Could not find checkpoint weights at: {state_path}")
    model.load_state_dict(load_file(state_path), strict=True)

    print(
        f"Calibrating conformal threshold on a stratified {conformal_args.conformal_cal_frac:.2f} "
        f"fraction of the test split ({len(cal_dataset)} examples)..."
    )
    cal_logits, cal_labels = collect_logits_labels(
        model=model, dataset=cal_dataset_tok, collate_fn=cal_collator, batch_size=training_args.per_device_eval_batch_size,
    )
    cal_probs = torch.softmax(cal_logits.float(), dim=-1)

    alpha = 1.0 - conformal_args.target_coverage
    if conformal_args.conformal_method == "aps":
        cal_scores = aps_nonconformity(cal_probs, cal_labels)
    else:
        cal_scores = score_nonconformity(cal_probs, cal_labels)
    qhat = fit_conformal_threshold(cal_scores, alpha=alpha)
    print(
        f"alpha={alpha:.3f} (target coverage {conformal_args.target_coverage:.3f}), "
        f"method={conformal_args.conformal_method}, qhat={qhat:.6f}, n_cal={len(cal_labels)}"
    )

    # Metadata passthrough (review item 1.5): capture metadata columns
    # before collect_logits_labels strips them, and reattach by row order
    # below -- the same convention predict()/predict_laplace()/
    # predict_evidential() use.
    metadata_cols = [c for c in eval_dataset.column_names if c not in ("sequence", "label", "labels")]

    def _plain(v):
        return v.item() if hasattr(v, "item") else v

    eval_metas = [{col: _plain(eval_dataset[col][i]) for col in metadata_cols} for i in range(len(eval_dataset))]

    eval_ds_tok, eval_collator = prep_for_trainer(
        eval_dataset, tokenizer, max_length=training_args.model_max_length, metadata_cols=metadata_cols
    )

    print(f"Scoring the held-out evaluation fraction ({len(eval_ds_tok)} examples)...")
    eval_logits, eval_labels = collect_logits_labels(
        model=model, dataset=eval_ds_tok, collate_fn=eval_collator, batch_size=training_args.per_device_eval_batch_size,
    )
    eval_probs = torch.softmax(eval_logits.float(), dim=-1)

    membership = predict_sets(eval_probs, qhat, method=conformal_args.conformal_method)  # [N,C] bool
    set_sizes = membership.sum(dim=-1)
    in_set = membership.gather(1, eval_labels.view(-1, 1).long()).squeeze(1)
    empirical_coverage = in_set.float().mean().item()
    mean_set_size = set_sizes.float().mean().item()
    print(
        f"Empirical coverage = {empirical_coverage:.4f} (target {conformal_args.target_coverage:.3f}); "
        f"mean set size = {mean_set_size:.3f} / {eval_probs.shape[-1]} classes "
        f"(n_eval={eval_probs.shape[0]}, n_cal={len(cal_labels)})"
    )

    predicted_class = eval_probs.argmax(dim=-1)
    max_confidence = eval_probs.max(dim=-1).values

    rows = []
    n_classes = eval_probs.shape[-1]
    for i in range(eval_probs.shape[0]):
        included = [c for c in range(n_classes) if bool(membership[i, c])]
        row = {
            "labels": int(eval_labels[i]),
            "pred": int(predicted_class[i]),
            "max_confidence": float(max_confidence[i]),
            "set_size": int(set_sizes[i]),
            "in_set": bool(in_set[i]),
            "pred_set": "|".join(str(c) for c in included),
            "qhat": qhat,
            "alpha_target": alpha,
            "target_coverage": conformal_args.target_coverage,
            "conformal_method": conformal_args.conformal_method,
            "conformal_cal_frac": conformal_args.conformal_cal_frac,
        }
        rows.append({**row, **eval_metas[i]})

    os.makedirs(training_args.output_dir, exist_ok=True)
    outfile = os.path.join(training_args.output_dir, "inference_uncertainty.csv")
    pd.DataFrame(rows).to_csv(outfile, index=False)
    print(
        f"[conformal] wrote {outfile} with {len(rows)} rows; "
        f"empirical_coverage={empirical_coverage:.4f}, mean_set_size={mean_set_size:.3f}"
    )


if __name__ == "__main__":
    run_conformal()
