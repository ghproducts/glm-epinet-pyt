# file to run last-layer diagonal Laplace inference on a trained DNABERT2 base checkpoint
"""
Last-layer diagonal Laplace inference for DNABERT2. See
nn_proj/models/laplace/laplace_head.py for the method itself (including why
the `laplace-torch` package was not used) and its predictive machinery;
this script only does the DNABERT2-specific plumbing (checkpoint loading,
tokenizer, dataset), mirroring inference.py's structure. Like inference.py,
it never trains anything -- the fitting step here is fitting the posterior
variance, a forward-only pass over the (already fine-tuned, frozen)
checkpoint.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from safetensors.torch import load_file
from transformers import set_seed

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.models.laplace import (
    LaplaceConfig,
    LaplaceSeqClassifier,
    fit_diagonal_laplace,
    predict_laplace,
    tune_prior_precision,
)
from .config import ModelArguments, DataArguments, TrainingArguments

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LaplaceArguments:
    prior_precision: float = field(
        default=1.0, metadata={"help": "Scalar Gaussian prior precision over the classifier weights/bias. "
                                        "Ignored (used only as a fallback if the search finds nothing) when "
                                        "--tune_prior_precision is set."}
    )
    laplace_fit_examples: Optional[int] = field(
        default=2000,
        metadata={"help": "Cap on how many training examples are used to fit the GGN diagonal (None = full training set)."},
    )
    classifier_attr: str = field(
        default="classifier", metadata={"help": "Dotted name of the model's final nn.Linear classification layer."}
    )
    tune_prior_precision: bool = field(
        default=False,
        metadata={"help": "Opt-in: instead of using --prior_precision as-is, grid search over "
                           "--prior_precision_candidates, minimizing predictive NLL on a held-out validation "
                           "slice carved out of the training data (same data_seed convention as scaling.py's "
                           "90/10 split). Off by default -- the hardcoded --prior_precision default is used "
                           "unless this is passed. See nn_proj/models/laplace/laplace_head.py's "
                           "tune_prior_precision() docstring."},
    )
    prior_precision_candidates: str = field(
        default="0.01,0.1,1.0,10.0,100.0,1000.0",
        metadata={"help": "Comma-separated candidate prior_precision values for --tune_prior_precision's grid search."},
    )
    prior_precision_val_frac: float = field(
        default=0.1,
        metadata={"help": "Fraction of the training split (stratified by label, held out via the same "
                           "90/10-style train_test_split convention scaling.py uses) reserved as the "
                           "validation slice --tune_prior_precision scores candidates against."},
    )


def run():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LaplaceArguments)
    )
    model_args, data_args, training_args, laplace_args = parser.parse_args_into_dataclasses()
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

    is_nt_task = data_args.data_path.startswith(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
    )
    if is_nt_task:
        task = data_args.data_path.split("/")[-1]
        fit_dataset = load_NT_tasks(task=task, split="train")
        test_dataset = load_NT_tasks(task=task, split="test", encode_labels=False)
    else:
        fit_dataset = load_local_dataset(
            path=data_args.data_path, encode_labels=True, rank=data_args.taxa_rank, taxa_df=data_args.taxa_df
        )
        # Local datasets carry no separate test split under this repo's
        # convention (inference.py's --data_path already points at the test
        # file for those); fitting and evaluating on the same file here is
        # only appropriate for a smoke test, not a reported number.
        test_dataset = fit_dataset

    if laplace_args.tune_prior_precision:
        if not (0.0 < laplace_args.prior_precision_val_frac < 1.0):
            raise ValueError(
                f"--prior_precision_val_frac must be in (0, 1); got {laplace_args.prior_precision_val_frac}"
            )
        # Held out from the GGN-fitting pool, not from the (separate) test
        # split -- same 90/10-style train_test_split convention scaling.py
        # uses for temperature scaling, so this validation slice is
        # train-derived just like the GGN fit itself, and both stay
        # disjoint from `test_dataset`.
        tune_split = fit_dataset.train_test_split(
            test_size=laplace_args.prior_precision_val_frac,
            seed=training_args.data_seed,
            stratify_by_column="labels",
        )
        ggn_fit_dataset, val_dataset_raw = tune_split["train"], tune_split["test"]
    else:
        ggn_fit_dataset, val_dataset_raw = fit_dataset, None

    fit_dataset_tok, fit_collator = prep_for_trainer(
        ggn_fit_dataset, tokenizer, max_length=training_args.model_max_length
    )

    num_labels = config.num_labels if hasattr(config, "num_labels") else data_args.num_labels
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        trust_remote_code=True,
    ).to(DEVICE)
    state_path = os.path.join(model_args.checkpoint, "model.safetensors")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Could not find checkpoint weights at: {state_path}")
    model.load_state_dict(load_file(state_path), strict=True)
    model.eval()

    prior_precision = laplace_args.prior_precision
    if laplace_args.tune_prior_precision:
        val_dataset_tok, val_collator = prep_for_trainer(
            val_dataset_raw, tokenizer, max_length=training_args.model_max_length
        )
        candidates = [float(x) for x in laplace_args.prior_precision_candidates.split(",") if x.strip()]
        tune_cfg = LaplaceConfig(
            classifier_attr=laplace_args.classifier_attr,
            max_examples=laplace_args.laplace_fit_examples,
        )
        print(f"Tuning prior_precision by grid search over {candidates} (minimizing held-out val NLL)...")
        best_precision, nll_by_precision = tune_prior_precision(
            model,
            fit_dataset_tok,
            fit_collator,
            val_dataset_tok,
            val_collator,
            cfg=tune_cfg,
            candidates=candidates,
            k_samples=model_args.num_samples,
            batch_size=training_args.per_device_eval_batch_size,
        )
        print(
            "[laplace] prior_precision grid search (val NLL): "
            + ", ".join(f"{p:g}={nll:.4f}" for p, nll in sorted(nll_by_precision.items()))
        )
        print(f"[laplace] selected prior_precision={best_precision:g} (lowest val NLL), overriding --prior_precision={laplace_args.prior_precision:g}")
        prior_precision = best_precision

    print("Fitting diagonal Laplace over the classification head...")
    lap_cfg = LaplaceConfig(
        classifier_attr=laplace_args.classifier_attr,
        prior_precision=prior_precision,
        max_examples=laplace_args.laplace_fit_examples,
    )
    weight_var, bias_var, n_used = fit_diagonal_laplace(
        model, fit_dataset_tok, fit_collator, lap_cfg, batch_size=training_args.per_device_eval_batch_size,
    )
    print(
        f"Fit on {n_used} examples. weight_var: mean={weight_var.mean():.3e} std={weight_var.std():.3e}; "
        f"bias_var: {bias_var.tolist()}"
    )

    lap_model = LaplaceSeqClassifier(
        model, weight_var.to(DEVICE), bias_var.to(DEVICE), classifier_attr=laplace_args.classifier_attr
    ).to(DEVICE)

    metadata_cols = [c for c in test_dataset.column_names if c not in ("sequence", "label", "labels")]
    test_ds_tok, test_collator = prep_for_trainer(
        test_dataset, tokenizer, max_length=training_args.model_max_length, metadata_cols=metadata_cols
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    outfile = os.path.join(training_args.output_dir, "inference_uncertainty.csv")
    predict_laplace(
        model=lap_model,
        dataset=test_ds_tok,
        collator=test_collator,
        k_samples=model_args.num_samples,
        batch_size=training_args.per_device_eval_batch_size,
        outfile=outfile,
        metadata_cols=metadata_cols,
    )


if __name__ == "__main__":
    run()
