# file to run inference for a DNABERT2 checkpoint trained by train_evidential.py
"""
Evidential inference for a DNABERT2 checkpoint from train_evidential.py.
Mirrors inference.py's structure; the uncertainty computation itself is
`nn_proj.models.evidential.evidential.predict_evidential`.

Reads `evidence_activation`/`loss_type`/`annealing_step` back from the
`evidential_config.json` sidecar written into `--checkpoint`, rather than
hardcoding defaults -- avoids silently scoring a non-default checkpoint
(e.g. `--evidence_activation relu`) with the wrong activation. Checkpoints
without the sidecar fall back to the old defaults with a printed warning.
"""
import json
import os

import torch
import transformers
from safetensors.torch import load_file
from transformers import set_seed

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.models.evidential import EvidentialConfig, EvidentialWrapper, HFEvidentialSeqClassifier, predict_evidential
from .config import ModelArguments, DataArguments, TrainingArguments

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    # train_evidential.py explicitly saves config.json/tokenizer files into
    # its own output_dir (see the comment there for why), so `--checkpoint`
    # here should point at *that* directory, not the plain base checkpoint.
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
    elif data_args.data_path.startswith("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
        task = data_args.data_path.split("/")[-1]
        task_dataset = load_NT_tasks(task=task, split="test", encode_labels=False)
    else:
        task_dataset = load_local_dataset(
            path=data_args.data_path, encode_labels=False, rank=data_args.taxa_rank, taxa_df=data_args.taxa_df
        )

    num_labels = config.num_labels if hasattr(config, "num_labels") else data_args.num_labels
    metadata_cols = [c for c in task_dataset.column_names if c not in ("sequence", "label", "labels")]
    task_dataset, data_collator = prep_for_trainer(
        task_dataset, tokenizer, max_length=training_args.model_max_length, metadata_cols=metadata_cols
    )

    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        trust_remote_code=True,
    )

    evidential_cfg_path = os.path.join(model_args.checkpoint, "evidential_config.json")
    if os.path.isfile(evidential_cfg_path):
        with open(evidential_cfg_path) as f:
            saved_cfg = json.load(f)
        evidential_cfg = EvidentialConfig(
            num_classes=num_labels,  # from the dataset/config at hand, not the sidecar
            evidence_activation=saved_cfg.get("evidence_activation", "softplus"),
            loss_type=saved_cfg.get("loss_type", "mse"),
            annealing_step=saved_cfg.get("annealing_step", 10),
        )
        print(f"[inference_evidential] loaded {evidential_cfg_path}: "
              f"evidence_activation={evidential_cfg.evidence_activation!r}, loss_type={evidential_cfg.loss_type!r}")
    else:
        evidential_cfg = EvidentialConfig(num_classes=num_labels)
        print(
            f"[inference_evidential] WARNING: no evidential_config.json found at {evidential_cfg_path} "
            f"(checkpoint predates this sidecar); falling back to EvidentialConfig defaults "
            f"(evidence_activation={evidential_cfg.evidence_activation!r}). If this checkpoint was trained "
            f"with a non-default --evidence_activation, these results will be wrong -- retrain or "
            f"reconstruct the sidecar by hand."
        )
    wrapper = EvidentialWrapper(base_model, evidential_cfg)
    model = HFEvidentialSeqClassifier(wrapper).to(DEVICE)

    state_path = os.path.join(model_args.checkpoint, "model.safetensors")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Could not find checkpoint weights at: {state_path}")
    model.load_state_dict(load_file(state_path), strict=True)
    model.eval()

    os.makedirs(training_args.output_dir, exist_ok=True)
    outfile = os.path.join(training_args.output_dir, "inference_uncertainty.csv")
    predict_evidential(
        model=model,
        dataset=task_dataset,
        collator=data_collator,
        batch_size=training_args.per_device_eval_batch_size,
        outfile=outfile,
        metadata_cols=metadata_cols,
    )


if __name__ == "__main__":
    evaluate()
