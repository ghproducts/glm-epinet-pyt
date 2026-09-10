# file to train an evidential-deep-learning DNABERT2 classifier from scratch
"""
Evidential deep learning training for DNABERT2 (see
nn_proj/models/evidential/evidential.py). Unlike train_epinet.py, this
fine-tunes the whole backbone from `model_name_or_path` with the
evidential loss from the start -- there's no frozen-base stage, since the
training objective itself changes rather than a head being added on top.

Mirrors train_base.py's dataset/model loading, swapping in
`EvidentialWrapper`/`HFEvidentialSeqClassifier` and a `Trainer` subclass
that computes the evidential loss (needs `self.state.epoch` for the KL
annealing coefficient, which only the `Trainer` has access to).
"""
import json
import os
from dataclasses import dataclass, field

import transformers
from transformers import set_seed
from datasets import ClassLabel

from nn_proj.common.utils import preprocess_logits_for_metrics, compute_metrics
from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.models.evidential import EvidentialConfig, EvidentialWrapper, HFEvidentialSeqClassifier, dirichlet_loss
from .config import ModelArguments, DataArguments, TrainingArguments
from .train_base import safe_save_model_for_hf_trainer


@dataclass
class EvidentialArguments:
    evidence_activation: str = field(
        default="softplus", metadata={"help": "'softplus' or 'relu' non-negative evidence activation."}
    )
    evidential_loss_type: str = field(
        default="mse", metadata={"help": "'mse' (Sensoy et al. eq. 5) or 'ce' (digamma, eq. 4)."}
    )
    annealing_step: int = field(
        default=10, metadata={"help": "Epochs until the KL regularizer reaches full weight."}
    )


class EvidentialTrainer(transformers.Trainer):
    """
    Overrides `compute_loss` (rather than computing loss inside
    `model.forward()`, as every other wrapper in this repo does) solely to
    get at `self.state.epoch` for the KL annealing coefficient -- see
    `HFEvidentialSeqClassifier`'s docstring for why.
    """

    def __init__(self, *args, evidential_cfg: EvidentialConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.evidential_cfg = evidential_cfg

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        evidence = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = dirichlet_loss(
            evidence,
            labels,
            num_classes=self.evidential_cfg.num_classes,
            epoch=self.state.epoch or 0.0,
            annealing_step=self.evidential_cfg.annealing_step,
            loss_type=self.evidential_cfg.loss_type,
        )
        inputs["labels"] = labels  # restore in case the Trainer reuses `inputs` after this call
        return (loss, outputs) if return_outputs else loss


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, EvidentialArguments)
    )
    model_args, data_args, training_args, evidential_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

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
        train_dataset = load_NT_tasks(task=task, split="train")
    else:
        train_dataset = load_local_dataset(
            path=data_args.data_path, encode_labels=True, rank=data_args.taxa_rank, taxa_df=data_args.taxa_df
        )

    print(len(train_dataset), "training examples loaded.")
    print("number of classes:", train_dataset.features["labels"].num_classes)

    split = train_dataset.train_test_split(
        test_size=0.1, seed=training_args.data_seed, stratify_by_column="labels"
    )
    train_dataset, val_dataset = split["train"], split["test"]

    train_dataset, data_collator = prep_for_trainer(
        train_dataset, tokenizer, max_length=training_args.model_max_length, metadata_cols=("taxid", "split")
    )
    val_dataset, _ = prep_for_trainer(
        val_dataset, tokenizer, max_length=training_args.model_max_length, metadata_cols=("taxid", "split")
    )

    label_feature = train_dataset.features["labels"]
    if isinstance(label_feature, ClassLabel):
        num_labels = label_feature.num_classes
    else:
        raise ValueError("Expected ClassLabel for 'labels' feature")

    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=num_labels,
        trust_remote_code=True,
        problem_type="single_label_classification",
    )

    evidential_cfg = EvidentialConfig(
        num_classes=num_labels,
        evidence_activation=evidential_args.evidence_activation,
        loss_type=evidential_args.evidential_loss_type,
        annealing_step=evidential_args.annealing_step,
    )
    wrapper = EvidentialWrapper(base_model, evidential_cfg)
    model = HFEvidentialSeqClassifier(wrapper)

    trainer = EvidentialTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        evidential_cfg=evidential_cfg,
    )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        # HFEvidentialSeqClassifier isn't a PreTrainedModel, so Trainer._save
        # writes no config.json on its own; save the base model's config and
        # the tokenizer explicitly so inference_evidential.py can reload both
        # from training_args.output_dir the same way inference.py does for
        # a plain base checkpoint (train_epinet.py has the same
        # tokenizer.save_pretrained call for the same reason -- its wrapper,
        # HFEpinetSeqClassifier, isn't a PreTrainedModel either).
        base_model.config.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

        # Persist the run's EvidentialConfig so inference_evidential.py can
        # read back the actual training-time settings instead of assuming
        # defaults (review item 3.5 / NEW_UQ_METHODS_REVIEW.md section 3.5:
        # inference used to hardcode evidence_activation="softplus"
        # regardless of what training actually used, silently producing
        # wrong evidence/alpha/vacuity for any --evidence_activation other
        # than the default). Small JSON sidecar next to config.json, the
        # same spirit as how every other run-specific setting in this repo
        # that inference needs back is either baked into config.json or (for
        # epinet's non-PreTrainedModel wrapper) saved alongside the
        # checkpoint -- see train_epinet.py's model_epinet.pt.
        evidential_cfg_path = os.path.join(training_args.output_dir, "evidential_config.json")
        with open(evidential_cfg_path, "w") as f:
            json.dump(
                {
                    "num_classes": evidential_cfg.num_classes,
                    "evidence_activation": evidential_cfg.evidence_activation,
                    "loss_type": evidential_cfg.loss_type,
                    "annealing_step": evidential_cfg.annealing_step,
                },
                f,
                indent=2,
            )
        print(f"Wrote {evidential_cfg_path}")


if __name__ == "__main__":
    train()
