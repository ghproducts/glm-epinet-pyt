import os
import json
import torch
import transformers
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from transformers import TrainerCallback

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import preprocess_logits_for_metrics, compute_metrics
from .config import ModelArguments, DataArguments, TrainingArguments
from nn_proj.models.epinet import (
    EpinetConfig,
    EpinetWrapper,
    HFEpinetSeqClassifier,
    MLPEpinetWithPrior,
    MLPEpinetWithConvPrior,
    predict,
)
from nn_proj.models.epinet.feature_fns import hyenaDNA_feature_fn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _first_nonempty(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


class InferenceEachEpochCallback(TrainerCallback):
    """Run epinet inference on a fixed test set at the end of each epoch.

    Writes: <output_dir>/inference_uncertainty_epoch{E}.csv
    """

    def __init__(self, dataset, collator, k_samples, uncertainty_method, out_dir, config):
        self.dataset = dataset
        self.collator = collator
        self.k_samples = int(k_samples)
        self.uncertainty_method = uncertainty_method
        self.out_dir = out_dir
        self.config = config
        os.makedirs(self.out_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Write only once in distributed training
        if not state.is_world_process_zero:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        was_training = model.training
        model.eval()

        # state.epoch can be float (e.g., 0.999...), so round sensibly
        epoch_idx = int(round(state.epoch)) if state.epoch is not None else 0
        if epoch_idx < 1:
            # If Trainer reports epoch 0 at the first callback, still name it epoch1
            epoch_idx = 1

        outfile = os.path.join(self.out_dir, f"inference_uncertainty_epoch{epoch_idx}.csv")

        with torch.no_grad():
            predict(
                model=model,
                dataset=self.dataset,
                collator=self.collator,
                k_samples=self.k_samples,
                batch_size=args.per_device_eval_batch_size,
                uncertainty_method=self.uncertainty_method,
                outfile=outfile,
            )

        # Optional: relabel preds + prob columns to original string labels (mirrors inference.py)
        try:
            df = pd.read_csv(outfile)
            if hasattr(self.config, "label2id") and isinstance(self.config.label2id, dict) and len(self.config.label2id) > 0:
                id2label = {v: k for k, v in self.config.label2id.items()}
                if "pred" in df.columns:
                    df["pred"] = df["pred"].map(id2label)

                rename_map = {}
                for i in range(len(id2label)):
                    old_name = f"prob_{i}"
                    if old_name in df.columns:
                        rename_map[old_name] = f"prob_{id2label.get(i, i)}"
                if rename_map:
                    df = df.rename(columns=rename_map)

                df.to_csv(outfile, index=False)
        except Exception as e:
            print(f"[warn] postprocess relabel failed for {outfile}: {e}")

        if was_training:
            model.train()

        return control


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = transformers.AutoConfig.from_pretrained(
        model_args.checkpoint,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    # tokenizer
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

    # ---- TRAIN dataset ----
    if data_args.data_path is None:
        raise ValueError("data_path must be specified.")
    elif data_args.data_path.startswith("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
        task = data_args.data_path.split("/")[-1]
        train_dataset = load_NT_tasks(task=task, split="train")
    else:
        train_dataset = load_local_dataset(path=data_args.data_path)

    split = train_dataset.train_test_split(
        test_size=0.1, seed=training_args.seed, stratify_by_column="labels"
    )
    train_dataset, val_dataset = split["train"], split["test"]

    train_dataset, data_collator = prep_for_trainer(train_dataset, tokenizer)
    val_dataset, _ = prep_for_trainer(val_dataset, tokenizer)

    num_labels = data_args.num_labels if data_args.num_labels is not None else train_dataset.features["labels"].num_classes
    num_labels = config.num_labels if hasattr(config, "num_labels") else num_labels

    # ---- TEST dataset (for per-epoch inference) ----
    if data_args.test_path is None:
        raise ValueError("data_path must be specified.")
    elif data_args.test_path.startswith("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
        task = data_args.test_path.split("/")[-1]
        test_dataset = load_NT_tasks(task=task, split="test", encode_labels=False)
    else:
        test_dataset = load_local_dataset(path=data_args.test_path, encode_labels=False)

    test_dataset, _ = prep_for_trainer(test_dataset, tokenizer)


    

    # base model
    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.checkpoint,
        cache_dir=training_args.cache_dir,
        config=config,
        trust_remote_code=True,
    )

    # attach epinet
    epi_cfg = EpinetConfig(num_classes=num_labels, include_inputs=True, vocab_size=config.vocab_size)
    wrapper = EpinetWrapper(base_model, hyenaDNA_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=8).to(DEVICE)

    # freeze base
    backbone = model.wrapper.base
    epinet = model.wrapper.epinet
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    # stop gradient through features (epinet-only training)
    model.wrapper.cfg.stop_grad_features = True

    # build epinet internals
    with torch.no_grad():
        dummy = tokenizer(
            "ACGT",
            truncation=True,
            padding="max_length",
            max_length=training_args.model_max_length,
            return_tensors="pt",
        )
        dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
        _ = model(**dummy)

    # Optional: load epinet weights (weights-only resume)
    epinet_path = _first_nonempty(
        getattr(model_args, "epinet_path", None),
        getattr(training_args, "epinet_path", None),
    )
    if epinet_path:
        sd_path = os.path.join(epinet_path, "pytorch_model.bin")
        if os.path.exists(sd_path):
            sd = torch.load(sd_path, map_location="cpu")
            model.load_state_dict(sd, strict=True)
            model.to(DEVICE)
        else:
            raise FileNotFoundError(f"--epinet_path was set, but {sd_path} does not exist")

    # sanity prints
    print("Trainable (epinet) params:", sum(p.requires_grad for p in epinet.parameters()))
    print("Trainable (base)   params:", sum(p.requires_grad for p in backbone.parameters()))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"{len(trainable)} trainable tensors:", trainable[:10], "...")

    # callback config
    k_samples = int(getattr(model_args, "num_samples", 10))
    uncertainty_method = getattr(model_args, "uncertainty_method", "epinet")

    callbacks = [
        InferenceEachEpochCallback(
            dataset=test_dataset,
            collator=data_collator,
            k_samples=k_samples,
            uncertainty_method=uncertainty_method,
            out_dir=training_args.output_dir,
            config=config,
        )
    ]

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer.train()

    # save epinet model (matches your original script)
    if getattr(training_args, "save_model", False):
        trainer.save_state()
        os.makedirs(training_args.output_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "base_model_name": model_args.model_name_or_path,
                "num_labels": num_labels,
                "epinet_cfg": model.cfg.__dict__,
                "feature_fn": "hyenaDNA_feature_fn",
            },
            os.path.join(training_args.output_dir, "model_epinet.pt"),
        )
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
