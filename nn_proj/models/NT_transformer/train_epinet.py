import torch
import transformers
import numpy as np
import json
import os
from torch.utils.data import Dataset
from safetensors.torch import load_file

from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import preprocess_logits_for_metrics, compute_metrics
from .config import ModelArguments, DataArguments, TrainingArguments
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithPrior, MLPEpinetWithConvPrior
from nn_proj.models.epinet.feature_fns import NT_feature_fn, NT_first_last_feature_fn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    config = transformers.AutoConfig.from_pretrained(
        model_args.checkpoint,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    # load tokenizer
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

    # load datasets
    # was worried about label encoding, but should be fine since I am using the exact same dataset as the base model
    if data_args.data_path is None:
        raise ValueError("data_path must be specified.")
    elif data_args.data_path.startswith("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
        task = data_args.data_path.split("/")[-1]
        train_dataset = load_NT_tasks(task=task, split="train")
    else:
        train_dataset = load_local_dataset(path=data_args.data_path)

    # data split
    split = train_dataset.train_test_split(test_size=0.1, seed=training_args.seed, stratify_by_column="labels")
    train_dataset, val_dataset = split["train"], split["test"]

    train_dataset, data_collator = prep_for_trainer(train_dataset, tokenizer)
    val_dataset, _   = prep_for_trainer(val_dataset, tokenizer)

    num_labels = data_args.num_labels if data_args.num_labels is not None else train_dataset.features["labels"].num_classes

    # load model and attatch epinet
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=num_labels,
        trust_remote_code=True,
    )

    # load checkpoint weights
    state_path = os.path.join(model_args.checkpoint, "model.safetensors")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Could not find checkpoint weights at: {state_path}")
    sd = load_file(state_path)
    model.load_state_dict(sd, strict=True)

    # combine with epinet to form model
    epi_cfg = EpinetConfig(num_classes=num_labels, include_inputs=True, vocab_size=config.vocab_size)
    #wrapper = EpinetWrapper(model, NT_first_last_feature_fn, epi_cfg, epinet=MLPEpinetWithPrior)
    wrapper = EpinetWrapper(model, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
    model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=8).to(DEVICE)

    # freeze base
    backbone = model.wrapper.base          # HF model 
    epinet = model.wrapper.epinet        # epinet 

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()                          

    #for p in epinet.parameters():
    #    p.requires_grad = True
    
    # Stop gradient
    model.wrapper.cfg.stop_grad_features = True

    # build epinet internals
    with torch.no_grad():
        dummy = tokenizer("ACGT", truncation=True, padding="max_length", max_length=training_args.model_max_length, return_tensors="pt")
        dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
        _ = model(**dummy)  # builds epinet internals

    # temp checks
    print("Trainable (epinet) params:", sum(p.requires_grad for p in epinet.parameters()))
    print("Trainable (base)   params:", sum(p.requires_grad for p in backbone.parameters()))

    # Make sure your optimizer only sees epinet params
    trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    print(f"{len(trainable)} trainable tensors:", trainable[:10], "...")

    # trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    trainer.train()

    # save epinet model
    if training_args.save_model:
        trainer.save_state()  # optional: trainer internals

        # Save state dict + metadata so you can rebuild later
        os.makedirs(training_args.output_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "base_model_name": model_args.model_name_or_path,
                "num_labels": train_dataset.features["label"].num_classes,
                "epinet_cfg": model.cfg.__dict__,   # EpinetConfig fields
                "feature_fn": "NT_feature_fn",      # or "NT_tokens_feature_fn"
            },
            os.path.join(training_args.output_dir, "model_epinet.pt"),
        )
        tokenizer.save_pretrained(training_args.output_dir)




if __name__ == "__main__":
    train()