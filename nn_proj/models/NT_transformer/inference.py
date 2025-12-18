import argparse
import os
import torch
import transformers
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithPrior, MLPEpinetWithConvPrior, predict
from nn_proj.models.epinet.feature_fns import NT_feature_fn, NT_first_last_feature_fn
from .config import ModelArguments, DataArguments, TrainingArguments    

import torch.nn as nn


# inference methods for NT model
# can use either epinet, MC-dropout, or base model only

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def enable_mc_dropout(model: nn.Module, p: float = 0.1) -> None:
    """Enable MC dropout exactly like base training: same modules, same p."""
    model.train()  # same mode as during training
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.p = p

def evaluate():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = transformers.AutoConfig.from_pretrained(
        model_args.checkpoint,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
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


    # load datasets
    if data_args.data_path is None:
        raise ValueError("data_path must be specified.")
    elif data_args.data_path.startswith("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
        task = data_args.data_path.split("/")[-1]
        task_dataset = load_NT_tasks(task=task, split="test", encode_labels=False)
    else:
        task_dataset = load_local_dataset(path=data_args.data_path, encode_labels=False)

    if False:
        label2id = {k: int(v) for k, v in config.label2id.items()}
        task_dataset= task_dataset.map(
            lambda batch: {"labels": [label2id[str(t)] for t in batch["labels"]]},
            batched=True,
        )

    num_labels = data_args.num_labels if data_args.num_labels is not None else task_dataset.features["label"].num_classes
    task_dataset, data_collator = prep_for_trainer(task_dataset, tokenizer)

    num_labels = config.num_labels if hasattr(config, "num_labels") else num_labels

    #  build model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config, #num_labels=num_labels,
        trust_remote_code=True,
    )

    if model_args.uncertainty_method == "epinet":
        #epi_cfg = EpinetConfig(num_classes=num_labels)
        #wrapper = EpinetWrapper(model, NT_first_last_feature_fn, epi_cfg, epinet=MLPEpinetWithPrior)
        epi_cfg = EpinetConfig(num_classes=num_labels, include_inputs=True, vocab_size=config.vocab_size)
        wrapper = EpinetWrapper(model, NT_feature_fn, epi_cfg, epinet=MLPEpinetWithConvPrior)
        model = HFEpinetSeqClassifier(wrapper, k_train=8, k_eval=8).to(DEVICE)
        with torch.no_grad():
            dummy = tokenizer("ACGT", truncation=True, padding="max_length", max_length=training_args.model_max_length, return_tensors="pt")
            dummy = {k: v.to(DEVICE) for k, v in dummy.items()}
            _ = model(**dummy)  # builds epinet internals
    elif model_args.uncertainty_method == "mc_dropout":
        enable_mc_dropout(model, p=0.1)
        model.to(DEVICE)
    else:
        model.to(DEVICE) 

    # load weights
    checkpoint_path = model_args.checkpoint if not model_args.epinet_path else model_args.epinet_path
    state_path = os.path.join(checkpoint_path, "model.safetensors") #model_args.epinet_path, "model.safetensors")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Could not find checkpoint weights at: {state_path}")
    
    sd = load_file(state_path)
    model.load_state_dict(sd, strict=True)
    outfile = os.path.join(training_args.output_dir, "inference_uncertainty.csv")

    # perform inference
    predict(
        model=model,
        dataset=task_dataset,
        collator=data_collator,
        k_samples=model_args.num_samples,
        batch_size=training_args.per_device_eval_batch_size,
        uncertainty_method=model_args.uncertainty_method,
        outfile=outfile,
    )


    if True:
        # relabel preds with original string labels
        df = pd.read_csv(outfile)
        id2label = {v: k for k, v in config.label2id.items()}
        df["pred"] = df["pred"].map(id2label)
        rename_map = {}
        for i in range(len(id2label)):
            old_name = f"prob_{i}"
            if old_name in df.columns:      
                new_name = f"prob_{id2label[i]}"   
                rename_map[old_name] = new_name
        df = df.rename(columns=rename_map)

        df.to_csv(outfile, index=False)

if __name__ == "__main__":
    evaluate()