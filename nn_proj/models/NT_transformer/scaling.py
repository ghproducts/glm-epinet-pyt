# file to grab temp scaling paramsi
import torch
import transformers
import numpy as np
import json
import os
from torch.utils.data import Dataset
from safetensors.torch import load_file

import torch.nn as nn
from torch.utils.data import DataLoader
from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from nn_proj.common.utils import preprocess_logits_for_metrics, compute_metrics
from .config import ModelArguments, DataArguments, TrainingArguments
from nn_proj.models.epinet import EpinetConfig, EpinetWrapper, HFEpinetSeqClassifier, MLPEpinetWithPrior, MLPEpinetWithConvPrior
from nn_proj.models.epinet.feature_fns import hyenaDNA_feature_fn 


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def collect_logits_labels(model, dataset, collate_fn, batch_size: int):
    model.eval().to(DEVICE)

    # Keep only what the model needs (optional but keeps things robust)
    keep = {"input_ids", "attention_mask", "labels", "label"}
    drop = [c for c in dataset.column_names if c not in keep]
    if drop:
        dataset = dataset.remove_columns(drop)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_logits, all_labels = [], []
    for batch in loader:
        labels_key = "labels" if "labels" in batch else "label"
        labels = batch.pop(labels_key).to(DEVICE)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        logits = model(**batch).logits  # [B, C]
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
 
    logits = logits.float()
    labels = labels.long()

    log_T = torch.zeros((), requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T)
        loss = loss_fn(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).detach().cpu().item())

def get_scalilng_params():
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
        train_dataset = load_NT_tasks(task=task, split="train")
    else:
        train_dataset = load_local_dataset(path=data_args.data_path, encode_labels=True, rank=data_args.taxa_rank, taxa_df=data_args.taxa_df)

    train_dataset.filter(lambda ex: ex["labels"] != -1)  # remove unlabelled examples

    split = train_dataset.train_test_split(test_size=0.1, seed=training_args.data_seed, stratify_by_column="labels")
    __, val_dataset = split["train"], split["test"]

    val_dataset, data_collator = prep_for_trainer(val_dataset, tokenizer, metadata_cols=("taxid", "split"),)

    num_labels = config.num_labels if hasattr(config, "num_labels") else num_labels
 
    # buiild model 
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, #model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config, #num_labels=num_labels,
        trust_remote_code=True,
    )

    checkpoint_path = model_args.checkpoint if not model_args.epinet_path else model_args.epinet_path
    state_path = os.path.join(checkpoint_path, "model.safetensors") #model_args.epinet_path, "model.safetensors")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Could not find checkpoint weights at: {state_path}")
    
    sd = load_file(state_path)
    model.load_state_dict(sd, strict=True)

    logits, labels = collect_logits_labels(
        model=model,
        dataset=val_dataset,
        collate_fn=data_collator,
        batch_size=training_args.per_device_eval_batch_size,
    )
    T = fit_temperature(logits, labels)
    print(f"T = {T:.6f}")

    #  os.makedirs(training_args.output_dir, exist_ok=True)
    #  outpath = os.path.join(training_args.output_dir, "temperature.json")
    #  with open(outpath, "w") as f:
    #      json.dump({"temperature": T}, f, indent=2)
    #  print(f"Wrote {outpath}")




if __name__ == "__main__":
    get_scalilng_params()