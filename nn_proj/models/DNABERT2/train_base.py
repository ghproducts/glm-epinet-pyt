
# This file is derived from:
#   MAGICS-LAB/DNABERT_2 (https://github.com/MAGICS-LAB/DNABERT_2)
#   File: finetune/train.py
#   License: Apache License 2.0
#
# Copyright (c) 2023-2025 the original authors
# Modifications copyright (c) 2025 <Your Name / Your Lab>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass, field

import transformers
from transformers import set_seed
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn as nn

from datasets import ClassLabel

from nn_proj.common.utils import preprocess_logits_for_metrics, compute_metrics
from nn_proj.common.datasets import load_local_dataset, load_NT_tasks, prep_for_trainer
from .config import ModelArguments, DataArguments, TrainingArguments
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

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
    if data_args.data_path is None:
        raise ValueError("data_path must be specified.")
    elif data_args.data_path.startswith("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"):
        task = data_args.data_path.split("/")[-1]
        train_dataset = load_NT_tasks(task=task, split="train")
    else:
        train_dataset = load_local_dataset(path=data_args.data_path, encode_labels=True, rank=data_args.taxa_rank, taxa_df=data_args.taxa_df)


    print(len(train_dataset), "training examples loaded.")
    train_dataset.filter(lambda ex: ex["labels"] != -1)  # remove unlabelled examples
    print(len(train_dataset), "training examples after filtering unlabelled.")

    print("number of classes:", train_dataset.features["labels"].num_classes)

    # sklearn data split
    split = train_dataset.train_test_split(test_size=0.1, seed=training_args.data_seed, stratify_by_column="labels")
    train_dataset, val_dataset = split["train"], split["test"]
    
    train_dataset, data_collator = prep_for_trainer(train_dataset, tokenizer, metadata_cols=("taxid", "split"),)
    val_dataset, _   = prep_for_trainer(val_dataset, tokenizer, metadata_cols=("taxid", "split"))

    # get label mappings
    label_feature = train_dataset.features["labels"]
    if isinstance(label_feature, ClassLabel):
        names = label_feature.names  # these are original label values as strings
        id2label = {i: name for i, name in enumerate(names)}
        label2id = {name: i for i, name in enumerate(names)}
        num_labels = label_feature.num_classes
    else:
        raise ValueError("Expected ClassLabel for 'label' feature")

    # load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
        trust_remote_code=True,
        problem_type = "single_label_classification"
    )
    
    # for name, m in model.named_modules():
    #     if isinstance(m, nn.Dropout):
    #         print(name, m.p)
    
    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)




if __name__ == "__main__":
    train()
