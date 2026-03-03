from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import transformers
import numpy as np
from torch.utils.data import Dataset

"""
General configuration dataclasses for DNABERT2 training.
"""

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="LongSafari/hyenadna-small-32k-seqlen-hf")
    checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to model checkpoint."})
    epinet_path: Optional[str] = field(default=None, metadata={"help": "Path to epinet model."})
    num_samples: int = field(default=1, metadata={"help": "Number of samples for MC dropout or epinet."})
    uncertainty_method: str = field(default=None, metadata={"help": "Uncertainty method to use: epinet or mc_dropout."})
    temperature: float = field(default=1.0, metadata={"help": "Temperature scaling parameter."})

@dataclass
class DataArguments:
    data_path: str
    test_path: Optional[str] = field(default=None, metadata={"help": "Path to test dataset."})
    num_labels: Optional[int] = field(default=None, metadata={"help": "Number of labels for classification."})
    taxa_rank: Optional[str] = field(default=None, metadata={"help": "Taxonomic rank for labeling (e.g., 'genus', 'family')."})
    taxa_df: Optional[str] = field(default=None, metadata={"help": "Path to taxa lineage dataframe CSV file."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_strategy: str = field(default="epoch")
    eval_strategy: str = field(default="epoch")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    save_safetensors: bool = field(default=False)



