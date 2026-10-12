# datasets.py
# Utilities to load NT tasks or local datasets, tokenize, and build a DNABERT-friendly collator.

import os
import pandas as pd

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Iterable

from datasets import load_dataset, ClassLabel, Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding

import torch



# taxID rank labeling function:
def taxid_rank_labeling(taxid: int, rank: str, lineage_df: pd.DataFrame) -> int:
    species_to_target = dict(zip(lineage_df["species"].astype("Int64"), lineage_df[rank].astype("Int64")))
    
    return species_to_target.get(taxid, -1)
    

"""
Functions to load datasets for local or huggingface tasks.

"""
#
# 
#  HF NT downstream tasks
def load_NT_tasks(task: str, split: str, encode_labels: bool = True) -> Dataset:
    """
    Load a specific downstream task from the InstaDeep NT tasks dataset.
    Ensures columns: 'sequence', 'labels' (int-encoded).
    """
    #dataset_name = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
    ds_all = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", split=split)

    if "task" not in ds_all.column_names:
        raise ValueError(f"Expected a 'task' column; found: {ds_all.column_names}")

    ds = ds_all.filter(lambda ex: ex["task"] == task)

    cols = set(ds.column_names)
    if "sequence" not in cols:
        raise ValueError(f"{task}: expected 'sequence' column, found: {sorted(cols)}")

    if "labels" not in cols:
        if "label" in cols:
            ds = ds.rename_column("label", "labels")
        else:
            raise ValueError(f"{task}: expected a 'label'/'labels' column, found: {sorted(cols)}")

    # Ensure integer-encoded labels
    if encode_labels and not isinstance(ds.features["labels"], ClassLabel):
        ds = ds.class_encode_column("labels")

    return ds


# Local single-file datasets (csv/tsv/json/jsonl/parquet)
def load_local_dataset(path: str, encode_labels: bool = True, rank=None, taxa_df=None,
                        drop_unmapped: bool = True) -> Dataset:
    """
    Load a single local file (CSV/TSV/JSON/JSONL/Parquet) as a Dataset (single split named 'train').
    Expects columns: 'sequence', 'label' or 'labels'.

    When ``rank`` remaps a taxid through ``taxa_df``, any row whose taxid is
    absent from the lineage table gets label -1. ``drop_unmapped=True``
    (default) removes those rows *before* label encoding, so they can never
    become a trainable "unknown" class. Set it to False only to inspect
    what would be dropped; passing unmapped rows on to
    ``class_encode_column`` will legitimize -1 as a real class.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find dataset at: {path}")

    ext = Path(path).suffix.lower()
    if ext in (".csv", ".tsv"):
        fmt = "csv"
    elif ext in (".json", ".jsonl"):
        fmt = "json"
    elif ext == ".parquet":
        fmt = "parquet"
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    ds = load_dataset(fmt, data_files=path, split="train")

    cols = set(ds.column_names)
    if "sequence" not in cols:
        raise ValueError(f"Expected 'sequence' column in {path}, found: {sorted(cols)}")

    if "labels" not in cols:
        if "label" in cols:
            ds = ds.rename_column("label", "labels")
        else:
            raise ValueError(f"Expected a 'label'/'labels' column in {path}, found: {sorted(cols)}")
        
    if rank in ["species", "genus", "family", "order", "class", "phylum", "kingdom"]:
        if "taxid" not in cols:
            raise ValueError(f"taxa_rank labeling requested but no 'taxid' column found in {path}")
        
        lineage_df = pd.read_csv(taxa_df, index_col=0)

        species_to_target = dict(zip(lineage_df["species"].astype("Int64"),
                                     lineage_df[rank].astype("Int64")))
        ds = ds.map(lambda ex: {"labels": int(species_to_target.get(ex["taxid"], -1))})

        if drop_unmapped:
            n_before = len(ds)
            ds = ds.filter(lambda ex: ex["labels"] != -1)
            n_dropped = n_before - len(ds)
            if n_dropped:
                print(f"[load_local_dataset] dropped {n_dropped}/{n_before} rows with no "
                      f"'{rank}' in {taxa_df} (unmapped taxid -> label -1)")

    if encode_labels and not isinstance(ds.features["labels"], ClassLabel):
        ds = ds.class_encode_column("labels")

    return ds


# prepare dataset for huggingface Trainer
def prep_for_trainer(ds: Dataset, 
                     tokenizer: PreTrainedTokenizerBase, 
                     max_length:Optional[int] = None,
                     metadata_cols: Optional[Iterable[str]] = None) -> Tuple[Dataset, DataCollatorWithPadding]:
    """
    Tokenize sequences and format dataset for Huggingface Trainer.
    Assumes 'sequence' and 'labels' columns exist.
    """

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["sequence"],
            truncation=True,
            padding=False,
            max_length=max_length if max_length is not None else tokenizer.model_max_length,
        )
        tokenized["labels"] = examples["labels"]
        return tokenized

    cols_to_keep = {"sequence", "labels"}
    if metadata_cols is not None:
        cols_to_keep.update(metadata_cols)

    cols_to_drop = [c for c in ds.column_names if c not in cols_to_keep]
    ds_tokenized = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_drop)
    ds_tokenized.set_format(type="torch")

    return ds_tokenized, DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)




