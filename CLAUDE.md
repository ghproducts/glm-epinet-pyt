# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PyTorch framework for uncertainty-aware genomic sequence classification with genomic language models
(GLMs), built on Hugging Face `transformers`/`datasets`/`Trainer`. It supports:

- fine-tuning a pretrained GLM for sequence classification (`train_base`),
- training an **Epinet** uncertainty head on top of a frozen base model (`train_epinet`),
- fitting a **temperature scaling** factor for calibration (`scaling`),
- running inference with multiple uncertainty-quantification (UQ) methods (`inference`).

Supported backbones: `DNABERT2`, `NT_transformer`, `hyenaDNA`, `CARMANIA`.
Supported UQ methods: `base` (no UQ), `base_scaled` (temperature scaling), `mc_dropout`
(MC dropout), `epinet` (custom Epinet).

## Running the pipeline

All shell scripts live in `scripts/` and must be **run from inside that directory** — each one does
`cd ../` internally so relative paths (data, checkpoints, output dirs) resolve from the repo root.

```bash
cd scripts
bash train_base_model.sh    # 1. fine-tune a base model
bash train_epinet_model.sh  # 2. train Epinet on the frozen base checkpoint
bash get_temp_factor.sh     # 3. fit a temperature-scaling factor
bash test_model.sh          # 4. run inference with a chosen UQ method
```

There is no config file layer above the shell scripts — hyperparameters and paths are edited
directly as shell variables at the top of each script, then passed as `HfArgumentParser` CLI flags
to a `python -m nn_proj.models.<MODEL>.<stage>` invocation. Steps must run in order: base model →
Epinet (needs the base checkpoint) → temp scaling (needs the base checkpoint) → inference (needs
`base`, `epinet`, or temp-scaled checkpoint depending on `UQ_method`).

Key variables to set per script: `DATA` (an `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/<task>`
dataset id, or a local csv/tsv/json/jsonl/parquet path), `CHECKPOINT`/`BASE_CKPT`/`EPI_CKPT`, `MODEL`
(one of the four backbone names, matching the `nn_proj/models/<MODEL>` package), `SEED`, `LR`, and
`MAX_LENGTH`. `MAX_LENGTH` follows a backbone-specific rule of thumb: DNABERT2 uses ~0.25× the
sequence length (its tokenizer is BPE-based), NT_transformer uses ~1/6× (6-mer tokenization),
hyenaDNA/CARMANIA use the full sequence length (character-level tokenization).

For `test_model.sh`, `UQ_method=epinet` requires `BASE_CKPT` to point at the trained **Epinet**
checkpoint (not the plain base checkpoint), and `UQ_method=base_scaled` requires `TEMP` to be set
to the value printed by `get_temp_factor.sh`. Output is written to
`<OUT_PATH>/inference_uncertainty.csv`.

There is no test suite, linter, or build step configured in this repo.

## Architecture

```
nn_proj/
├── common/
│   ├── datasets.py   # load_NT_tasks / load_local_dataset / prep_for_trainer (tokenize + collate)
│   ├── utils.py       # compute_metrics, compute_uncertainty, enable_mc_dropout, etc.
│   ├── blast.py        # standalone BLAST-based nearest-neighbor baseline (not part of the main pipeline)
│   └── mmseqs.py     # standalone MMseqs2-based nearest-neighbor baseline (not part of the main pipeline)
└── models/
    ├── epinet/         # backbone-agnostic Epinet implementation, shared by all four backbones
    │   ├── epinet.py       # EpinetConfig, EpinetWrapper, MLPEpinetWithPrior/MLPEpinetWithConvPrior,
    │   │                   # HFEpinetSeqClassifier (HF Trainer-compatible wrapper), predict()
    │   └── feature_fns.py  # per-backbone (base_model, batch) -> (mu, hidden[, extras]) adapters
    ├── DNABERT2/
    ├── NT_transformer/
    ├── hyenaDNA/
    └── CARMANIA/
```

Each `nn_proj/models/<BACKBONE>/` package follows the same four-stage pattern and is largely
copy-pasted across backbones (mainly differing in the default HF hub id and which `feature_fns.py`
adapter is wired in):

- `config.py` — `ModelArguments`/`DataArguments`/`TrainingArguments` dataclasses (the last extends
  `transformers.TrainingArguments`), parsed via `HfArgumentParser` in every stage script.
- `train_base.py` — loads the pretrained backbone as `AutoModelForSequenceClassification` and
  fine-tunes it with the HF `Trainer`.
- `train_epinet.py` — reloads the fine-tuned base checkpoint, freezes it, wraps it in
  `EpinetWrapper` + `HFEpinetSeqClassifier` via the backbone's feature function, and trains only the
  Epinet head (base params have `requires_grad=False`).
- `scaling.py` — collects base-model logits/labels on a held-out split and fits a scalar
  temperature `T` by minimizing NLL with LBFGS (`fit_temperature`); the printed `T` is pasted
  manually into `test_model.sh`.
- `inference.py` — reloads the chosen checkpoint (plain base, temp-scaled base, or Epinet), applies
  the requested `UQ_method`, and calls `nn_proj.models.epinet.predict()` to write
  `inference_uncertainty.csv`.

**CARMANIA is the odd one out**: its checkpoint isn't a native HF `AutoModelForSequenceClassification`,
so `nn_proj/models/CARMANIA/model.py` defines `CarmaniaForSequenceClassification` — a custom
`PreTrainedModel` that wraps an `AutoModel` encoder with mean-pooling and a linear classification
head — and its `train_base.py`/`inference.py` build this custom class instead of calling
`AutoModelForSequenceClassification.from_pretrained` directly. `nn_proj/models/CARMANIA/test.py` is
a standalone, hardcoded dev/scratch script (not wired into the shared pipeline — don't treat it as
a template for the other stages).

`nn_proj/models/epinet/utils.py` is a stale scratch file (duplicates `compute_uncertainty`/
`compute_metrics` from `nn_proj/common/utils.py`, imports from a non-package-relative `epinet`
module, references an undefined `MODEL_NAME`) — it is not imported anywhere else in the codebase.

### Epinet mechanics (`nn_proj/models/epinet/epinet.py`)

`EpinetWrapper` combines a frozen base model with a small trainable "epinet" head indexed by a
random Gaussian vector `z` (implementing the Epistemic Neural Network idea,
https://arxiv.org/abs/2107.08924): `feature_fn(base_model, batch)` extracts `(mu, hidden[, extras])`
from the base model's forward pass; the epinet maps `(hidden, z)` (optionally plus `extras`, e.g.
raw tokens for a conv prior) to a logit correction that is added to `mu`. Drawing `K` independent
`z` samples (`n_index_samples`) and averaging gives the predictive distribution;
`nn_proj/common/utils.compute_uncertainty` decomposes the resulting `[K, B, C]` logit stack into
total/epistemic/aleatoric entropy, mean confidence, and vote agreement across samples. `mc_dropout`
inference reuses the same `compute_uncertainty` machinery but draws its `K` samples via repeated
stochastic forward passes with dropout enabled at eval time instead of Epinet index sampling.

## Data & output layout (gitignored, regenerated locally)

`trained_models*/`, `inference_results*/`, `logs/`, `mmseqs/`, `temp/`, `temp_scaling_factors_*.tsv`,
`scripts/old/`, and `scripts/full_experiments/` are all gitignored — treat anything under them as
local run artifacts / one-off experiment scripts, not tracked source.
