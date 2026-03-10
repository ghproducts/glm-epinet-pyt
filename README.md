# glm-epinet-pyt

PyTorch framework for uncertainty-aware genomic sequence classification with genomic language models (GLMs), and is primarily designed to work with huggingface models and datasets. This framework for epinet can also be reused for other projects if needed.

This project supports:

- fine-tuning a pretrained genomic language model for classification,
- training an **epinet** uncertainty head on top of a frozen base model,
- fitting a **temperature scaling** factor for calibration,
- running inference with multiple uncertainty methods.

Supported backbones:

- `DNABERT2`
- `NT_transformer`
- `hyenaDNA`
- `CARMANIA`

Supported uncertainty methods:

- `base`: no additional uncertainty
- `base_scaled`: temp scaling based calibration
- `mc_dropout`: monte-carlo dropout on all dropout layers.
- `epinet`: custom epinet implementation of epistemic neural network

## Repository layout

```text
.
├── nn_proj/
│   ├── common/
│   └── models/
│       ├── CARMANIA/
│       ├── DNABERT2/
│       ├── NT_transformer/
│       ├── hyenaDNA/
│       └── epinet/
└── scripts/
```

Main folders:

- `nn_proj/common/`: shared utilities
- `nn_proj/models/<MODEL>/`: model-specific `train_base.py`, `train_epinet.py`, `scaling.py`, and `inference.py`
- `nn_proj/models/epinet/`: shared Epinet implementation
- `scripts/`: shell scripts for the main workflow

## Running

Run all shell scripts from inside the `scripts/` directory. 

### 1. Train a base model

Edit `train_base_model.sh` and set:

- `DATA`
- `CHECKPOINT`
- `MODEL`
- `SEED`
- `LR`
- `MAX_LENGTH`

Then run:

```bash
bash train_base_model.sh
```

### 2. Train Epinet

Edit `train_epinet_model.sh` and set:

- `DATA`
- `BASE_CKPT`
- `EPI_CKPT`
- `MODEL`
- `SEED`
- `LR`
- `MAX_LENGTH`

Then run:

```bash
bash train_epinet_model.sh
```

### 3. Fit temperature scaling

Edit `get_temp_factor.sh` and set:

- `DATA`
- `BASE_CKPT`
- `MODEL`
- `MAX_LENGTH`
- `SEED`

Then run:

```bash
bash get_temp_factor.sh
```

Copy the printed temperature value into `test_model.sh` when using `UQ_method="base_scaled"`.

### 4. Run inference

Edit `test_model.sh` and set:

- `DATA`
- `NUM_LABELS`
- `BASE_CKPT`
- `OUT_PATH`
- `MODEL`
- `MAX_LENGTH`
- `SEED`
- `UQ_method`
- `TEMP`
- `K_SAMPLES`

Then run:

```bash
bash test_model.sh
```

Results are written to:

```text
<OUT_PATH>/inference_uncertainty.csv
```

## Choosing `UQ_method`

Use one of these in `test_model.sh`:

- `base`: standard prediction
- `base_scaled`: temperature-scaled base prediction
- `mc_dropout`: dropout-based uncertainty with repeated forward passes
- `epinet`: Epinet-based uncertainty

For `epinet`, make sure `BASE_CKPT` points to the trained **Epinet checkpoint**, not the original base checkpoint.

## Default example

The provided scripts use:

```bash
DATA="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/promoter_all"
MODEL="DNABERT2"
```

Typical run order:

```bash
cd scripts
bash train_base_model.sh
bash train_epinet_model.sh
bash get_temp_factor.sh
bash test_model.sh
```

## Notes

- Train the base model first.
- Train Epinet second using the saved base checkpoint.
- Fit temperature scaling after base training.
- Run all scripts from the `scripts/` directory.
- This project has only been tested with Python 3.11 and Nvidia a100 GPUs. Your configuration may vary.


