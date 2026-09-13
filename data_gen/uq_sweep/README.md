# uq_sweep

One resumable grid over backbone x UQ method x eval dataset, replacing the
per-run hardcoded scripts. Cells whose checkpoints are missing are reported,
never silently substituted.

```bash
python verify_base_accuracy.py --backbones DNABERT2,NT_transformer,hyenaDNA,CARMANIA
python fit_temperatures.py            # required before base_scaled
python run_sweep.py --all --dry-run   # plan + what's blocked
python run_sweep.py --all
```

Checkpoints and generated CSVs are gitignored, so both scripts fall back to the
main checkout when run from a worktree (`$GLM_CKPT_ROOT` overrides).

## Layout

- `sweep_lib.py` — backbone registry, dataset builders, checkpoint/result IO.
- `run_sweep.py` — the UQ methods and the driver.
- `fit_temperatures.py` — fits T per (backbone, seed) into `temperatures.json`.
- `verify_base_accuracy.py` — checks each `base` checkpoint against the
  manuscript's reported ID `promoter_all` error before trusting a sweep.
- `results/<dataset>/<backbone>/<method>__seed<N>.csv` — one row per example.

## Per-backbone quirks encoded here

- hyenaDNA saves `pytorch_model.bin`; the others save `model.safetensors`.
- CARMANIA is not an `AutoModelForSequenceClassification` and is the only
  backbone whose config must load from the **checkpoint** (its `config.json`
  carries `encoder_name`); everything else loads config from the hub id,
  because DNABERT2's remote code fails on a local path.
- Laplace's target head is auto-detected (`find_classifier_attr`): NT's
  `classifier` is a two-layer `EsmClassificationHead`, so the final Linear is
  `classifier.out_proj`, not `classifier`.

## Notes

- Rows carry the full mean predictive distribution (`prob_0`, `prob_1`, ...),
  so ECE/NLL/Brier are computable directly rather than by inverting entropy.
- `base_scaled`, `laplace`, and `conformal` fit on the held-out 10% of the
  training split, never on the evaluation set.
- `conv_epinet` refuses the pre-z-fix `epinet` checkpoints; it wants
  `epinet_zfix` (trained under the per-example-z fix).
- `evidential`'s `U_epistemic` is 0 by construction (single Dirichlet); read
  the `vacuity` column instead.
