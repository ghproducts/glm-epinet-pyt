# aleatoric_boundary

Aleatoric-shift eval set: real `promoter_all` test sequences, no synthetic
label noise. Ambiguity is measured, not injected — margin score
`m = |2p_1 - 1|` from the base model's own softmax, then split into
quartiles (`pd.qcut`, rank-based to avoid duplicate-edge errors). Q1 =
near the decision boundary, Q4 = confident.

## Scripts

- `make_boundary_eval.py` — computes margin scores, writes
  `csv_data/margin_scores.csv`, then runs the base/mc_dropout/conv_epinet/
  evidential methods over the fixed row order.
- `make_boundary_eval_extra_methods.py` — same fixed row order, remaining
  methods (ensembles, CNN variants, RF, laplace).
- `run_conv_epinet_zfix.py` — re-runs `conv_epinet` under a corrected
  checkpoint; writes `csv_data/uncertainty_by_method_zfix.csv`.

## Output columns

`margin_scores.csv`: `name, label, margin, quartile`.
