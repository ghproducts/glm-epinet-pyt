# uq_metrics_followup

Analysis scripts run against `promoter_alisim`/`aleatoric_boundary` (and,
for `dnabert_only/`, the label-noise/OOD-severity axes) per-example
uncertainty output. Not runnable standalone — point them at your own
inference output.

- `compute_analysis1.py` — total vs. epistemic-alone uncertainty as an
  OOD/ambiguity detector, per method/axis (Spearman rho + Mann-Whitney).
- `compute_analysis2.py` — ECE/ACE/NLL/Brier, with bin-count sensitivity
  (10/15/20/25 bins).
- `dnabert_only/*.py` — same calibration metrics, DNABERT2-only, pooled and
  stratified (per-quartile / per-branch-length) variants.
