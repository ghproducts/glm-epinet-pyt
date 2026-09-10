# Epinet retrain (`epinet_zfix`) + AliSim grid extension: consolidated summary

Fixes the most serious issue identified in `../critique_independent.md`
(Finding 1.1): every `conv_epinet` result produced this session used
`checkpoints/seed_1/DNABERT2/label_noise_r00/epinet`, a checkpoint trained
**before** the epinet per-example-`z` fix (`fb38241`) existed. The fix was
only ever applied as an eval-time monkeypatch (swap in the fixed
`epinet.py`, then run inference) — never by retraining the epinet head
itself under the corrected sampling code. Since training uses `k_train=8`
index samples per step, every training step under the old code drew 8
*batch-shared* `z` vectors, not 8×B independent per-example draws: a real
train/inference mismatch, not a provenance technicality.

This directory holds the cross-dataset comparison artifacts; the full
per-dataset methodology and tables live in
`data_gen/promoter_alisim/README.md` ("Retrain + grid extension follow-up")
and `data_gen/aleatoric_boundary/README.md` ("Retrain follow-up: conv_epinet
re-run under epinet_zfix").

## What was done

1. **Retrained `conv_epinet`'s checkpoint** — `nn_proj/models/epinet/epinet.py`
   was already fixed on this branch (merged from `worktree-fix-epinet-batch-z`).
   Ran `nn_proj/models/DNABERT2/train_epinet.py` from scratch against the
   frozen `checkpoints/seed_1/DNABERT2/label_noise_r00/base` checkpoint, same
   data/hyperparameters/seed as the original (`configs/experiments.yaml`'s
   `epinet:` block: LR 2e-5, 2 epochs, `k_train=8`, `index_dim=30`,
   `hidden_sizes=[50]`, `prior_scale=1.0`, `conv_prior_scale=1.0`, seed 1).
   New checkpoint: `checkpoints/seed_1/DNABERT2/label_noise_r00/epinet_zfix/`
   (original `epinet/` kept for provenance). Converged normally:
   `eval_accuracy=0.887` at epoch 2, matching the original's ~87-88%.
2. **Extended the AliSim branch-length grid** 3 more points past the dense
   grid's previous max (3.0): **5.0, 8.0, 12.0**, same 400 anchors/model/seed.
   `data_gen/promoter_alisim/csv_data/promoter_alisim_extended.csv` (1200
   rows). Mean identity to anchor: 0.340 / 0.301 / 0.274 — still descending
   but **not yet at the ~0.25 theoretical floor even at t=12.0**.
3. **Re-ran `conv_epinet`** (full 19-point re-run, new checkpoint) and the
   other 8 methods (3 new points only) + `base`/`base_scaled` (19 points) on
   `promoter_alisim`, and **re-ran `conv_epinet`** (checkpoint swap only, no
   new data) on `aleatoric_boundary`.
4. **Recomputed** pooled/stratified ECE/NLL/Brier for all 11 methods on both
   datasets, decomposition tables, Spearman over the full 19-point grid, and
   — the critique's Finding 3.3 requirement — peak location +
   peak-to-baseline/peak-to-endpoint magnitude for **all 9 decomposable
   methods**, not just the 3 originally evaluated at this resolution.

## (a) Did retraining change `conv_epinet`'s qualitative conclusions?

**Mostly no — with one instructive, reportable exception once the grid is
extended far enough.**

- **AliSim, apples-to-apples 16-point window**: shape classification
  ("still rising, no interior peak") is IDENTICAL between old and new
  checkpoints. Spearman rho is similar order (0.122 old vs. an implied
  ~0.15 for the new checkpoint on the same window). Absolute `U_epistemic`
  magnitude is ~3x SMALLER under the correctly-trained checkpoint
  (0.0644→0.0202 at t=0, 0.0920→0.0317 at t=3.0) — plausibly because the
  stale checkpoint's batch-shared-z training produced a head whose
  eval-time per-example outputs happen to spread out more, an artifact of
  the train/inference mismatch rather than a more genuine signal.
- **Boundary dataset**: pattern unchanged (`U_epistemic` rises sharply and
  significantly near the decision boundary for both checkpoints, p≈1e-130
  both times); Q1/Q4 ratio is actually LARGER for the new checkpoint (26.8x
  vs. 17.8x) even as absolute magnitude shrinks (~2.4x at Q1).
- **The one place retraining's effect is entangled with a real finding**:
  once the AliSim grid is extended to t=5.0-12.0, the retrained
  checkpoint's `U_epistemic` PEAKS at t=3.0 and DECLINES 10.3% by t=12.0 —
  changing its shape classification from "still rising" to "declines after
  peaking," now matching `mc_dropout`/`evidential`/both ensembles instead of
  standing out as an exception. This is attributable to the GRID EXTENSION,
  not the retrain itself (see (c) below) — but it does mean the retrained
  checkpoint's full picture only becomes visible with the extended axis.
- **Net effect on the paper's claims**: none of the roster-wide claims
  ("no method cleanly separates aleatoric/epistemic," "RF is the partial
  exception") depended on `conv_epinet`'s exact numbers and are unaffected.
  The epinet-specific claim from the original critique ("the fix was
  validated") is now actually true rather than aspirational — this
  retraining was necessary methodology, not optional polish — but it does
  not change the qualitative story the paper already told.

Full side-by-side: `conv_epinet_old_vs_new_alisim.csv`,
`conv_epinet_old_vs_new_boundary.csv` (this directory).

## (b) Full new 19-point dose-response + shape data, all 11 methods

See `data_gen/promoter_alisim/uncertainty_eval/dense_grid_extended_zfix/`
for the complete machine-readable tables
(`per_example_uncertainty_19pt.csv`, `results_summary_dose_response.csv`,
`results_summary_spearman.csv`, `results_summary_mwu_endpoints.csv`,
`results_summary_shape.csv`, `results_summary.md`) and
`data_gen/uq_metrics_followup/dnabert_only/{pooled,stratified}_ece_allmethods_extended_zfix.csv`
for calibration. Shape summary (9 decomposable methods; `base`/`base_scaled`
have no epistemic/aleatoric decomposition, so shape characterization does
not apply to them by construction — their calibration is in the ECE tables
instead):

| method | shape (19pt grid) | peak branch_length | % rise to peak | % change peak→12.0 |
|---|---|---:|---:|---:|
| conv_epinet | declines after peaking | 3.0 | +57.1% | -10.3% |
| mc_dropout | declines after peaking | 0.2 | +57.7% | -41.3% |
| evidential | declines after peaking | 0.4 | +13.0% | -22.6% |
| laplace | declines after peaking | 3.0 | +40.1% | -5.1% |
| ensemble_k5 | declines after peaking | 0.6 | +101.3% | -35.4% |
| ensemble_k3 | declines after peaking | 0.4 | +73.7% | -31.6% |
| cnn_mc_dropout | declines after peaking | 5.0 | +118.5% | -8.1% |
| cnn_ensemble | declines after peaking | 1.0 | +83.5% | -6.7% |
| rf_kmer | plateaus near peak (<3%) | 5.0 | +89.8% | -2.7% |

Spearman rho (19-point grid, `U_epistemic`): rf_kmer 0.595, cnn_mc_dropout
0.279, cnn_ensemble 0.226, ensemble_k5 0.193, conv_epinet 0.168, laplace
0.151, ensemble_k3 0.143, evidential 0.036, mc_dropout 0.029. Note how
poorly rho alone tracks the actual dose-response magnitude for several
methods (e.g. `cnn_mc_dropout` rises 118.5% to its peak yet has a
mid-roster rho of 0.28; `mc_dropout` rises 57.7% yet rho≈0.03) — this is
exactly the Finding-3.3 pathology the critique flagged, now quantified for
every method via the shape table rather than just the 1 method (mc_dropout)
originally spot-checked.

## (c) Did the extended grid reach real saturation?

**No — not fully, though it gets much closer.** Mean identity to anchor:
3.0→0.392, 5.0→0.340, 8.0→0.301, 12.0→0.274, vs. a ~0.25 theoretical floor.
Even at 4x the previous max branch length, the axis has not fully
decorrelated from the anchor.

**What this does to the "epistemic barely moves" framing**: it survives,
but with an important addition. Every DNABERT2/CNN-based method's
`U_epistemic` now demonstrably PEAKS and DECLINES once evaluated far enough
(not "still rising indefinitely" as the 16-point grid suggested for
`conv_epinet`/`laplace`) — meaning the signal these methods produce is
better described as "responds non-monotonically to divergence, peaking in
a mid-range and reverting toward baseline at extreme divergence" than
"monotonically tracks novelty." That reversion toward baseline at high
divergence, on inputs objectively MORE divergent than before, is if
anything a STRONGER piece of evidence that these methods' epistemic
signal does not track genuine distributional novelty in a reliable way —
rather than the axis simply not having gone far enough. The one exception,
`rf_kmer`, is the only method whose signal keeps climbing to and then
plateaus (rather than reverting) near this now-more-divergent endpoint —
consistent with the existing "RF is the partial exception" framing, now
demonstrated up to t=12.0 rather than assumed to hold indefinitely past
t=3.0.

## (d) Where everything lives

- New checkpoint: `checkpoints/seed_1/DNABERT2/label_noise_r00/epinet_zfix/`
  (gitignored local artifact, not committed — original `epinet/` untouched).
- Extended AliSim data: `data_gen/promoter_alisim/csv_data/promoter_alisim_extended.csv`
  (+ `_params.json`), `data_gen/promoter_alisim/make_alisim_data_extended.py`.
- Full 19-point eval: `data_gen/promoter_alisim/uncertainty_eval/dense_grid_extended_zfix/`
  (scripts + all result CSVs/markdown; original `dense_grid/` untouched for
  provenance).
- Boundary zfix eval: `data_gen/aleatoric_boundary/run_conv_epinet_zfix.py`,
  `data_gen/aleatoric_boundary/csv_data/conv_epinet_zfix.csv` and
  `uncertainty_by_method_zfix.csv` (original `uncertainty_by_method.csv`
  untouched).
- Updated ECE/NLL/Brier: `data_gen/uq_metrics_followup/dnabert_only/recompute_ece_extended_zfix.py`,
  `pooled_ece_allmethods_extended_zfix.csv`, `stratified_ece_allmethods_extended_zfix.csv`
  (originals `pooled_ece_allmethods.csv`/`stratified_ece_allmethods.csv` untouched).
- Old-vs-new comparison: this directory's
  `conv_epinet_old_vs_new_alisim.csv` / `conv_epinet_old_vs_new_boundary.csv`.
- README updates explaining all of the above in context: both
  `data_gen/promoter_alisim/README.md` and
  `data_gen/aleatoric_boundary/README.md` gained a "retrain follow-up"
  section each.
- `train_epinet.py` bugfix (unconditional `trainer.save_model()` call +
  `"label"`→`"labels"` typo fix needed to actually produce a usable
  checkpoint): `nn_proj/models/DNABERT2/train_epinet.py`.

## (e) Honest framing

This retrain was necessary methodology (the prior checkpoint's provenance
was genuinely mismatched to the training procedure it was claimed to
validate), and it DID change the practical numbers (roughly 2.5-3x smaller
absolute `U_epistemic` magnitudes across both datasets) — but it did NOT
overturn any qualitative conclusion in the existing tables. If anything, the
correctly-trained checkpoint now conforms MORE closely to the rest of the
roster's pattern (rise-then-decline once evaluated far enough) rather than
standing out as an exception, which slightly strengthens rather than
weakens the paper's roster-wide claim. This is itself a useful, reportable
result: the epinet bug was real and worth fixing on principle, but it was
not responsible for the (much larger) finding that this project's UQ
methods generally fail to cleanly separate epistemic from aleatoric
uncertainty.
