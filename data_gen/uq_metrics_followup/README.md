# UQ metrics follow-up: total-vs-epistemic detection power, and ECE done properly

Two analyses requested as a follow-up to this session's four uncertainty-test
axes (label-noise rate, OOD severity, margin-boundary quartile, AliSim
branch-length). Both are pure aggregation/recomputation on already-generated
per-example or per-seed data — **no new model inference was run** for
Analysis 1, and Analysis 2 uses only the per-example probability data that
already existed (see "Data availability audit" below for exactly what did
and didn't survive).

## Why a new directory, not an extension of the existing ones

`data_gen/aleatoric_boundary/` and `data_gen/promoter_alisim/` each belong to
one axis; `data_gen/label_noise/` covers two (rate, OOD severity) but has no
single results-summary file that spans methods the way this follow-up needs
to. Since both analyses here are explicitly cross-axis (every method, every
axis, in one consolidated table each), a new directory keeps that
cross-cutting view in one place instead of splitting it three ways. Judgment
call, not a hard rule.

## Layout

```
uq_metrics_followup/
├── inputs/                              # copies of the exact source CSVs used (see audit below)
│   ├── label_noise/                     # from data_gen/label_noise/ (main checkout; untracked there)
│   ├── aleatoric_boundary/              # from branch worktree-fix-epinet-batch-z (git show)
│   └── promoter_alisim/{dense,sparse}_grid/
├── compute_analysis1.py                 # total-vs-epistemic, all 4 axes
├── compute_analysis2.py                 # ECE/ACE/NLL/Brier
├── analysis1_total_vs_epistemic.csv     # 191 rows: axis x method x score x test
├── analysis2_ece_ace_nll_brier.csv      # 32 rows: dataset x method x condition x bin_count
├── analysis2_bin_sensitivity.csv        # pivoted view: ECE at each bin count + range
└── results_summary.md                   # full write-up, both analyses
```

Re-run either script directly (`python3 compute_analysisN.py` from this
directory) — both are pure-Python/pandas/scipy over the `inputs/` CSVs, no
GPU, checkpoints, or special venv required, unlike the scripts that
originally produced those CSVs.

## Data availability audit (why some cells are gaps, not numbers)

**Analysis 1 (total vs. epistemic)** needs only already-computed
`U_epistemic`/`U_aleatoric` (or their sum, `U_total`, exact by the BALD
identity) — every method on every axis has at least this much, so Analysis 1
has no true gaps, only different *granularities*:

| axis | granularity | consequence |
|---|---|---|
| margin-boundary quartile | per-example (n=1584/method) | full Spearman + Mann-Whitney(Q1 vs Q4) on `U_total`, computed directly, same footing as the pre-existing epistemic/aleatoric numbers |
| AliSim branch-length | per-example (n=6400/method, dense grid) | same — full Spearman + Mann-Whitney(t=3.0 vs t=0.0) on `U_total` |
| OOD severity | per-example for 7 of 12 method-configs (base, base_scaled, mc_dropout, conv_epinet, laplace, evidential, conformal — MWU already computed this session, reused from `.docs/ALL_METHODS_OOD_SEVERITY.md`, not recomputed); **aggregate-only** (mean/std, n=1584, no raw array) for the other 5 (ensemble_k5, ensemble_k3, cnn_mc_dropout, cnn_ensemble, rf_kmer) | `U_total_mean` is still exact (sum of the two saved means), but no Mann-Whitney p-value is possible for `U_total` on those 5 without re-deriving per-example values — flagged explicitly in the output table rather than fabricated or silently omitted |
| label-noise rate | **aggregate-only everywhere** (mean per rate, per rate x seed at best — 3 seeds for conv_epinet/mc_dropout/laplace, 1 for everything else) | Spearman is computed across the available rate[-x-seed] points (n=15 for the 3-seed methods, n=5 for the rest) — a real, much lower-powered correlation than the per-example axes, reported with its true n throughout, not disguised as a per-example statistic |

**Analysis 2 (ECE/ACE/NLL/Brier)** needs the actual per-example predicted
probability, not just the derived `U_*` scores — and this turns out to be
the real casualty of how this session's scripts were written. Every UQ
script computed a probability internally (`compute_uncertainty`'s
`max_confidence`/`mean_probs`) but **almost none of them saved it** to the
per-example CSV — only the already-decomposed uncertainty scores were kept.
Grepping every CSV under `data_gen/{label_noise,aleatoric_boundary,promoter_alisim}/`
for a probability/confidence column found exactly two survivors:

| file | axis | method covered | column |
|---|---|---|---|
| `aleatoric_boundary/csv_data/margin_scores.csv` | margin-boundary | `base` only (the margin score's own source model) | `prob_class1` |
| `label_noise/evidential_ood_retest.csv` | OOD severity | `evidential` only | `max_confidence` |

That's **2 of the ~30 method x axis cells** this follow-up covers in
Analysis 1. Every other method/axis combination — every method on the
label-noise-rate and AliSim axes, and 6 of 7 methods on the boundary and
OOD-severity axes — has no persisted probability, so ECE/ACE/NLL/Brier are
**not computable from existing data** for those cells. This is reported as a
gap in `results_summary.md`, not filled in by rerunning inference: the task
explicitly excludes new inference for this unless it is cheap and the
environment is already documented and ready. It is documented and ready
(`aleatoric_boundary_venv`, checkpoints, and a live GPU are all present on
this host) but instrumenting six more scripts to persist `max_confidence`
and rerunning them is a real, non-trivial piece of new work, not a rerun of
something already built for this — so it was left as a documented gap
rather than attempted partway.

A further consequence: **neither ECE-eligible dataset has more than one
seed**. Multi-seed ECE (mean +/- std) — the specific fix `FINDINGS.md` #1
calls for — cannot be produced from any existing data in this project; the
axes that do have 3-seed replication (label-noise rate, for conv_epinet/
mc_dropout/laplace) never saved a probability column. Reported as-is.

## Headline results

See `results_summary.md` for the full write-up. In short:

- **Analysis 1**: on the label-noise-rate axis, `U_total` tracks noise rate
  almost perfectly for every one of 9 methods (Spearman rho 0.75-1.0), while
  `U_epistemic` alone is *anti-correlated* with rate for 5 of 9 (rho -0.91 to
  -1.0) — using epistemic alone would point the wrong way, not just a weaker
  way. On the margin-boundary axis, `U_total` beats `U_epistemic` alone for
  all 9 methods (sometimes by a wide margin, e.g. evidential -0.910 vs
  -0.056). The AliSim branch-length axis is the one genuine exception:
  `U_epistemic` alone beats `U_total` for `rf_kmer` (0.563 vs 0.359) and
  `ensemble_k5` (0.191 vs 0.145), and for `mc_dropout`, `U_total` is
  *negatively* correlated with branch length (rho=-0.070, wrong sign) while
  `U_epistemic` is weakly positive (rho=+0.035, right sign). The
  OOD-severity axis reproduces the task's own precedent case exactly
  (`mc_dropout`, mild `shuffled` shift: `U_total` p=0.355 n.s. vs
  `U_epistemic` p=4.2e-6) but that pattern does not generalize to the other
  two methods with a real epistemic component on that axis (`conv_epinet`,
  `laplace`), where total and epistemic are both significant at the mild
  shift.
- **Analysis 2**: bin-count sensitivity (10 vs 15 vs 20 vs 25 equal-width
  bins) on this data is real but modest — up to ~1.5 percentage points of
  ECE, non-monotonic in bin count, smaller than the ~5pp single-seed-vs-mean
  gap `FINDINGS.md` #1 already found. ACE (equal-mass bins) tracks standard
  ECE closely in every case tested (differences under 1.5pp), i.e. this
  particular data does not show ACE and standard ECE disagreeing sharply.
  The R2-3 pathology — ECE looking "fine" on a badly-performing slice — is
  directly reproduced on the margin-boundary Q1 slice: 67.4% accuracy, but
  ECE of only 5.3-6.6% (the kind of number that reads as "well calibrated"
  in isolation); NLL (0.62) and Brier (0.214) make the same slice look much
  worse than Q4's (accuracy 99.2%, ECE 0.2%, NLL 0.044, Brier 0.0075),
  exactly the R2-3-recommended fix of pairing ECE with a proper scoring
  rule. On the OOD-severity axis, by contrast, `evidential`'s ECE rises
  in step with its accuracy collapse (real 0.87/ECE 5% -> random_dna
  0.49/ECE 36%) — no masking there; the pathology is slice-dependent, not
  universal.
