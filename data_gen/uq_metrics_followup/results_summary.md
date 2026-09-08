# UQ metrics follow-up: results

Full data: `analysis1_total_vs_epistemic.csv` (191 rows), `analysis2_ece_ace_nll_brier.csv`
(32 rows), `analysis2_bin_sensitivity.csv`. This file is the narrative read of both.

---

## Analysis 1: does using epistemic-alone add detection power over total uncertainty?

`U_total = U_aleatoric + U_epistemic` exactly, per-example, by construction of the
BALD decomposition. Every number below is pure aggregation on already-computed
per-example (boundary, AliSim) or per-seed/per-rate (label-noise, OOD) data — no
new model inference.

### Axis 1: margin-boundary quartile (DNABERT2, `promoter_all`, per-example, n=1584)

Spearman rho vs. continuous margin score (all negative = uncertainty falls as
margin/confidence rises, as expected; magnitude is what matters here):

| method | U_total | U_epistemic (or analogue) | U_aleatoric | total vs. epistemic |
|---|---:|---:|---:|---|
| conv_epinet | **-0.948** | -0.838 | -0.952 | total wins |
| mc_dropout | **-0.879** | -0.715 | -0.886 | total wins |
| laplace | **-0.956** | -0.488 | -0.973 | total wins big |
| ensemble_k5 | **-0.986** | -0.914 | -0.986 | total wins |
| ensemble_k3 | **-0.985** | -0.858 | -0.986 | total wins |
| cnn_mc_dropout | **-0.660** | -0.632 | -0.661 | total wins (small margin) |
| cnn_ensemble | **-0.782** | -0.747 | -0.775 | total wins (small margin) |
| rf_kmer | **-0.775** | -0.443 | -0.806 | total wins big |
| evidential | **-0.910** | -0.056 (n.s. borderline, p=0.025) | -0.910 | total wins big |

**Total beats epistemic-alone for all 9 of 9 methods on this axis**, exactly
the pattern flagged as precedent in the task brief (evidential -0.910 vs
-0.056 reproduces exactly). Every method's Mann-Whitney Q1-vs-Q4 test is
significant at p<1e-48 for both scores (full table in the CSV — p-values
alone don't distinguish effect size at n=396/group), so the story here is in
the correlation magnitudes, not significance.

### Axis 2: AliSim branch-length (DNABERT2, `promoter_all`, per-example, dense 16-point grid, n=6400/method)

| method | U_total rho | U_epistemic rho | U_aleatoric rho | total vs. epistemic |
|---|---:|---:|---:|---|
| conv_epinet | 0.142 | 0.122 | 0.143 | total wins (small) |
| laplace | 0.140 | **0.147** | 0.138 | epistemic wins (small) |
| ensemble_k5 | 0.145 | **0.191** | 0.143 | **epistemic wins** |
| ensemble_k3 | 0.130 | 0.129 | 0.130 | tie |
| cnn_mc_dropout | **0.287** | 0.269 | 0.287 | total wins (small) |
| cnn_ensemble | **0.244** | 0.224 | 0.246 | total wins (small) |
| rf_kmer | 0.359 | **0.563** | 0.179 | **epistemic wins big** |
| evidential | 0.033 | 0.036 | 0.031 | tie (both weak, p<0.02) |
| mc_dropout | **-0.070** (wrong sign) | +0.035 (right sign) | -0.078 | **epistemic wins on direction** |

**This is the one axis where the task's "sometimes epistemic wins" pattern
clearly replicates and generalizes, not just as one cherry-picked case.**
Three of nine methods have `U_epistemic` beating `U_total` (laplace,
ensemble_k5, rf_kmer — rf_kmer by the widest margin of anything found in
this whole analysis, 0.563 vs 0.359), one is an exact tie (ensemble_k3), and
`mc_dropout` is the most interesting case of all: `U_total` is *negatively*
correlated with branch length (more evolutionary divergence -> lower total
uncertainty, the wrong direction entirely) while `U_epistemic` alone is
weakly but correctly positive. This happens because `mc_dropout`'s
`U_aleatoric` component actively falls with branch length here (rho=-0.078)
and dominates the sum — a case where the aleatoric component is actively
*hurting* the total signal, not just diluting a real epistemic one.

### Axis 3: OOD severity (real / shuffled / random_dna, DNABERT2, n=1584/condition)

Reusing this session's already-computed Mann-Whitney tests
(`.docs/ALL_METHODS_OOD_SEVERITY.md`) for the 7 methods with a real
per-example `U_total`, plus 5 additional methods with aggregate-only means
(no `U_total` MWU possible — see README's data-availability audit):

| method | score | mean(real) | mean(shuffled) | p(shuf>real) | mean(random_dna) | p(rand>real) |
|---|---|---:|---:|---:|---:|---:|
| base | U_total | 0.352 | 0.397 | 6.4e-4 | 0.443 | 1.9e-34 |
| base_scaled | U_total | 0.326 | 0.368 | 2.0e-3 | 0.407 | 8.6e-34 |
| **mc_dropout** | **U_total** | 0.390 | 0.401 | **0.355 (n.s.)** | 0.401 | 8.0e-11 |
| **mc_dropout** | **U_epistemic** | 0.0122 | 0.0159 | **4.2e-6** | 0.0156 | 2.7e-58 |
| conv_epinet | U_total | 0.415 | 0.464 | 2.8e-5 | 0.551 | 1.7e-43 |
| conv_epinet | U_epistemic | 0.0714 | 0.0730 | 1.9e-3 | 0.0844 | 8.7e-27 |
| laplace | U_total | 0.352 | 0.401 | 5.6e-4 | 0.448 | 2.0e-36 |
| laplace | U_epistemic | 0.00165 | 0.00196 | 5.2e-4 | 0.00193 | 1.4e-17 |
| evidential | vacuity (primary) | 0.237 | 0.245 | 7.3e-3 | 0.268 | 1.4e-49 |
| conformal | set size (primary) | 1.037 | 1.051 | 0.064 (n.s.) | 1.036 | 0.531 (n.s.) |
| ensemble_k5 | U_total (mean-only, no MWU) | 0.3605 | 0.3943 | n/a — gap | 0.4344 | n/a — gap |
| ensemble_k5 | U_epistemic | 0.0131 | 0.0182 | 1.8e-15 | 0.0216 | 3.7e-78 |
| ensemble_k3 | U_total (mean-only) | 0.3612 | 0.3946 | n/a — gap | 0.4243 | n/a — gap |
| ensemble_k3 | U_epistemic | 0.0114 | 0.0132 | 1.3e-4 | 0.0155 | 8.6e-63 |
| cnn_mc_dropout | U_total (mean-only) | 0.3117 | 0.3685 | n/a — gap | 0.5961 | n/a — gap |
| cnn_mc_dropout | U_epistemic | 0.0609 | 0.0730 | 6.1e-9 | 0.1232 | 1.7e-114 |
| cnn_ensemble | U_total (mean-only) | 0.3481 | 0.3839 | n/a — gap | 0.6003 | n/a — gap |
| cnn_ensemble | U_epistemic | 0.1057 | 0.1185 | 5.2e-7 | 0.1902 | 2.4e-86 |
| rf_kmer | U_total (mean-only) | 0.6778 | 0.6934 | n/a — gap | 0.9424 | n/a — gap |
| rf_kmer | U_epistemic | 0.1496 | 0.2017 | 6.6e-65 | 0.2768 | 0.0 |

**The task's own precedent case reproduces exactly**: `mc_dropout`'s
`U_total` does not significantly separate the mild `shuffled` condition
(p=0.355) while its `U_epistemic` does (p=4.2e-6). But this does **not**
generalize to the other two methods with a real epistemic component on this
axis — `conv_epinet` and `laplace` are both significant on `U_total` *and*
`U_epistemic` at the mild shift, with total's p actually smaller
(more significant) for conv_epinet. So on this axis, "epistemic saves a
missed detection" is real for exactly one of three testable methods, not a
general property of the axis.

For the 5 aggregate-only methods, `U_total`'s mean rises monotonically with
severity in every case (directionally consistent with detection), but no
significance test is possible without per-example data — genuinely
undetermined here, not silently assumed to replicate the core-7 pattern.

### Axis 4: label-noise rate (DNABERT2, `promoter_all`, per-rate[-per-seed] means, 5 rates: 0.00/0.05/0.10/0.20/0.40)

Spearman rho of rate vs. mean score (n=15 for the 3-seed methods, n=5
otherwise — this axis never had per-example data, so these correlations are
computed at coarser granularity than the other three axes; see README):

| method | n | U_total rho (p) | U_epistemic (or analogue) rho (p) | U_aleatoric rho (p) |
|---|---:|---|---|---|
| conv_epinet | 15 | 0.982 (8.6e-11) | 0.753 (1.2e-3) | 0.982 (8.6e-11) |
| mc_dropout | 15 | 0.982 (8.6e-11) | **-0.971 (1.8e-9)** | 0.982 (8.6e-11) |
| laplace | 15 | 0.982 (8.6e-11) | **-0.906 (3.4e-6)** | 0.982 (8.6e-11) |
| ensemble_k5 | 5 | 1.000 | **-1.000** | 1.000 |
| ensemble_k3 | 5 | 1.000 | **-1.000** | 1.000 |
| rf_kmer | 5 | 1.000 | **-1.000** | 1.000 |
| cnn_ensemble | 5 | 1.000 | 0.100 (n.s., p=0.87) | 1.000 |
| cnn_mc_dropout | 5 | 1.000 | 0.900 (p=0.037) | 1.000 |
| evidential | 5 | 1.000 (ad hoc vacuity+aleatoric sum) | vacuity 0.900 (p=0.037) | 1.000 |

**This is the most one-sided result in the whole analysis.** `U_total`
tracks label-noise rate almost perfectly for every one of 9 methods
(rho >= 0.75, mostly >= 0.98). `U_epistemic` alone is not just weaker — for
**5 of 9 methods it is anti-correlated with the true noise rate**
(mc_dropout, laplace, ensemble_k5, ensemble_k3, rf_kmer: rho -0.91 to
-1.00). Using epistemic-alone as a label-noise detector on these 5 methods
would point in the *opposite* direction from the truth, not merely a weaker
one. Mechanically: as label noise rises, several methods' *disagreement*
across samples/trees/ensemble-members *shrinks* even as their *average*
per-example entropy rises sharply (the model settles into a more uniformly
low-confidence prediction that every sample/member agrees on, rather than a
confidently-wrong prediction some samples/members dispute) — consistent
with, and a sharper quantification of, the qualitative finding already
written up in `.docs/ALEATORIC_EPISTEMIC_CROSSCHECK.md` ("epistemic … also
rises" was the earlier, gentler finding for conv_epinet specifically; the
fuller roster here shows most methods' epistemic component actually reverses
sign under the same manipulation).

### Cross-axis summary: when does epistemic-alone win?

| axis | epistemic-alone ever wins? | how often |
|---|---|---|
| margin-boundary quartile | **No** | 0 / 9 methods |
| label-noise rate | **No** (worse: often backwards) | 0 / 9 methods win; 5 / 9 anti-correlated |
| OOD severity (mild shift only) | **Yes, but narrowly** | 1 / 3 testable methods (mc_dropout) |
| AliSim branch-length | **Yes, and clearly** | 3-4 / 9 methods (laplace, ensemble_k5, rf_kmer beat total; mc_dropout beats it on direction) |

Reported plainly: the task's own motivating precedent (boundary axis: total
wins essentially everywhere; OOD axis: mc_dropout is an exception) both
replicate exactly on this data. What's new here is that the **AliSim
branch-length axis is where "sometimes epistemic wins" is the norm, not the
exception** — a third to nearly half the roster does better with
epistemic-alone there, including the largest single margin found anywhere
in this analysis (rf_kmer, 0.563 vs 0.359). There is no method that wins
with epistemic-alone on every axis, and no method that loses on every axis
either (rf_kmer: worst-behaved decomposition among methods tested on the
boundary axis (`U_epistemic` conflated with margin, per the boundary
README), but best-behaved epistemic *sensitivity* to real evolutionary
divergence on the AliSim axis) — the answer is axis- and
method-specific, not a single global verdict either way.

---

## Analysis 2: ECE done properly

### Data availability (read this before the tables)

Per-example predicted-class probability survives in exactly **two** places
in this project's saved CSVs — see `README.md`'s audit table. Everything
else (every method on the label-noise-rate and AliSim axes; 6 of 7 methods
on the boundary and OOD-severity axes) only kept derived `U_*` scores, not
the underlying probability, so ECE/ACE/NLL/Brier are **not computable**
there without rerunning inference (out of scope here; not attempted).
**No axis/method combination has more than one seed of per-example
probability**, so the seed-mean-+/-std requirement from the task cannot be
met anywhere with existing data — a real gap, reported here rather than
papered over with a single-seed number presented as if it were a mean.

### Table: bin-count sensitivity, ACE, NLL, Brier

| dataset | method | condition | n | accuracy | ECE@10 | ECE@15 | ECE@20 | ECE@25 | range(ECE) | ACE@15 | NLL | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| margin_boundary | base | overall | 1584 | 0.879 | 0.0298 | 0.0298 | 0.0322 | 0.0330 | 0.0032 | 0.0271 | 0.290 | 0.0887 |
| margin_boundary | base | Q1 (hardest) | 396 | 0.674 | 0.0532 | 0.0532 | 0.0627 | 0.0659 | 0.0127 | 0.0636 | 0.620 | 0.2144 |
| margin_boundary | base | Q2 | 396 | 0.874 | 0.0576 | 0.0576 | 0.0576 | 0.0576 | 0.0000 | 0.0652 | 0.386 | 0.1107 |
| margin_boundary | base | Q3 | 396 | 0.977 | 0.0066 | 0.0066 | 0.0066 | 0.0066 | 0.0000 | 0.0224 | 0.110 | 0.0222 |
| margin_boundary | base | Q4 (easiest) | 396 | 0.992 | 0.0019 | 0.0019 | 0.0019 | 0.0019 | 0.0000 | 0.0108 | 0.044 | 0.0075 |
| ood_severity | evidential | real | 1584 | 0.873 | 0.0529 | 0.0467 | 0.0617 | 0.0603 | 0.0150 | 0.0623 | 0.319 | 0.0951 |
| ood_severity | evidential | shuffled | 1584 | 0.727 | 0.1435 | 0.1436 | 0.1457 | 0.1440 | 0.0022 | 0.1511 | 0.637 | 0.2093 |
| ood_severity | evidential | random_dna | 1584 | 0.493 | 0.3638 | 0.3639 | 0.3638 | 0.3649 | 0.0011 | 0.3638 | 1.140 | 0.3913 |

(Full precision in `analysis2_ece_ace_nll_brier.csv` / `analysis2_bin_sensitivity.csv`.)

### Finding 1: bin-count sensitivity is real but modest here, not the dominant error source

Across the 8 dataset/condition cells, ECE's range across bin counts
{10,15,20,25} is **0.0-1.5 percentage points**, and non-monotonic in three
cells (e.g. `evidential`/real: 5.29% -> 4.67% -> 6.17% -> 6.03% as bin count
increases — not a clean "more bins = more ECE" trend, consistent with
Nixon et al. 2019's point that binning is not just biased but can be
noisy/non-monotonic at fixed n). This confirms the *qualitative* R2-6
concern (arbitrary bin count is not a free choice) is legitimate, but on
this project's own data the magnitude (up to ~1.5pp) is **smaller** than the
~5pp single-seed-vs-5-seed-mean gap `FINDINGS.md` #1 already found for the
manuscript's reported ECE values. Both are real, independent sources of
ECE instability; on this evidence, seed choice was the bigger problem for
this project's manuscript than bin-count choice would have been, though
both are worth controlling for.

### Finding 2: ACE (equal-mass) agrees closely with standard ECE here

ACE-at-15-bins never differs from equal-width ECE-at-15-bins by more than
~1.5 percentage points in any of the 8 cells (closest: Q3/Q4, both under
0.002; largest gap: OOD/real, 4.67% vs 6.23%, and boundary/Q1, 5.32% vs
6.36%). **This data does not show ACE and standard ECE substantially
disagreeing** — reported plainly per the task's instruction not to shade
results toward an expected conclusion. The instability shown in Finding 1 is
present in both binning schemes at similar magnitude, i.e. equal-mass
binning is not a free fix for bin-count sensitivity on this data, just a
different (and in the literature, generally more sample-stable) way of
forming the bins.

### Finding 3: the R2-3 pathology — reproduced directly, and it's slice-dependent

**Margin-boundary axis, Q1 vs Q4 — the pathology, reproduced directly on
this project's own data**: Q1 has accuracy 67.4% (the classifier is wrong
roughly 1 in 3 times) but ECE of only 5.3-6.6% across every bin count
tested — a number that, read alone, suggests "fairly well calibrated."
Reading NLL and Brier alongside it tells a different story: Q1's NLL (0.62)
is **14x** Q4's (0.044), and Q1's Brier (0.214) is **28x** Q4's (0.0075) —
both proper scoring rules make the huge reliability gap between the two
slices obvious in a way ECE's binned-average design does not. This is
exactly the R2-3 concern (`.docs/RESUBMISSION_PLAN.md` item 7): "a modest
ECE gain/value on a mostly-wrong classifier shouldn't be called improved
reliability" — here it's not even a gain being over-read, just a
moderate-looking single ECE value that undersells how bad Q1 actually is
relative to Q4, exactly the failure mode NLL/Brier are supposed to catch.

**OOD-severity axis, by contrast — no masking.** `evidential`'s ECE rises
in step with its accuracy collapse: real (87.3% acc, 5.3-6.2% ECE) ->
shuffled (72.7% acc, 14.4-14.6% ECE) -> random_dna (49.3% acc — chance —
36.4% ECE). NLL and Brier rise proportionally too (NLL 0.32 -> 0.64 -> 1.14;
Brier 0.095 -> 0.209 -> 0.391). **ECE is not universally broken in the R2-3
sense on this project's data — it tracks reliability correctly when the
population shift is large and uniform (OOD severity), and only "masks" the
problem on a within-distribution, same-model slice where the model's
average confidence happens to roughly match its average accuracy even
though that accuracy is mediocre (margin-boundary Q1).** This is the
honest, non-monolithic answer to the task's own framing question ("does
ECE calculated separately on Q1 vs Q4 mask the accuracy gap") — it partially
does (the single ECE number for Q1 undersells the gap versus Q4 relative to
what NLL/Brier show) but it does not fail to notice a problem exists (Q1's
ECE is still 3-30x higher than Q3/Q4's, in the right direction) — the
practical fix R2-3 asks for (pair ECE with NLL/Brier, and stratify by
sub-population/accuracy-bucket rather than reporting one global number) is
directly supported by this evidence, not merely asserted.

### Verdict on the reviewers' R2-3 / R2-6 concerns, grounded in this project's own numbers

- **R2-6 (bin-count sensitivity, no NLL/Brier reported)**: legitimate.
  Bin-count sensitivity is real (up to 1.5pp here) even if smaller than the
  seed-variance problem; NLL/Brier were not previously computed at all for
  any of these four axes and, once computed, materially change the read of
  the margin-boundary Q1 slice (Finding 3) in a way ECE alone did not.
- **R2-3 (ECE improving without accounting for low accuracy)**: legitimate,
  and directly demonstrated on this project's own margin-boundary Q1 slice
  — not merely a hypothetical concern.
- **FINDINGS.md #1 (single-seed ECE reported as if multi-seed)**: this
  follow-up could not re-confirm or extend that finding with new numbers —
  no axis in this follow-up has both multi-seed data *and* a saved
  probability column at the same time (the 3-seed axis, label-noise rate,
  never saved probabilities; the 2 axes with saved probabilities are both
  single-seed). The original finding stands on its own evidence
  (`tables/calibration_summary.csv`, per `FINDINGS.md`); this follow-up
  simply could not add to or subtract from it with the data available here.
