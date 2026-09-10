# Independent expert review: AliSim / boundary-sampling / calibration follow-up work

Reviewer stance: adversarial, independent (no anchoring on any prior review of
this branch). Scope: everything built on `worktree-fix-epinet-batch-z` this
session — the epinet per-example-`z` fix
(`nn_proj/models/epinet/epinet.py`), `data_gen/promoter_alisim/`,
`data_gen/aleatoric_boundary/`, and `data_gen/uq_metrics_followup/`
(including `paper_tables.tex`). Numbers below are re-derived from the actual
CSVs and code on this branch, not copied from the READMEs — spot-checks are
noted where I independently recomputed a number and it matched (or didn't).

---

## Verdict up front

**Needs substantial additional work before the headline claim can be used as
written**, specifically because of Finding #1 below. The statistical
analysis on top of the existing per-example data (Sections 3 and the
`uq_metrics_followup` layer) is careful, mostly well-documented, and the
numbers I spot-checked against source CSVs reproduced exactly. But the
flagship new method in every table — `conv_epinet` — is evaluated with a
checkpoint that was never actually trained under the mechanism being
validated. That is not a nitpick; it undercuts the specific claim this
session exists to support ("the fix works, and even with it fixed, the
decomposition still doesn't separate cleanly"). The broader claim (no method
in a 9-11-method roster cleanly separates aleatoric from epistemic, and RF is
the partial exception) is much better supported, because it does not depend
on that one checkpoint — the CNN, RF, ensemble, and Laplace results are
methodologically independent of the epinet fix and tell a consistent story
on their own. So: **the roster-wide claim is close to solid; the
epinet-specific claim is not, as currently evidenced.**

---

## 1. Is the core headline finding well-supported?

**Mostly yes for the roster-wide claim, no for the epinet-specific claim, for
a reason the project has not identified.**

### 1.1 [SINKS THE EPINET-SPECIFIC CLAIM] The `conv_epinet` checkpoint was trained under the *old*, disproven shared-`z` code — the fix was only ever applied as an eval-time patch

This is the single most damaging finding in this review, and it appears to
be new — I don't see it flagged anywhere in the READMEs or in
`.docs/RESUBMISSION_PLAN.md`.

- `checkpoints/seed_1/DNABERT2/label_noise_r00/epinet` (the checkpoint every
  `conv_epinet` result in `promoter_alisim/`, `aleatoric_boundary/`, and
  `uq_metrics_followup/` reuses) has an on-disk mtime of **2026-09-03
  16:02:58** (`ls -la --time-style=full-iso
  /scratch/home/glh52/glm-epinet-pyt/checkpoints/seed_1/DNABERT2/label_noise_r00/`).
- The per-example-`z` fix (commit `fb38241`, "epinet: draw index sample z
  independently per example, not shared across the batch") was committed
  **2026-09-07 16:54:57** — four days *after* that checkpoint was trained.
- Both READMEs describe the fix as something applied to run *inference*:
  "`nn_proj/models/epinet/epinet.py` was replaced wholesale with the version
  from branch `worktree-fix-epinet-batch-z` ... before running `conv_epinet`"
  (`data_gen/promoter_alisim/README.md:304-306`,
  `data_gen/aleatoric_boundary/README.md:369-372`). Neither README ever says
  the epinet checkpoint was retrained under the fixed code, and the
  checkpoint timestamp confirms it wasn't — it predates the fix.

Why this matters mechanically, not just as a provenance technicality: the
pre-fix `GaussianIndexer.forward()` returned a single `[Dz]` vector that
`ProjectedMLP.forward` then broadcast to the whole batch
(`z.unsqueeze(0).expand(B, -1)`, confirmed by diffing `fb38241~1` against
`fb38241`). `train_epinet.py` for DNABERT2 trains with `k_train=8`
(`nn_proj/models/DNABERT2/train_epinet.py:85`), i.e. **every training step
drew 8 index samples and averaged cross-entropy over them** — but under the
code this checkpoint was actually trained with, those 8 samples were each a
*single* `[Dz]` vector shared by every example in the minibatch, not 8×B
independent draws. That is a materially different training signal for
`train_head`/`prior_head`/`conv_prior`'s contraction: the network learned to
produce a correction conditioned on a per-step, batch-wide index draw, not a
per-example one. Patching the sampling code at inference time to draw a
different `z` per example doesn't retroactively give the network the training
signal it would need to have learned a *meaningful per-example* index
mapping — it just changes what gets fed through weights that were never
optimized against that input distribution.

**Consequence for every `conv_epinet` result in this session's new work**:
the `U_epistemic` values reported for `conv_epinet` throughout
`promoter_alisim/`, `aleatoric_boundary/`, and `paper_tables.tex` reflect a
train/inference mismatch, not a clean test of "does the fixed per-example-z
epinet decompose uncertainty correctly." A reviewer who checks checkpoint
provenance (as I just did, with two `ls` calls and one `git log`) will find
this in minutes, and it directly undermines the paper's own methods-section
framing (`.docs/RESUBMISSION_PLAN.md` row #8: "cheap targeted confirmation —
patch inference-time sampling on the flagged cells only, no retraining, see
if AUROC moves" was explicitly proposed as *stage (a)*, a cheap sanity check
before deciding whether full retraining/propagation was warranted — not as
the final, publication-ready state). This session's work never advanced past
stage (a), but `paper_tables.tex` presents `conv_epinet`'s results as final,
citable numbers (Tables 1–4) without that caveat anywhere in the prose.

**What would fix it**: retrain the epinet head (cheap — it's a small MLP +
frozen conv prior on top of a frozen backbone, not a full fine-tune) under
the corrected code, and confirm the reported `U_epistemic` dose-response
curves are qualitatively unchanged. Until that's done, every `conv_epinet`
number in this body of work should be captioned as evaluating "the fixed
sampling code applied to a pre-fix-trained checkpoint," not as validating
the fix's effect on the epinet's actual learned behavior.

**Mitigating factor**: this does *not* touch `mc_dropout`, `laplace`,
`ensemble_k5/k3`, `cnn_mc_dropout`, `cnn_ensemble`, or `rf` — none of those
six methods touch `nn_proj.models.epinet` at all. Since RF shows the
cleanest signal on the epistemic axis and the ensembles/CNN show the
strongest conflation on the aleatoric axis, **the roster-wide "nothing
separates cleanly" claim survives this finding**; only the specific claims
made *about* `conv_epinet` (e.g., "the epinet fix was validated," or reading
too much into `conv_epinet`'s particular rho values in Tables 3–4) are
undermined.

### 1.2 Checkpoint substitution (real, but adequately disclosed)

Both READMEs state plainly that `trained_models_*/DNABERT2/promoter_all`
(the manuscript's original checkpoint) no longer exists on disk, and that
`checkpoints/seed_1/DNABERT2/label_noise_r00/{base,epinet,evidential}` is
substituted instead — "same substitution already used and flagged in prior
work this session." This is honestly reported and the substitute is
byte-identical training data (0% injected noise), so it's a reasonable
stand-in for *this* exploratory follow-up. It does mean, though, that none of
this session's numbers can be directly compared to any number the original
manuscript published (different hyperparameters/training run, not just a
different name) — worth stating explicitly if any of these tables get placed
next to the manuscript's original Figures 4–9, which use a differently
trained checkpoint family entirely.

### 1.3 Single-seed evaluation — inadequate for the strength of claims made, adequate for the exploratory framing actually used

Every new dataset here uses **seed 1 only**, while the project's own
established convention (documented at length in `.docs/REVISION_PLAN.md`'s
`promoter_motifs_v2` 5-seed table) is 5 seeds, precisely because seed
variance is large enough to matter: `FINDINGS.md` item 1 documents a single
seed (42) being off by 5.6-7.6 percentage points of ECE from the 5-seed mean
in exactly this kind of regulatory-classification setting. There is no
seed-variance estimate anywhere in this session's new work, so a claim like
"`rf` shows the cleanest epistemic signal of any method" cannot currently be
distinguished from "seed 1's particular RF fit happened to show the cleanest
signal." Given `rf`/CNN are retrained fresh (not just re-evaluated) for this
work, a from-scratch RF/CNN training run has its own extra source of
seed-to-seed variance on top of whatever the DNABERT2 checkpoint seed
contributes — and no CI/std is reported for `rf`'s reported rho=0.563
(Table 3) at all. This is explicitly and honestly flagged in both READMEs
("Single seed (seed 1) for every method... no seed-variance estimate") — so
it is not a hidden problem, but it is a real limitation on how strong a
claim like "RF has THE cleanest signal" (superlative, singular) can support.
A more defensible version of the claim is "RF showed the cleanest signal
*at this one seed*," which is a materially weaker statement than what
`paper_tables.tex`'s prose currently states ("`rf` shows a real, clean rise
(ρ=0.563)" — true as stated, but the surrounding text treats it as if it
generalizes to the method, not to this one instantiation of it).

### 1.4 Sample sizes

- **400 anchors** (AliSim) and **1,584 examples / 396 per quartile**
  (boundary) are adequate for the specific hypothesis tests run — the
  Mann-Whitney and Spearman tests reported are almost all significant at
  p≪0.001, so underpowering isn't the concern (if anything, see §3 on
  overpowered small-effect significance). The concern is representativeness,
  not raw N: 400 anchors is a small fraction of `promoter_all`'s ~30k
  training sequences and the resulting dose-response curves (e.g. the
  hump-shaped ones) are estimated from means over only 400 points per
  branch-length bin, so a "peak at t=0.2" vs. "peak at t=0.4" distinction in
  the shape-characterization table (`results_summary_shape.csv`) is itself
  fairly noisy at this N, and no bootstrap CI is given for any peak location.
- The boundary axis reuses n=396 per quartile for **11 separate MWU tests
  per method** (aleatoric, epistemic — repeated across 9-11 methods) off the
  *same* 1,584-example test set — every method's Q1/Q4 split is the same 792
  examples (with per-method uncertainty scores), so these are not 9-11
  independent samples; they are 9-11 different lenses on the identical 396
  boundary examples. That's fine for the specific pairwise comparisons (each
  method's own Q1 vs its own Q4), but it means the "confirms roster-wide"
  framing in `data_gen/aleatoric_boundary/README.md` ("across a Bayesian
  post-hoc method... a classical non-neural ensemble — five qualitatively
  different UQ mechanisms") is five different *scoring functions* on one
  fixed set of hard examples, not five independent experiments — a
  meaningfully weaker form of generalization than the prose implies.

---

## 2. Dataset construction critique

### 2.1 AliSim divergence axis (epistemic-shift)

**Real strength**: using AliSim to evolve *real* anchor sequences under an
explicit, cited substitution model with a fixed root is a substantial
methodological upgrade over the "shuffled/random_dna" ad hoc corruption this
project used before, and the base-frequency-from-corpus / kappa-alpha-from-
literature split is a defensible, explicitly justified choice (fitting
kappa/alpha via ModelFinder on non-orthologous sequences would indeed be a
category error, as documented). The `ref` leaf sanity check (branch length
0.0 must exactly reproduce the anchor) and the monotonicity assertion in
`verify()` are genuine, automated invariants, not just narrative claims — I
confirmed `make_alisim_data_dense.py:353-380` actually asserts both.

**Real weakness — "OOD" here never gets very OOD.** I recomputed mean
`realized_identity_to_anchor` by branch length directly from
`data_gen/promoter_alisim/csv_data/promoter_alisim_dense.csv`:

| branch_length | mean identity |
|---:|---:|
| 0.0 | 1.000 |
| 0.5 | 0.695 |
| 1.0 | 0.567 |
| 1.5 | 0.498 |
| 2.0 | 0.454 |
| 3.0 | **0.392** |

The README states (about the original 8-point grid's t=1.5 endpoint) that
identity is "close to the ~0.25 expected under a uniform stationary
distribution at full saturation." At the dense grid's *actual* extreme
(t=3.0, more than double the original max), mean identity is still **0.392**
— nowhere near the claimed 0.25 saturation floor. This is very plausibly a
real consequence of Gamma(4) rate heterogeneity with shape α=1.0 (an
exponential rate distribution): a substantial fraction of sites draw a very
low rate and barely substitute even at large nominal branch length, so the
*bulk-average* identity converges to the 0.25 floor far more slowly than a
single-rate model would. Whatever the mechanism, the practical consequence
is that **the "far" end of this severity axis (t=3.0, called "well past
typical divergence" in the README) is still ~40% identical to the anchor on
average** — a real evolutionary/compositional shift, certainly, but not the
"fully decorrelated, structurally unrelated" input a true OOD/far-domain
comparison would need. This weakens the strength of "epistemic barely moves
even under real evolutionary divergence" as a maximally damning finding
against the decomposition: it's also possible epistemic barely moves because
the tested inputs, even at the grid's extreme, never got as far from the
training distribution as, say, `nonbacterial` sequences are from bacterial
training data elsewhere in this project. A reviewer would reasonably ask for
either (a) a report of identity at t=3.0 in the *main* write-up (currently
only in the 8-point grid's older, less-extreme numbers) or (b) an even more
extreme branch length / an indel-driven variant to confirm the plateau is
genuine rather than an artifact of insufficient divergence.

**Label-validity caveat is applied inconsistently.** The AliSim README
correctly hedges that `labels`/`original_label` should not be treated as a
validated ground truth at high branch length ("this dataset is meant for
epistemic-uncertainty evaluation, not as a validated classification
benchmark at high severity"). But `paper_tables.tex` §1's Table 1
("Calibration under simulated divergence") reports **accuracy, ECE, and
NLL** at t=1.0 and t=3.0 computed directly against that same
possibly-invalid label, with only accuracy explicitly hedged in the prose
("I don't assume the label still holds at high divergence — accuracy is
tracked, not treated as ground truth there"). ECE and NLL are computed
*identically* against the same label and are not hedged at all, even though
the exact same objection applies to them — if the label may not hold, "rf
has the best NLL/ECE at t=3.0" is exactly as uninterpretable as "rf has the
best accuracy at t=3.0" would be, and the paper only hedges the latter. This
is a real overclaiming gap, not just a wording nitpick — see §5.

### 2.2 Margin-quartile boundary axis (aleatoric-shift)

**Real strength, and the stronger-designed of the two datasets.** The
independent, non-circular sanity check (true-label accuracy by margin
quartile, computed without reference to any method's own uncertainty score)
is exactly the right control, and it passes cleanly and dramatically
(67.4% → 99.2% accuracy Q1→Q4, verified directly against
`data_gen/aleatoric_boundary/csv_data/margin_scores.csv`). This genuinely
establishes the margin score tracks real difficulty, not an artifact.

**Real weakness — circularity is present at exactly one remove, and it's
structurally, not just incidentally, relevant.** The margin quartiles are
built from the `base` DNABERT2 checkpoint's own decision function. Every
downstream UQ method evaluated (`mc_dropout` runs dropout on that same
`base` checkpoint; `conv_epinet` is a head trained on top of it; `laplace` is
a post-hoc approximation fit on it; the ensembles reuse independently
fine-tuned siblings of the same architecture; `evidential` shares the same
base architecture/training data) is a close structural relative of the very
model that defined "hard." This does not invalidate the aleatoric-axis
finding (aleatoric rising near a boundary is expected and the effect is
enormous and consistent), but it plausibly explains *why* the epistemic axis
also moves so much: near a decision boundary, any per-sample perturbation
(a dropout mask, an index vector, an ensemble member's idiosyncratic
weights) is mechanically more likely to flip which side of the boundary a
given draw lands on, inflating a disagreement-based "epistemic" proxy for
reasons that have nothing to do with input novelty. Both READMEs already
name this exact mechanism as a "plausible mechanical reason... not confirmed
further here" — which is honest, but it means the headline claim
("epistemic doesn't separate from aleatoric near a decision boundary") is,
on the evidence presented, indistinguishable from a narrower and less novel
claim: "variance/disagreement-based uncertainty estimates are structurally
noisier near a 50/50 decision boundary," which is a known property of
sampling-based uncertainty estimates in general, not a genomics- or
GLM-specific finding. The paper would be more defensible citing that this is
a *structural* property of variance-based decompositions near a boundary
(with a citation to the general ML literature on this, if one exists) rather
than implying it's a novel discovery about epistemic/aleatoric decomposition
in genomic LMs specifically.

**A second, milder circularity concern**: "matched" is not tested across
different base checkpoints, so the entire boundary-difficulty ranking is
checkpoint-dependent by construction — a different seed's `base` checkpoint
would produce a different margin ranking (possibly correlated but not
identical), and nothing here checks how stable the Q1/Q4 assignment is
across seeds. Given §1.3's single-seed limitation, this compounds: both the
*difficulty partition itself* and *every uncertainty score evaluated against
it* come from checkpoints trained (or derived from a checkpoint trained)
with seed 1 only.

---

## 3. Statistical rigor

### 3.1 Large-n p-value theater — present, but not maliciously so

Several reported p-values are literally `0.0` from `scipy.stats.spearmanr`
underflow (e.g. `margin_boundary,cnn_ensemble,U_total,spearman_vs_margin`
in `analysis1_total_vs_epistemic.csv` reports `p=0.0` at rho=-0.78, n=1584;
several `conv_epinet`/`ensemble_k3` rows do the same). Reporting an
underflowed `0.0` as if it were a measured value (rather than "p < 1e-300"
or similar) is sloppy, though the effect sizes accompanying these
underflowed p-values are themselves large (|rho| > 0.75-0.99), so the
underlying claims are not "large-n theater masking a small effect" in these
particular cases — the effect sizes are large and the significance is
real. The theater risk is more relevant to the OOD-severity axis, where
`README.md` for `uq_metrics_followup` reports Spearman rho as low as 0.026
(evidential vs. branch length, p=0.138 n.s., correctly reported as not
significant) alongside rho=0.055 (mc_dropout `U_epistemic` vs. branch
length, p=0.002, "significant") in the *same table*
(`data_gen/promoter_alisim/README.md`'s 8-point-grid Spearman table). A
Spearman rho of 0.055 at n=3200, while nominally significant, explains on
the order of 0.3% of rank variance — reporting this as "rises with weak but
significant magnitude" alongside p=0.0020 is technically correct but is
exactly the kind of large-n significance that a skeptical reader should be
warned reads as much smaller in practical terms than the p-value alone
suggests. Both READMEs elsewhere do explicitly caution against reading
p-values in isolation ("the quartile means and Spearman rho are the numbers
to read for effect size, not p-values") — but this caution isn't applied
uniformly to *its own* rho=0.055 case, which is treated as a real, if weak,
"partial confirmation" rather than as also being borderline-noise at that
effect size.

### 3.2 No multiple-comparisons correction anywhere

A grep across `data_gen/uq_metrics_followup/`, `data_gen/promoter_alisim/`,
and `data_gen/aleatoric_boundary/` for "bonferroni," "multiple comparison,"
"fdr," or "holm" returns nothing. This project runs on the order of 40+
Mann-Whitney tests (9-11 methods × 2 score axes × 2 datasets, plus the
label-noise-rate axis) and dozens of Spearman tests, with no stated
correction. In most of the cells here the effect sizes are so large that a
Bonferroni correction over even a few hundred tests would not flip any
conclusion — but there are borderline cases (evidential/branch-length,
p=0.138 and p=0.275, already reported as n.s.; mc_dropout/branch-length
epistemic, p=0.0020) where a stated correction policy would matter for
whether "significant" claims survive scrutiny. Absent that, a reviewer is
entitled to ask whether the handful of "barely significant" results
scattered through this roster (p in the 0.001-0.02 range) would survive
even a simple Bonferroni correction over the ~40+ tests actually run. Likely
some would not.

### 3.3 Spearman rho as a summary statistic actively misrepresents at least one reported "hump" relationship

This is a concrete, checkable problem, not a general worry. From
`data_gen/promoter_alisim/uncertainty_eval/dense_grid/results_summary_shape.csv`:

| method | baseline U_epi (t=0) | peak U_epi | % rise to peak | last U_epi (t=3.0) | % above baseline at t=3.0 |
|---|---:|---:|---:|---:|---:|
| mc_dropout | 0.0270 | 0.0426 (at t=0.2) | **+57.7%** | 0.0276 | **+2.3%** |

`paper_tables.tex` Table 3 reports mc_dropout's `rho_epi = 0.035` and its
prose concludes "epistemic barely moves for most methods." Taken at face
value, rho=0.035 reads as "essentially no relationship." But the actual
per-branch-length trajectory shows a real, non-trivial 58% rise to a peak at
t=0.2, followed by an almost-complete decline back to baseline by t=3.0 — a
real, replicable, non-monotonic dose-response that a linear rank correlation
is mathematically close to blind to (a symmetric hump nets to ≈0 Spearman
rho regardless of amplitude). The paper's own qualitative shape
classification (`results_summary_shape.csv`'s `shape` column: "declines
after peaking") already contains this information — it just isn't
reconciled with the single-number rho reported in the main table.
Reporting rho alone here **actively understates** the strength of the
epistemic response for mc_dropout, in a direction that happens to favor the
paper's headline claim ("epistemic barely moves"). This is worth flagging as
directional, not just a general statistics nitpick: the omission makes the
"epistemic doesn't respond" story look more uniform across the roster than
the underlying data supports. A reviewer doing exactly the shape-CSV
cross-check I just did would catch this immediately. The fix is cheap:
report peak-relative-to-baseline rise (already computed) alongside rho in
the main table, or use a non-monotonic-aware statistic (e.g., max absolute
deviation from baseline, or a quadratic/polynomial fit R²) as the headline
number instead of/alongside Spearman.

### 3.4 The `evidential` U_epistemic inconsistency between the two datasets is unresolved, not just noted

`all_methods_stratified_ece.py:28-34` documents, in a code comment, that
`evidential`'s `U_total = U_aleatoric` exactly on the boundary dataset
(canonical, `U_epistemic ≈ 0` since it's a single forward pass) but that on
AliSim, `evidential`'s `U_epistemic` column is "a nonzero, differently-scaled
quantity... that does NOT sum with U_aleatoric to a normalized total" and is
of unknown provenance ("some Dirichlet-sampling extension not in the
canonical implementation"). The script's response is to sidestep the
inconsistency (use `U_aleatoric` alone for `evidential` on both datasets),
which is a defensible practical workaround for the ECE computation
specifically, but the underlying inconsistency — the same method producing
two different, incompatible epistemic-quantity semantics across two
datasets in the same project — is never actually diagnosed. It's possible
this is an artifact of two different script authors/sessions computing
"evidential epistemic" two different ways at different points in the
project's history; either way, a reviewer who reads the two READMEs closely
enough to notice this (as I did) will reasonably ask "which one is right,
and why does the analysis paper over rather than resolve it?"

---

## 4. Missing baselines/controls

- **No held-out re-verification of the epinet fix's effect on the
  manuscript's original headline claims.** Covered in depth in §1.1 — worth
  repeating here because it is explicitly the kind of "missing baseline"
  this review was asked to check for, and it's the most consequential one
  found.
- **RF's aleatoric-axis calibration failure is not mechanistically
  explained, and the manuscript prose's own description of it doesn't match
  its table.** `paper_tables.tex` Table 2's ECE column for `rf` across
  Q1→Q4 is **.089 / .111 / .143 / .068** — it peaks at Q3 (.143), then
  *drops* at Q4 to below its own Q1 value. The prose claims "evidential/rf
  actually get worse-calibrated from Q1→Q4" — true for `evidential`
  (.079→.056→.098→.099, net rise Q1→Q4), but not accurate for `rf` read as
  a Q1-vs-Q4 endpoint comparison, since .068 < .089. What the table
  actually shows for `rf` is non-monotonic (worst at Q3), not a clean
  "gets worse toward the easy end" story — a reader who checks the table
  cells against the prose (as I just did) will find the two don't quite
  agree for one of the two methods being generalized about. Separately from
  that phrasing issue: given RF is simultaneously praised as showing "the
  cleanest epistemic signal of any method" on the divergence axis, a
  reviewer will want to know whether RF's k-mer+count-feature representation
  is fundamentally miscalibrated on the confident majority class in a way
  that's independent of, or entangled with, its comparatively good epistemic
  behavior elsewhere. This is flagged as an open question in the source
  material but never investigated (e.g., checking per-tree probability
  concentration, or whether `min_samples_leaf=20` — chosen specifically to
  avoid degenerate pure leaves — has a side effect of *under*-concentrating
  probability mass on the easy majority).
- **No cross-backbone check.** Confirmed directly: every file under
  `data_gen/uq_metrics_followup/dnabert_only/` is DNABERT2-only by name and
  by content (`CKPT = ".../DNABERT2/label_noise_r00/base"` hardcoded), and
  the AliSim/boundary raw evals are likewise DNABERT2-only throughout. The
  manuscript otherwise covers four backbones (DNABERT2, NT_transformer,
  HyenaDNA, CARMANIA) with meaningfully different behavior documented
  elsewhere in this project (e.g. `REVISION_PLAN.md`'s promoter-motifs table
  shows CARMANIA as an outlier with an 11pp novelty-generalization drop
  where other backbones show ~3-6pp). There is no basis in this session's
  work for claiming the decomposition-conflation finding generalizes beyond
  DNABERT2, and the write-ups do not claim it does — but neither
  `paper_tables.tex` nor the READMEs state the DNABERT2-only scope as
  explicitly as a reviewer will want (it has to be inferred from directory
  names and hardcoded checkpoint paths, not from an explicit "this section
  covers one backbone" sentence in the manuscript prose itself).
- **No ablation isolating whether the epinet's conv prior (a frozen random
  function of the input, contributing `conv_prior_scale=1.0 × convp` to
  every logit) is itself responsible for some of `conv_epinet`'s aleatoric
  leakage**, independent of the per-example-z question. `.docs/MODEL_CODE_FIXES.md`
  item 4.2 names this exact ablation (`prior_scale=0`, `conv_prior_scale=0`)
  as still open project-wide; this session's work doesn't run it either, so
  `conv_epinet`'s reported leakage could in principle be partly a "generic
  confidence shrinkage from an added frozen random term" artifact (Reviewer
  2's original hypothesis) rather than purely an epistemic/aleatoric
  entanglement — compounding rather than replacing the §1.1 concern about
  this specific method's evidentiary weight.

---

## 5. Overclaiming risk

- **"I mostly can't cleanly separate epistemic from aleatoric uncertainty...
  on either axis, with any method tested — including true
  independently-trained deep ensembles"** (`paper_tables.tex` §3 opening).
  This claim is **well-scoped and defensible for the aleatoric-shift axis**
  (all 9 methods checked show real epistemic movement on margin quartile,
  confirmed directly from `results_summary.csv`'s Q1-vs-Q4 numbers). It is
  **overstated for the epistemic-shift axis**: RF shows rho=0.563 (clean,
  the single largest effect size in either table), and the text's own next
  sentence ("only rf shows a real, clean rise") already contradicts the
  "with any method tested" framing two sentences earlier — the paragraph
  states the absolute claim and its own exception in adjacent sentences
  without reconciling them. A more accurate framing: "the decomposition
  fails to separate cleanly for every *pretrained-transformer-based* method
  tested, and for most sampling mechanisms overall, but a classical
  from-scratch k-mer model is close to a genuine counterexample on the
  epistemic-shift axis specifically." That's a more interesting and more
  defensible claim than the current blanket statement, and the data already
  supports it — it just isn't stated that way in the prose.
- **Needs an explicit single-checkpoint/single-seed hedge in the actual
  manuscript prose, not just in READMEs.** `paper_tables.tex` itself
  contains no sentence stating this is one checkpoint, one seed, one
  backbone — that context currently lives only in `README.md` files that
  won't ship with a manuscript. Given how much the roster-wide framing
  ("across a Bayesian post-hoc method... a classical non-neural ensemble —
  five [nine] qualitatively different UQ mechanisms") leans on breadth of
  *methods* as the source of confidence, the prose should say plainly that
  breadth is not matched by depth (1 seed, 1 backbone, 1 substituted
  checkpoint family).
- **ECE/NLL at t≥1.0 on AliSim are reported without inheriting the
  label-validity hedge that accuracy gets** — covered in §2.1. This is a
  real overclaiming risk specifically because Table 1's caption
  ("Calibration under simulated divergence") and its bolding of `rf` as
  best on ECE/NLL at t=1.0/3.0 reads as an unqualified quantitative
  ranking, when by the paper's own logic elsewhere the ground truth at
  those branch lengths is explicitly not asserted to be valid.
- **RF's "cleanest epistemic signal" framing needs the single-seed caveat
  attached at the point of the claim**, not three paragraphs later or in a
  different file (§1.3).

---

## 6. Presentation/rigor gaps

- **Method-roster mismatch between calibration tables (11 methods) and
  decomposition tables (6 methods) in `paper_tables.tex`** — already
  self-disclosed in the file's own header comment ("NOTE on method roster
  mismatch... Not reconciled across sections; flagging so this isn't
  mistaken for a data inconsistency"). Good that it's disclosed to a
  co-author reading the `.tex` source, but **a comment in the LaTeX source
  is not a disclosure to a journal reviewer** — if this table set ships,
  the mismatch needs a sentence in the actual manuscript prose (e.g., a
  table footnote: "shown for the 6 methods with the clearest independent
  provenance; laplace/ensemble_k3/evidential omitted here for space, see
  Table X"), or the tables should simply be made consistent.
- **`aleatoric_boundary/README.md`'s own count mismatch, self-caught but
  worth noting**: the task instructions apparently described the combined
  table as "8 method-configurations" in two places while separately saying
  to add 6 new methods to the existing 3 (=9). The README catches and
  reports this discrepancy transparently rather than silently forcing a
  count — a genuine strength in process transparency, but it also signals
  that the instructions this work was produced under were internally
  inconsistent, which is worth someone tracing back to source.
- **Literal `p=0.0` values in CSVs** (scipy underflow, not a true zero) —
  minor, but should be reformatted as `<1e-300` or similar before any of
  these numbers reach a table.
- **The `T_FITTED = 1.1496... # from prior fit, LBFGS/NLL, held-out 10% of
  promoter_all train, seed 42` comment in `dnabert_only/all_methods_stratified_ece.py`
  and `dnabert_stratified_ece.py` is genuinely ambiguous** — "seed 42" reads,
  on first pass, as if it refers to a different model checkpoint than the
  seed-1 checkpoint used for every other method in the same script (I
  initially misread it this way while reviewing). It in fact refers to the
  `data_seed` used only for the temperature-fitting validation split
  (`fitted_temperature.txt` confirms `checkpoint=.../seed_1/.../base`,
  `data_seed=42`), and is not a mismatch — but the comment as written invites
  the misreading and should be rephrased (e.g., "validation split seed 42,
  same seed-1 base checkpoint used throughout this script").
- **Absolute, hardcoded paths throughout** (`/scratch/home/glh52/...`,
  `/home/glh52/.claude/jobs/80f6ac1e/tmp/...` as an output path inside
  `all_methods_stratified_ece.py:181-182`) mean none of these scripts are
  currently reproducible outside this exact machine/session — a much
  smaller issue than the above, but worth a pass before archival, since this
  project otherwise cares a lot about reproducibility (`REPRODUCE.md`,
  `tests/test_entrypoints.py`).

---

## 7. The single most damaging critique, and is it fair

**Most damaging: the flagship confirmatory finding for this session's stated
purpose — "the epinet per-example-z fix is real and I re-verified its
downstream effect on the decomposition" — is not actually a test of the fix
at all, because the epinet checkpoint used everywhere was trained four days
before the fix existed, under the exact shared-z regime the fix was written
to correct.** (§1.1.)

Is it fair? Yes, and it's checkable in about five minutes with two `ls`
calls and a `git log` — which is itself part of why it's damaging: it's not
a subtle statistical judgment call, it's a provenance fact the existing
documentation had every opportunity to state and didn't. It is, however,
**containable rather than fatal to the whole session's output**: the
roster-wide claim (aleatoric leaks into epistemic under label-noise/margin
difficulty; epistemic barely moves under real divergence except for RF)
holds up using only the six methods that don't touch epinet code at all
(mc_dropout, laplace, the two ensembles, the two from-scratch baselines),
and those results are independently well-supported by the spot-checks in
this review. The fix, if applied (retrain the epinet checkpoint under the
corrected code, rerun `conv_epinet` on both datasets, confirm the qualitative
pattern), is cheap relative to the rest of this session's compute budget —
this is a rerun, not a redesign.

---

## Findings ranked by severity

**Would sink the paper as currently drafted (needs a fix before
submission):**
1. §1.1 — `conv_epinet`'s checkpoint predates the per-example-z fix by four
   days; every `conv_epinet` number in this session's new work reflects an
   eval-time patch on pre-fix-trained weights, not a genuine test of the
   fixed training+inference pipeline. This is the load-bearing method in
   both new datasets and appears in every table in `paper_tables.tex`.

**Needs revision (should be fixed, does not require redoing the whole
effort):**
2. §2.1 / §5 — ECE/NLL at high AliSim branch length are reported and
   ranked (Table 1) without inheriting the accuracy column's explicit
   label-validity hedge, despite depending on the identical
   possibly-invalid label.
3. §3.3 — Spearman rho as the sole headline statistic actively understates
   mc_dropout's real (hump-shaped, 58%-rise-then-decline) epistemic response
   on the AliSim axis, in the direction that favors the paper's "epistemic
   barely moves" framing.
4. §1.3 / §5 — single-seed, single-checkpoint, single-backbone scope is
   disclosed in READMEs but not carried into `paper_tables.tex`'s own prose,
   where superlative claims ("RF shows THE cleanest signal," "I mostly
   can't cleanly separate...with any method tested") are stated without the
   hedge attached at point of claim.
5. §6 — method-roster mismatch (11 vs. 6) between calibration and
   decomposition tables needs a manuscript-facing footnote, not just a
   `.tex`-source comment, if these tables ship as drafted.

**Minor nitpicks:**
6. §3.1/§3.2 — literal `p=0.0` underflow values; no multiple-comparisons
   policy stated anywhere despite ~40+ hypothesis tests run.
7. §3.4 — `evidential`'s two datasets compute structurally different
   "epistemic" quantities, papered over rather than resolved.
8. §4 — RF's Q1→Q4 calibration *worsening* is observed and reported but
   never mechanistically investigated.
9. §6 — ambiguous `T_FITTED`/"seed 42" code comment; hardcoded
   machine-specific absolute paths throughout the newest scripts.

---

## Genuine strengths (an honest review isn't purely negative)

- **The numbers check out.** Every spot-check I ran against source CSVs
  (the `base` row of the boundary-calibration table across all four
  quartiles; the `rf` row of the AliSim-calibration table across all four
  branch lengths; the `ensemble` row of the AliSim-decomposition table,
  including both means and both Spearman rhos) reproduced exactly to the
  reported precision. This is not a project papering over sloppy
  arithmetic — the aggregation and table-generation layer is careful and
  faithful to its inputs.
- **The exact-recovery trick for reconstructing per-example confidence from
  saved entropy** (`invert_binary_entropy` in
  `all_methods_stratified_ece.py`) is a genuinely clever, mathematically
  exact (not approximate) solution to a real data-availability gap: for
  binary classification, the normalized total predictive entropy uniquely
  determines the winning class's mean probability via bisection on the
  binary entropy function, so ECE/NLL/Brier could be recovered for methods
  that only persisted `U_total`/`U_epistemic`/`U_aleatoric` without ever
  re-running inference. This is good, resourceful engineering, correctly
  reasoned through and explicitly justified in-line.
- **Sanity checks are real, not narrative.** The AliSim `ref`-leaf
  reproduction check, the monotonic-identity assertion, and the boundary
  dataset's independent (non-circular) accuracy-by-quartile check are all
  automated and actually gate the pipeline (`verify()` raises on failure),
  not just described as having been eyeballed once.
- **Honest self-reporting of disconfirming results throughout.** Both
  READMEs state plainly, repeatedly, that the "flat aleatoric" half of the
  hypothesis failed for every method tested, including the project's own
  favored `conv_epinet` method — there's no visible pattern of suppressing
  or downplaying inconvenient results, including the method-count
  discrepancy and data-availability gaps that a less careful writeup could
  have quietly glossed over.
- **The roster is genuinely broad and mechanistically diverse** — a
  from-scratch CNN and Random Forest with zero shared pretraining, alongside
  post-hoc Laplace, genuine independently-fine-tuned deep ensembles, epinet,
  MC-dropout, and Dirichlet evidential heads — which is exactly the kind of
  breadth Reviewer 2's original R2-7 baseline complaint asked for, applied
  here to a new question rather than left as an unaddressed gap.
