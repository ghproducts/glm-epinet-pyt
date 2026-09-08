# Aleatoric boundary-sampling test

Does `U_aleatoric` actually rise on genuinely hard/ambiguous examples? This
is a redesign of this project's aleatoric-uncertainty test, replacing
uniform label-flip noise injection (`data_gen/label_noise/`, already run
separately) with **boundary sampling**: a static, non-circular difficulty
score computed directly on the real `promoter_all` test split, with no
synthetic corruption of any kind.

`data_gen/promoter_motifs/` (FIMO/JASPAR motif work) was explicitly **not**
used or referenced anywhere in this design — not as source, not as
methodology, not as a template.

## Method

1. **Margin score.** Run the plain `base` DNABERT2 checkpoint once over the
   real `promoter_all` test split (1584 sequences,
   `datasets.load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", split="test")`
   filtered to `task == "promoter_all"`, via this repo's own
   `nn_proj.common.datasets.load_NT_tasks`). For each sequence,

   ```
   margin = |p1 - 0.5| * 2
   ```

   where `p1` is the softmax probability of class 1. `margin == 0` at the
   decision boundary, `margin == 1` for a maximally confident prediction.
   Sequences are split into quartiles by margin: **Q1** (boundary, hardest)
   through **Q4** (canonical, easiest), using a rank-based `qcut` so every
   quartile has exactly 396 examples regardless of ties.

   The first check is **independent of any method's own uncertainty
   estimate**: accuracy against the *true* label, grouped by margin
   quartile. Accuracy is not a function of the model's own margin score, so
   a monotonic rise from Q1 to Q4 is real evidence the margin score tracks
   genuine example difficulty rather than being circular by construction.

2. **Stochastic-K inference**, all three methods sharing the exact same
   margin-based Q1-Q4 partition (same tokenizer, same `prep_for_trainer`
   call, same un-shuffled `DataLoader`, so row `idx` refers to the same
   sequence in every CSV in this directory):
   - `mc_dropout`: the `base` checkpoint with dropout re-enabled at eval
     (`p=0.1`), K=10 stochastic forward passes.
   - `conv_epinet`: the `epinet` checkpoint, K=10 index samples, using a
     **fixed** `epinet.py` (see below).
   - `evidential`: the `evidential` checkpoint, one forward pass, Dirichlet
     evidential head (Sensoy et al. 2018).

   For each, per-example `U_total`/`U_epistemic`/`U_aleatoric` are computed
   with the same BALD/entropy decomposition used everywhere else in this
   repo (`nn_proj.common.utils.compute_uncertainty`, on the `[K, B, C]`
   logit stack). For `evidential`, which never draws K samples (it directly
   parameterizes a Dirichlet in one forward pass), `U_epistemic` is
   identically 0 by construction — **vacuity** (`num_classes / sum(alpha)`)
   is the epistemic-axis analogue to read instead, matching the convention
   already used in `data_gen/label_noise/decomp_compare_label_noise_evidential.csv`.

3. **Merge and report**: per method, per quartile, mean `U_aleatoric` and
   `U_epistemic`(-analogue), a Mann-Whitney U test comparing Q1 vs Q4 for
   each, and (supplementary) the Spearman rank correlation between the
   continuous margin score and each uncertainty score across all 1584
   examples.

## Hypothesis under test

- `U_aleatoric` should be markedly higher in Q1 (boundary) than Q4
  (canonical): these are the examples that are genuinely hard to call.
- `U_epistemic` should **not** track the margin quartile much, if at all:
  `promoter_all`'s test split is still in-distribution data the checkpoint
  was fine-tuned on — just examples whose *label* is harder to call, not
  examples whose *input* is novel. Epistemic uncertainty is supposed to be
  about novelty of input, not label difficulty. A method that conflates the
  two would show epistemic uncertainty rising in Q1 as well; a method that
  correctly separates the two axes should not.

## Results

### Sanity check: accuracy by margin quartile (independent check)

| quartile | n   | accuracy | mean margin |
|----------|-----|----------|-------------|
| Q1 (boundary)  | 396 | 0.674 | 0.419 |
| Q2       | 396 | 0.874 | 0.863 |
| Q3       | 396 | 0.977 | 0.968 |
| Q4 (canonical) | 396 | 0.992 | 0.989 |

Strongly, cleanly monotonic (0.674 -> 0.874 -> 0.977 -> 0.992). The margin
score is doing exactly what it should: Q1 really is where the base
checkpoint is wrong most often, using ground truth the score itself never
sees.

### Per-method, per-quartile mean U_aleatoric / U_epistemic(-analogue)

| method | score | Q1 (boundary) | Q2 | Q3 | Q4 (canonical) | MWU p (Q1 vs Q4) | Spearman rho vs margin |
|---|---|---:|---:|---:|---:|---:|---:|
| conv_epinet | U_aleatoric | 0.7116 | 0.4021 | 0.1819 | 0.0774 | 4.9e-131 | -0.952 |
| conv_epinet | U_epistemic | 0.1592 | 0.0875 | 0.0370 | 0.0089 | 2.1e-129 | -0.838 |
| mc_dropout  | U_aleatoric | 0.7260 | 0.5406 | 0.2502 | 0.0668 | 5.0e-131 | -0.886 |
| mc_dropout  | U_epistemic | 0.0488 | 0.0379 | 0.0250 | 0.0010 | 5.2e-128 | -0.715 |
| evidential  | U_aleatoric | 0.8785 | 0.6666 | 0.5287 | 0.4905 | 4.9e-131 | -0.910 |
| evidential  | vacuity     | 0.6007 | 0.3663 | 0.2389 | 0.2113 | 4.9e-131 | -0.914 |

(Full table with Q1/Q4 sample sizes and Mann-Whitney U statistics:
`results_summary.csv` / `results_summary.md`. Every p-value uses N=396 per
group.)

### Honest read: confirming and disconfirming evidence

**Confirming**: `U_aleatoric` rises sharply and monotonically from Q4 to Q1
for all three methods (conv_epinet 0.077 -> 0.712, mc_dropout 0.067 ->
0.726, evidential 0.491 -> 0.879), every Q1-vs-Q4 comparison significant at
p < 1e-127. This is the expected result: examples near the decision
boundary really do get flagged as more label-ambiguous by every method
tested.

**Disconfirming**: the "epistemic stays flat" half of the hypothesis
**did not hold for any of the three methods**. `U_epistemic`
(conv_epinet, mc_dropout) and vacuity (evidential) all rise substantially
and significantly from Q4 to Q1 as well — conv_epinet's U_epistemic rises
~17.8x (0.0089 -> 0.1592), mc_dropout's ~47x in relative terms off a small
base (0.0010 -> 0.0488), and evidential's vacuity ~2.8x (0.2113 -> 0.6007).
Every one of these Q1-vs-Q4 differences is significant at p < 1e-127, and
the Spearman correlations between the continuous margin score and the
epistemic-axis score are large in magnitude (-0.72 to -0.91) — not far off
the aleatoric-axis correlations (-0.89 to -0.95) for the same method. By
this test, none of the three methods cleanly separates "this input is
label-ambiguous" from "this input is novel" — both axes respond to margin,
just with aleatoric responding somewhat more than epistemic in most cases
(the gap is real but is a difference of degree, not of kind).

A plausible mechanical reason (not confirmed further here, offered as
interpretation): near the decision boundary the base representation itself
sits closer to the classifier's separating hyperplane, so *any* source of
per-sample perturbation — dropout masks, epinet index-vector jitter — has
a larger chance of flipping the predicted class or shifting predictions
across the K samples, independent of whether the underlying representation
is actually "novel" in the OOD sense. That inflates a spread/disagreement-
based epistemic proxy for structural reasons having nothing to do with
input novelty. This is a known category of confound for
variance/disagreement-based epistemic estimates near a decision boundary,
not something specific to this repo's implementation — but this experiment
does not by itself distinguish that explanation from an implementation gap.

**Bottom line**: the aleatoric axis behaves as expected for all three
methods (rises sharply and specifically on hard-to-label examples). The
epistemic axis is not flat for any of them on this in-distribution,
harder-vs-easier partition — it moves in the same direction as aleatoric,
just less strongly. Report this as it is: a real, method-general
epistemic/aleatoric entanglement near the decision boundary, not a clean
pass for the "epistemic == novelty, blind to label difficulty" hypothesis.

## Extension: full method roster (follow-up pass)

The original pass above covered 3 methods (conv_epinet, mc_dropout, evidential).
This follow-up extends the same margin-quartile partition
(`csv_data/margin_scores.csv`, reused verbatim, **not** recomputed) to the
rest of this project's UQ method roster, adapting working implementations
from `data_gen/label_noise/` (built for the label-noise axis) rather than
writing them from scratch:

| new method       | script/class adapted from                                                  | mechanism |
|---|---|---|
| `laplace`        | `decomp_compare_label_noise_laplace.py` / `nn_proj/models/laplace/laplace_head.py` | post-hoc last-layer diagonal Laplace (Daxberger et al. 2021), K=16 posterior samples |
| `ensemble_k5`    | `decomp_compare_label_noise_ensemble.py`                                    | softmax-averaging 5 independently fine-tuned DNABERT2 `base` checkpoints (seeds 1,2,3,4,42) |
| `ensemble_k3`    | same script, first 3 seeds                                                  | same, K=3 (seeds 1,2,3) |
| `cnn_mc_dropout` | `decomp_compare_label_noise_cnn.py` / `cnn_scratch.py`                      | from-scratch 1D-CNN (DeepBind/DeepSEA-lineage), dropout p=0.3 at eval, K=16 stochastic passes |
| `cnn_ensemble`   | same                                                                        | 5 independently-initialized-and-trained CNNs (seeds 1,2,3,4,42), no shared pretraining at all |
| `rf`             | `decomp_compare_label_noise_rf.py`                                          | 6-mer counts + `RandomForestClassifier` (200 trees), per-tree `predict_proba` as the K "samples" |

All six are run once (no rate/noise axis here -- this test's axis is margin
quartile, not label-noise rate), on the exact same 1584-example
`promoter_all` test set as every other method in this directory, producing
per-example `idx`/`U_epistemic`/`U_aleatoric` rows appended to
`csv_data/uncertainty_by_method.csv` with the same schema as the original 3
methods. Script: `make_boundary_eval_extra_methods.py`.

### Row-order verification (CNN/RF's data source)

`decomp_compare_label_noise_cnn.py`/`_rf.py` read
`data_gen/label_noise/csv_data_r00/{train,test}.csv` directly with `pandas`
rather than going through `load_NT_tasks`. Before reusing that csv's row
order as this test's `idx` (needed so the CNN/RF rows join correctly onto
`margin_scores.csv`'s quartile assignments), this was checked explicitly
rather than assumed:

```python
ds = load_NT_tasks(task="promoter_all", split="test")   # what margin_scores.csv's idx is built from
tdf = pd.read_csv("data_gen/label_noise/csv_data_r00/test.csv")
# sequence-by-sequence and label-by-label comparison, all 1584 rows
```

Result: **1584/1584 exact match**, both `sequence` text and `original_label`.
`csv_data_r00/test.csv` is a byte-faithful, identically-ordered csv snapshot
of the same `promoter_all` test split `make_boundary_eval.py` used -- so
`idx == row index` for the CNN/RF methods, exactly consistent with every
other method's `idx` in this directory. `csv_data_r00/train.csv` (30001
rows, clean 0%-noise labels) is likewise this project's standard
`promoter_all` training split, used for training the CNN and RF and for
fitting the Laplace posterior's GGN.

### Per-method setup and any retraining

- **`laplace`**: post-hoc, **no retraining** -- applied directly to
  `checkpoints/seed_1/DNABERT2/label_noise_r00/base` (the same seed-1
  checkpoint the other DNABERT2-based methods use), per this task's explicit
  instruction. The diagonal GGN (`nn_proj.models.laplace.fit_diagonal_laplace`,
  `classifier_attr="classifier"`, `prior_precision=1.0`) is fit on the first
  2048 examples of `csv_data_r00/train.csv` (`max_examples=2000` cap;
  `fit_diagonal_ggn` stops at the first batch boundary at-or-past the cap,
  landing on 2048 with `batch_size=64` -- the same cap `inference_laplace.py`
  defaults to). K=16 Monte-Carlo posterior samples per test example, same K
  used throughout `data_gen/label_noise/`'s laplace script.
- **`ensemble_k5` / `ensemble_k3`**: **no retraining** -- reuses this
  project's already-fine-tuned seeded `base` checkpoints
  (`checkpoints/seed_{s}/DNABERT2/label_noise_r00/base` for
  s in {1,2,3,4,42}, all five confirmed present on disk). K=5 uses all five
  seeds (this project's standard 5-seed convention, matching the *current*
  `decomp_compare_label_noise_ensemble.py`, which despite its docstring
  still describing a 3-seed design, was edited in-place at some point to use
  all 5 -- SEEDS=[1,2,3,4,42] is what the code on disk actually does); K=3
  uses the first three seeds (1,2,3) as a smaller ensemble for comparison.
  No separate `decomp_compare_label_noise_ensemble_k3.py` script exists on
  disk any more (only its historical output csv survives, predating the
  K3->K5 in-place edit) -- `make_boundary_eval_extra_methods.py`'s
  `run_ensemble()` takes `seeds` as a parameter and is called twice (once
  per K) instead.
- **`cnn_mc_dropout` / `cnn_ensemble`**: **trained fresh** in this script
  (`cnn_scratch.py` defines no saved-checkpoint format -- every
  `data_gen/label_noise/` CNN script retrains from scratch per invocation,
  so this follows the same convention rather than inventing checkpointing).
  `SmallCNN` (one-hot `[B,4,300]` input -> conv(64,k=15) -> maxpool(4) ->
  conv(128,k=9) -> global maxpool -> dropout(0.3) -> fc(64) -> dropout(0.3)
  -> fc(2)), trained with Adam (lr=1e-3, weight_decay=1e-5), 15 epochs,
  batch_size=128, on `csv_data_r00/train.csv` (30000 examples, clean labels)
  -- identical hyperparameters to
  `data_gen/label_noise/decomp_compare_label_noise_cnn.py`. `cnn_mc_dropout`
  trains one model (seed=1), enables dropout at eval, K=16 stochastic
  passes. `cnn_ensemble` trains 5 independently-seeded (1,2,3,4,42) models
  from random initialization each -- no pretraining or weight-sharing at
  all between members, a stronger "genuinely independent" ensemble than the
  DNABERT2 deep ensembles above.
- **`rf`**: **trained fresh** (a `RandomForestClassifier` has no natural
  "checkpoint" either, and `decomp_compare_label_noise_rf.py` retrains per
  invocation). 6-mers via `CountVectorizer(analyzer="char", ngram_range=(6,6))`
  on `csv_data_r00/train.csv`'s sequences (6-mer, matching
  NT_transformer's tokenization convention, same choice as the label-noise
  RF script), `RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
  random_state=0, n_jobs=-1)` -- `min_samples_leaf=20` again matches the
  label-noise script's justification (sklearn's default of 1 grows pure
  leaves, collapsing per-tree entropy to ~0 regardless of true difficulty).
  Per-tree `predict_proba` on the test set is the `[K=200,B,C]` "sample"
  stack fed to `compute_uncertainty`/`compute_uncertainty_variance`.

Both the BALD/entropy decomposition (`U_epistemic`/`U_aleatoric`, same
columns as the original 3 methods) and the variance-based decomposition
(`nn_proj.common.variance_decomp.compute_uncertainty_variance`, this
project's other standard convention) are computed for all six new methods
and stored per-example as extra `var_U_epistemic`/`var_U_aleatoric` columns
in `csv_data/uncertainty_by_method.csv` (blank/NaN for the original 3
methods' rows, which never had this decomposition run). The main combined
`results_summary.csv`/`.md` table below uses the BALD decomposition
throughout (directly comparable to the original 3 methods' numbers); a
separate `results_summary_variance_supplement.csv` reports the
variance-decomposition aggregates for the 6 new methods only.

### Combined results: all 9 method-configurations

(Task instructions describe this as an "8 method-configuration" combined
table in two places, while also explicitly saying to recount the new
methods as 6, not 5. Enumerated honestly: 3 pre-existing
[conv_epinet, mc_dropout, evidential] + 6 new
[laplace, ensemble_k5, ensemble_k3, cnn_mc_dropout, cnn_ensemble, rf] = **9**
method-configurations, not 8. The full 9-method table is reported below
rather than dropping one to force-fit the stated count.)

| method | score | Q1 (boundary) | Q2 | Q3 | Q4 (canonical) | MWU p (Q1 vs Q4) | Spearman rho vs margin |
|---|---|---:|---:|---:|---:|---:|---:|
| conv_epinet | U_aleatoric | 0.7116 | 0.4021 | 0.1819 | 0.0774 | 4.9e-131 | -0.952 |
| conv_epinet | U_epistemic | 0.1592 | 0.0875 | 0.0370 | 0.0089 | 2.1e-129 | -0.838 |
| mc_dropout | U_aleatoric | 0.7260 | 0.5406 | 0.2502 | 0.0668 | 5.0e-131 | -0.886 |
| mc_dropout | U_epistemic | 0.0488 | 0.0379 | 0.0250 | 0.0010 | 5.2e-128 | -0.715 |
| evidential | U_aleatoric | 0.8785 | 0.6666 | 0.5287 | 0.4905 | 4.9e-131 | -0.910 |
| evidential | vacuity | 0.6007 | 0.3663 | 0.2389 | 0.2113 | 4.9e-131 | -0.914 |
| laplace | U_aleatoric | 0.7989 | 0.4078 | 0.1934 | 0.1024 | 4.9e-131 | -0.973 |
| laplace | U_epistemic | 0.0593 | 0.0520 | 0.0465 | 0.0281 | 1.4e-65 | -0.488 |
| ensemble_k5 | U_aleatoric | 0.7971 | 0.3683 | 0.1212 | 0.0533 | 4.9e-131 | -0.986 |
| ensemble_k5 | U_epistemic | 0.0275 | 0.0119 | 0.0011 | 0.0001 | 6.1e-131 | -0.914 |
| ensemble_k3 | U_aleatoric | 0.8045 | 0.3785 | 0.1256 | 0.0538 | 4.9e-131 | -0.986 |
| ensemble_k3 | U_epistemic | 0.0199 | 0.0103 | 0.0010 | 0.0000 | 4.2e-130 | -0.858 |
| cnn_mc_dropout | U_aleatoric | 0.4365 | 0.3466 | 0.1600 | 0.0599 | 3.6e-102 | -0.661 |
| cnn_mc_dropout | U_epistemic | 0.1092 | 0.0874 | 0.0365 | 0.0106 | 5.2e-98 | -0.632 |
| cnn_ensemble | U_aleatoric | 0.4613 | 0.3407 | 0.1404 | 0.0271 | 1.5e-123 | -0.775 |
| cnn_ensemble | U_epistemic | 0.2245 | 0.1493 | 0.0438 | 0.0051 | 1.8e-120 | -0.747 |
| rf | U_aleatoric | 0.7339 | 0.6576 | 0.4950 | 0.2261 | 1.9e-124 | -0.806 |
| rf | U_epistemic | 0.1732 | 0.1669 | 0.1484 | 0.1101 | 9.0e-49 | -0.443 |

(Full table incl. Q1/Q4 n and MWU U statistics: `results_summary.csv`/`.md`.
Every Q1-vs-Q4 p-value uses N=396 per group.)

### Honest read: does the pattern hold, or break, across the full roster?

**It holds -- across all 9 methods, without exception.** Every single
method's `U_epistemic` (or `vacuity` for evidential) rises significantly
from Q4 to Q1 (every Q1-vs-Q4 MWU p < 1e-48, the *weakest* of the 9), the
same conflation the original 3-method pass found. None of the 6 new methods
breaks the pattern; if anything the roster as a whole confirms it more
broadly, across a Bayesian post-hoc method (laplace), two genuine deep
ensembles of independently-trained models (ensemble_k5/k3, cnn_ensemble),
a non-pretrained neural sampling method (cnn_mc_dropout), and a classical
non-neural ensemble (rf) -- five qualitatively different UQ mechanisms,
not just more dropout/index-perturbation variants of the same idea.

That said, the *degree* of conflation varies meaningfully by method, more
than the p-values alone suggest (see the "quartile means and Spearman rho
are the numbers to read for effect size, not p-values" caveat carried over
from the original pass):

- **Worst conflation (relative)**: `ensemble_k5`/`ensemble_k3` -- U_epistemic
  rises ~275x-460x in relative terms (0.0001->0.0275 for k5, 0.00004->0.0199
  for k3), Spearman rho -0.91/-0.86, nearly as strong as their own aleatoric
  correlation (-0.99). Despite being genuine independently-fine-tuned-model
  ensembles (the "textbook" epistemic-uncertainty mechanism), they show the
  *most* margin-driven epistemic inflation of any method tested, in relative
  terms -- consistent with the mechanical explanation in the original pass
  (near the boundary, small per-model differences are far more likely to
  flip which class a given member predicts, inflating spread-based epistemic
  proxies for structural reasons unrelated to input novelty).
- **Consistent with the original 3**: `conv_epinet`, `mc_dropout`,
  `cnn_mc_dropout`, `cnn_ensemble` all show substantial (10x-45x) epistemic
  rises with correlations in the -0.63 to -0.84 range -- squarely in the
  same regime the original pass found for conv_epinet/mc_dropout.
- **Mildest conflation (still real, still significant)**: `laplace` and
  `rf` show the *smallest* relative epistemic rises (~2.1x and ~1.6x) and
  the weakest epistemic-margin correlations of the roster (rho -0.49 and
  -0.44, versus -0.63 to -0.99 for everything else) -- clearly the
  best-separating methods tested, but still not flat: both Q1-vs-Q4
  epistemic differences remain significant (p=1.4e-65 for laplace,
  p=9.0e-49 for RF), and both correlations are still far from zero. Neither
  method passes a strict "epistemic == novelty, blind to margin" test
  either; they are simply less bad at it than the other 7.

**Bottom line, extended**: the full 9-method roster -- spanning DNABERT2 with
4 different UQ mechanisms (conv_epinet, mc_dropout, laplace, deep ensemble),
a Dirichlet evidential head, a from-scratch CNN under 2 sampling mechanisms,
and a classical Random Forest -- confirms rather than overturns the original
finding. `U_aleatoric` cleanly and strongly tracks margin for every method
(Spearman rho -0.66 to -0.99, uniformly strong across all 9). `U_epistemic`
is never flat on this in-distribution, harder-vs-easier partition for *any*
method in this
project's roster -- it always moves with margin too, to a degree ranging
from "nearly as strong as aleatoric" (ensemble_k5/k3) to "much weaker but
still significant" (laplace, rf). This looks like a structural property of
variance/disagreement-based (and Dirichlet-vacuity-based) epistemic proxies
near a decision boundary, not an artifact of any one sampling mechanism,
architecture, or training regime tested here.

## Checkpoint substitution

The original manuscript's `trained_models_*/DNABERT2/promoter_all`
checkpoint no longer exists on disk. This analysis uses
`checkpoints/seed_1/DNABERT2/label_noise_r00/{base,epinet,evidential}`
instead — DNABERT2 fine-tuned / epinet-trained / evidential-trained on
byte-identical `promoter_all` data (`label_noise_r00` = 0% injected noise),
under this repo's current standard pipeline. This is the same substitution
already used and flagged in prior work this session
(`data_gen/label_noise/`).

## Epinet fix used

Before running `conv_epinet` inference, `nn_proj/models/epinet/epinet.py`
in this worktree was overwritten with the fixed version from branch
`worktree-fix-epinet-batch-z` (`git show worktree-fix-epinet-batch-z:nn_proj/models/epinet/epinet.py`).
The stock version's `GaussianIndexer` drew a single `z` shared across the
*entire batch*, so every example's K "posterior draws" were driven by the
same z_s and were not independent of its batch-mates' draws — this directly
undermines getting a meaningful *per-example* epistemic signal, which is
exactly what this test needs. The fixed version draws one `z` per example
(`[B, Dz]` instead of `[Dz]`) and adds a `forward_multi`/`basis`/`contract`
fast path that computes the conv-prior's frozen sub-network outputs once per
batch instead of once per (sample, example) pair. This is the one exception
to this task's "don't touch anything outside `data_gen/aleatoric_boundary/`"
scope, done because it was explicitly required to get a valid per-example
epistemic estimate out of `conv_epinet` at all.

This constraint was given directly by this task's instructions
(pre-diagnosed in-session); it was not independently rediscovered here.

## Environment

DNABERT2 checkpoint loading hits a well-known 3-way version conflict. What
was actually verified to work, empirically, in this sandbox:

- **New virtualenv**: `/scratch/home/glh52/venvs/aleatoric_boundary_venv`,
  Python 3.10.12 (system `/usr/bin/python3.10`, bootstrapped with
  `--without-pip` + `get-pip.py` since `python3-venv`'s `ensurepip` wasn't
  installed and there's no sudo in this sandbox).
- **`transformers==4.30.2`**, not the originally-suggested "4.29-4.35"
  range at large. `4.33.3` was tried first and fails differently than the
  documented `pad_token_id` `AttributeError`: passing a config loaded via
  `trust_remote_code=True` from the *checkpoint directory* registers
  DNABERT2's dynamically-loaded `BertConfig` class under a module name keyed
  to the checkpoint dir's basename
  (`transformers_modules.<basename>.configuration_bert`), which differs from
  the module name used when `AutoModelForSequenceClassification.from_pretrained(HUB_ID, ...)`
  separately resolves the model class from the *hub* id
  (`transformers_modules.zhihan1996.DNABERT-2-117M...configuration_bert`).
  `transformers>=~4.31` added a strict `model_class.config_class ==
  type(config)` check inside `AutoModelForSequenceClassification.register()`
  that this class-identity mismatch fails, even though the config content is
  byte-identical either way. Fix used throughout this script: **always load
  config via `AutoConfig.from_pretrained(HUB_ID, num_labels=2,
  trust_remote_code=True)`**, never from the checkpoint directory — base,
  epinet, and evidential checkpoints all share the same architecture, so
  this loses nothing. `transformers==4.30.2` predates the strict check
  entirely (confirmed empirically: `4.29.2` also predates it, per the
  now-superseded `DNAbert_venv`; `4.30.2` was the first version tried in the
  new Python 3.10 env and it worked, so no further narrowing was done).
  Python 3.10 (required by `nn_proj.models.epinet`'s `X | None` type hints)
  is compatible with 4.29-4.35 in general; the actual constraint that ruled
  out most of that range in practice was the config-registration bug above,
  not a Python-version issue.
- **`triton` uninstalled from the venv.** DNABERT2's remote code
  (`bert_layers.py`) tries `from .flash_attn_triton import
  flash_attn_qkvpacked_func` and, if that import succeeds, always takes the
  Triton flash-attention path. On this Python 3.10 + torch 2.6.0 + CUDA
  environment that import *does* succeed (unlike the old `DNAbert_venv`,
  where it silently failed and fell back to a plain PyTorch attention
  implementation) — but DNABERT2's triton kernel then hit
  `assert q.is_cuda and k.is_cuda and v.is_cuda` even with the model and
  inputs correctly on GPU (a real bug in the pinned kernel, not this
  script). Uninstalling `triton` from the venv makes the `try/except
  ImportError` in `bert_layers.py` fall back to the plain PyTorch attention
  path deterministically, which is what actually gets used throughout this
  analysis. This was not mentioned in the task's original environment notes
  and was found empirically.
- Tokenizer loaded from the hub id `zhihan1996/DNABERT-2-117M` directly
  (never from a checkpoint dir — its `tokenizer.json` was serialized by a
  newer `tokenizers` than this env's older `transformers` can parse), with
  `model_max_length=75` (DNABERT2 `tokens_per_base=0.25` x 300bp) and
  `tokenizer.eos_token = tokenizer.pad_token`.
- The `evidential` checkpoint's saved state dict keys are prefixed
  `wrapper.base.*` (it was saved from an `HFEvidentialSeqClassifier` whose
  `.wrapper` attribute is an `EvidentialWrapper` wrapping `.base`, the plain
  `AutoModelForSequenceClassification`). `nn_proj/models/evidential/` is not
  present in this worktree (untracked in the main checkout, and out of this
  task's file-scope), so `make_boundary_eval.py` reimplements the handful of
  formulas needed to score an already-trained checkpoint (evidence =
  softplus(logits), alpha = evidence + 1, vacuity = num_classes / sum(alpha))
  in a small local class hierarchy that mirrors that attribute layout
  exactly, so the checkpoint's `strict=True` state-dict load succeeds.
  `nn_proj.common.utils` and `nn_proj.common.datasets`, and
  `nn_proj.models.epinet` (with the fix above), import and run without any
  Python-version friction under this environment, so they are imported
  directly rather than reimplemented.
- Checkpoint paths are referenced by absolute path
  (`/scratch/home/glh52/glm-epinet-pyt/checkpoints/...`), not repo-relative:
  `checkpoints/` is gitignored and this script runs from inside a git
  worktree that does not itself contain that directory — only the primary
  checkout does, on the same shared filesystem.

Run with:

```bash
PYTHONPATH=. /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \
    data_gen/aleatoric_boundary/make_boundary_eval.py
```

from the repo root (any worktree). Total runtime ~1 minute on an idle A100.

### Follow-up pass environment note

The same `aleatoric_boundary_venv` was reused as-is for
`make_boundary_eval_extra_methods.py` — no new packages were needed
(`scikit-learn` and `scipy` were already present and used successfully by
the `rf` and Mann-Whitney code paths). The same "load DNABERT2 config from
`HUB_ID`, never from a checkpoint dir" and "absolute `checkpoints/` path"
workarounds apply identically to the `laplace`/`ensemble_k5`/`ensemble_k3`
methods, since they load the same DNABERT2 architecture. Wall-clock: on
this run, laplace (GGN fit + K=16 sampling) finished in well under a
minute; the two DNABERT2 ensembles (5+3 model forward passes over 1584
examples) and the CNN trainings (6 fresh `SmallCNN`s, 15 epochs each) were
the dominant cost; the k-mer `RandomForestClassifier` fit (`n_jobs=-1`,
200 trees, unlimited depth apart from `min_samples_leaf=20`) was the
single slowest step, taking several minutes under heavy contention for CPU
cores from other jobs on this shared host at run time (confirmed via `ps`/
`top`: the RF fit was genuinely CPU-bound and multi-threaded — using
sklearn's threading backend, not a fork-based one, ruling out a
CUDA-context/fork deadlock as the cause of the wait). Total wall-clock for
all 6 new methods: **under 10 minutes** end to end on a shared, contended
node.

Run with:

```bash
PYTHONPATH=. /scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python \
    data_gen/aleatoric_boundary/make_boundary_eval_extra_methods.py
```

from the repo root (any worktree, after `make_boundary_eval.py` has already
produced `csv_data/margin_scores.csv` and the original 3-method
`csv_data/uncertainty_by_method.csv` — this script reads and extends both).

## Files

- `make_boundary_eval.py` — the reproducible script (deterministic seeding,
  a sanity-check function, house style matching
  `data_gen/label_noise/decomp_compare_label_noise.py`).
- `csv_data/margin_scores.csv` — per-example `idx`, `label`, `pred_base`,
  `prob_class1`, `margin`, `quartile`.
- `csv_data/uncertainty_by_method.csv` — per-example `idx`, `method`,
  `label`, `pred`, `U_total`, `U_epistemic`, `U_aleatoric`, `vote_pct`,
  `margin`, `quartile` (+ `vacuity`/`dirichlet_strength` for `evidential`).
- `results_summary.csv` / `results_summary.md` — the per-quartile
  aggregated table, Mann-Whitney U p-values (Q1 vs Q4), and supplementary
  Spearman correlations, for every method x score (now all 9, after the
  follow-up pass below).
- `make_boundary_eval_extra_methods.py` — follow-up script adding the 6 new
  methods (laplace, ensemble_k5, ensemble_k3, cnn_mc_dropout, cnn_ensemble,
  rf); reuses `make_boundary_eval.py`'s `sanity_check_margin_quartiles`/
  `summarize`/`write_results_summary_md` via import rather than duplicating
  them, and *extends* `csv_data/uncertainty_by_method.csv` and
  `results_summary.csv`/`.md` in place (reads the existing 3-method csv,
  concatenates the 6 new methods' rows, rewrites both files with all 9).
- `results_summary_variance_supplement.csv` — per-quartile
  `var_U_aleatoric`/`var_U_epistemic` means and Q1-vs-Q4 Mann-Whitney U, for
  the 6 new methods only (the variance decomposition was not computed for
  the original 3 methods' rows, so there is nothing to add there).

## Judgment calls / limitations

- **No entry added to `configs/experiments.yaml` / `configs/tasks.yaml`.**
  Those files define the formal training/eval grid that `scripts/run_grid.py`
  and `nn_proj/analysis/tasks.py` consume; this is a one-off diagnostic
  analysis, not a new training task or test pair in that grid — the same
  posture `data_gen/label_noise/` (the closest prior art, itself a multi-rate,
  multi-seed, multi-method analysis suite) already takes by leaving those
  files untouched. Judgment call: adding a stub entry there for a script that
  isn't invoked by `run_grid.py` would misrepresent this as part of the
  formal grid, so it was left out.
- Single seed (seed 1) for every method, matching the checkpoints named in
  this task's checkpoint substitution; no seed-variance estimate the way
  `decomp_compare_label_noise.py`'s 3-seed design gets one for conv_epinet
  elsewhere in this repo.
- `evidential`'s `U_epistemic` is identically 0 by construction (single
  forward pass, no K samples) — vacuity is reported as its epistemic-axis
  analogue throughout, and is *not* directly comparable in scale to the
  other two methods' `U_epistemic`, only to itself across quartile.
- The Q1-vs-Q4 Mann-Whitney tests use N=396 per group; at that sample size,
  even modest effect sizes are statistically significant, so the p-values
  alone don't distinguish "epistemic moves a little" from "epistemic moves a
  lot" — the quartile means and the supplementary Spearman correlations in
  `results_summary.md` are the numbers to read for effect size, not the
  p-values in isolation.
- This test only establishes correlation between margin and each
  uncertainty axis on one task (`promoter_all`), one backbone (DNABERT2),
  one seed. It does not establish *why* epistemic uncertainty tracks margin
  here (see the mechanical explanation offered above, which is plausible but
  untested by this script) or whether the same entanglement would appear on
  a different task/backbone.

### Follow-up pass (6 new methods) judgment calls

- **Worktree file gaps, filled by plain `cp`, not `git show`.** This
  worktree branched from `main`'s last *commit*; several files this pass
  needed are untracked in the main checkout (never committed anywhere:
  `nn_proj/models/laplace/`, `nn_proj/common/variance_decomp.py`,
  everything under `data_gen/label_noise/`), so `git show <ref>:<path>`
  cannot retrieve them (there is no commit that contains them). They were
  copied byte-for-byte with plain `cp` from the shared main checkout
  (`/scratch/home/glh52/glm-epinet-pyt/...`, same filesystem, not a git
  operation) into this worktree: `nn_proj/models/laplace/{__init__.py,
  laplace_head.py}`, `nn_proj/common/variance_decomp.py`, and from
  `data_gen/label_noise/`: `cnn_scratch.py`,
  `decomp_compare_label_noise_{laplace,ensemble,cnn,rf,mcdropout,evidential}.py`,
  `decomp_compare_label_noise.py`, and the `csv_data_r00/` directory
  (train+test csvs). None of this content was reconstructed from memory;
  every copied file was read from disk first.
- **Method count: 9, not 8.** The task instructions describe the combined
  table as spanning "8 method-configurations" (stated twice), while also
  explicitly saying to recount the new methods as 6, not 5. Adding the 3
  pre-existing methods (conv_epinet, mc_dropout, evidential) to the 6 new
  ones gives 9. Rather than dropping one method to force-fit the stated
  "8", all 9 are reported, and the discrepancy is flagged here plainly.
- **CNN/RF row-order reuse, verified not assumed.** Before treating
  `data_gen/label_noise/csv_data_r00/test.csv`'s row order as identical to
  `load_NT_tasks(task="promoter_all", split="test")`'s (needed for the
  CNN/RF rows' `idx` to line up with `margin_scores.csv`'s quartiles), an
  explicit full sequence-and-label comparison was run (see "Row-order
  verification" above) rather than assumed from the shared task name. It
  matched exactly, 1584/1584.
- **`ensemble_k3` has no standalone script on disk.** Only
  `decomp_compare_label_noise_ensemble.py` (current: 5 seeds) exists;
  `decomp_compare_label_noise_ensemble_k3.csv`'s historical output survives
  from before that script was edited in-place from a 3-seed to a 5-seed
  design, but the 3-seed script version itself does not. `run_ensemble()`
  in `make_boundary_eval_extra_methods.py` is parameterized by `seeds` and
  called twice (`[1,2,3,4,42]` and `[1,2,3]`) rather than trying to recover
  or guess at a lost script version.
- **`epinet.py` left untouched in this pass.** The per-example-`z` fix from
  `worktree-fix-epinet-batch-z` was *not* re-applied here: none of the 6 new
  methods touch `nn_proj.models.epinet`, and the existing `conv_epinet`
  numbers in the combined table are reused verbatim from the original pass
  (already computed under the fixed version there — see "Epinet fix used"
  above). Confirmed by diff that this worktree's checked-out
  `nn_proj/models/epinet/epinet.py` still differs from the fixed version;
  left as-is since nothing in this pass depends on it.
- Both BALD and variance decompositions were computed for all 6 new
  methods (the task's "var too if cheap to add" -- it was, both come from
  the same `[K,B,C]` logit stack already computed for BALD). The primary
  combined table still reports only BALD, to stay directly comparable to
  the original 3 methods (which never had variance decomposition computed);
  the variance numbers are reported separately in
  `results_summary_variance_supplement.csv` rather than silently mixed into
  the main comparison table.
- Same single-seed-1 posture as the original pass: `laplace` uses only the
  seed-1 base checkpoint (per this task's explicit instruction), and the
  CNN/RF are each trained with a single call per sampling variant (`cnn_
  mc_dropout` uses one seed-1 CNN; the ensembles use seeds 1,2,3,4,42/1,2,3
  as ensemble *members*, not as repeated independent trials of the same
  experiment) -- no cross-seed variance estimate on top of what each
  method's own K-sample decomposition already provides.
