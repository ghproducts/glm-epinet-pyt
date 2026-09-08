# promoter_alisim: a continuous evolutionary-distance OOD severity axis

This dataset provides a **principled, continuously controlled epistemic/OOD
severity axis** for the `promoter_all` task, built by simulating sequence
evolution away from real anchor sequences with
[AliSim](https://academic.oup.com/mbe/article/39/5/msac092/6577219)
(Ly-Trinh & Minh, MBE 2022), the phylogenetic sequence simulator bundled with
[IQ-TREE2](https://github.com/iqtree/iqtree2).

It replaces the ad hoc `shuffled`/`random_dna` character-corruption approach
used elsewhere in this project's `data_gen/label_noise/` pilot, which
conflates order-novelty and composition-novelty in an uncontrolled way. Here,
"distance from the training distribution" is a single, interpretable,
continuously varying quantity: expected substitutions per site since a real
anchor sequence.

This dataset is **entirely independent of `data_gen/promoter_motifs/`**
(the FIMO/JASPAR motif-annotation pipeline) — it does not reuse that code,
data, or design in any way. It is built solely from `promoter_all` real
sequences plus phylogenetic simulation.

## Tool used: AliSim (IQ-TREE2), not the Pyvolve fallback

**AliSim was used successfully; the Pyvolve fallback was not needed.**

- IQ-TREE2 was already installed at
  `/scratch/home/glh52/tools/iqtree-2.4.0-Linux-intel/bin/iqtree2`
  (version: `IQ-TREE multicore version 2.4.0 for Linux x86 64-bit built Feb 7 2025`).
  `make_alisim_data.py` looks for a binary in this order: `--iqtree-bin` CLI
  flag, `$ALISIM_IQTREE_BIN` env var, this known path, then `iqtree2`/`iqtree`
  on `$PATH`.
- The documented flag for fixing a real root sequence is
  `--root-seq FILE,SEQ_NAME` (comma-separated, no space — confirmed directly
  from `iqtree2 --alisim -h`, which is more authoritative than secondhand
  forum syntax). This was verified end-to-end: a branch length of exactly
  `0.0` from the root reproduces the anchor sequence **character-for-character**
  every single time (this is enforced as an automated invariant — see
  "Pilot / sanity checks" below), and small branch lengths introduce only a
  small number of substitutions, growing smoothly with branch length. This
  took well under the ~30-45 minute troubleshooting budget in the task spec
  (a `--root-seq` syntax variant had already been correctly identified in a
  prior scratch exploration at `/tmp/alisim_test/`, which this implementation
  independently re-verified from the tool's own `-h` output rather than
  trusting the scratch files as ground truth).

## Model: HKY + Gamma(4)

Substitution model string passed to AliSim (example, actual frequencies
computed at run time and saved to `csv_data/promoter_alisim_params.json`):

```
HKY{2.0}+F{0.246065/0.24993/0.255971/0.248034}+G4{1.0}
```

| Parameter | Value | Source |
|---|---|---|
| Base frequencies (A/C/G/T) | `0.246065 / 0.249930 / 0.255971 / 0.248034` | **Empirically estimated**: pooled nucleotide counts across all 30,000 sequences (9,000,000 bases) in the `promoter_all` **train** split. This is a legitimate corpus-derived statistic; no orthology assumption involved. |
| kappa (transition/transversion ratio) | `2.0` | **Literature default** for mammalian genomic DNA. NOT fit to this corpus. |
| alpha (Gamma-4 shape) | `1.0` | **Literature default**, a commonly used moderate-heterogeneity value. NOT fit to this corpus. |

**Why kappa/alpha were not fit to `promoter_all` directly:** `promoter_all`
sequences are unrelated (non-orthologous) human promoter loci — they are not
aligned homologs across species or individuals. Running IQ-TREE's
ModelFinder (or any alignment-based model-fitting procedure) on this corpus
would require treating unrelated sequences as if they were a real multiple
sequence alignment descending from a common ancestor, which is a category
error. Base frequencies are a zeroth-order compositional statistic that does
not require this assumption (any bag of DNA has well-defined per-base
frequencies), so those alone are corpus-derived; kappa and alpha, which
describe evolutionary *process* rather than raw composition, are taken from
the literature and documented plainly as defaults, not fitted values.

## Root sequence: real anchors, not model-drawn

Every simulation's root is the literal real anchor sequence, passed via
`--root-seq <anchor.fasta>,anchor`. Descendants are evolved starting from
this real sequence — they are never redrawn from the model's stationary
distribution.

## Anchors

- **Source split**: `promoter_all` **test** split, via
  `nn_proj.common.datasets.load_NT_tasks(task="promoter_all", split="test")`
  (HuggingFace `InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`,
  config `promoter_all`). 1584 rows, balanced 792 positive / 792 negative.
- **Sampling**: stratified random sample of 200 positive-label and 200
  negative-label sequences (400 total), fixed seed **20260907**
  (`random.Random(20260907)`, shuffle-then-take-first-N per label class).
- **Deduplication note (implementation judgment call)**: the upstream
  `promoter_all` test split contains a small number of exact duplicate rows
  (identical `name`, sequence, and label — 13 of 1584 rows, an upstream data
  artifact, not introduced by this pipeline). `sample_anchors()`
  deduplicates by `name` before sampling so all 400 sampled anchors are
  guaranteed to be distinct sequences (an early run without this dedup step
  happened to draw one duplicate pair, yielding only 399 distinct anchors;
  this was caught by inspection and fixed).
- `anchor_id` in the output CSV is the original dataset's `name` field
  (e.g. `chr20:3148108-3148408|1`), carried through as a stable identifier.

## Tree / branch-length grid

For each anchor, one `iqtree2 --alisim` call simulates a **star tree**: 9
leaves attached directly to the root (which AliSim seeds with the real
anchor sequence via `--root-seq`):

```
(t0:0.0, t1:0.02, t2:0.05, t3:0.1, t4:0.2, t5:0.4, t6:0.8, t7:1.5, ref:0.0);
```

- `t0..t7` correspond to the fixed branch-length grid (substitutions/site):
  **{0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5}**. This is mathematically
  identical to running 8 independent two-taxon (root -> descendant)
  simulations per anchor — a star tree's leaves each evolve independently
  from the shared root — batched into a single process call per anchor for
  efficiency (400 `iqtree2` invocations total instead of 3200).
- `ref` is an extra leaf pinned at branch length `0.0`, included in *every*
  run as a per-simulation sanity check: it must always come back identical
  to the anchor. This is checked automatically and raises an error if it
  ever fails (see `run_alisim_for_anchor` / `generate_rows_for_anchors` in
  `make_alisim_data.py`).
- The grid spans "nearly identical" (`t=0.0`, exact reproduction) to "well
  past typical divergence" (`t=1.5`, mean realized identity ~0.50, close to
  the ~0.25 expected under a uniform stationary distribution at full
  saturation). This is the continuous severity axis the dataset exists to
  provide.
- Per-anchor IQ-TREE seed: `20260907 + anchor_index` (deterministic,
  reproducible; anchor_index is the anchor's position in the sampled list,
  0-399).

No indels are simulated in this first pass (substitutions only, same length
as the anchor throughout) — **documented as a future extension** were
indel-driven length variation ever needed as an additional severity axis.

## Output

`data_gen/promoter_alisim/csv_data/promoter_alisim.csv` — 3200 rows (400
anchors x 8 branch lengths), columns:

| Column | Description |
|---|---|
| `sequence` | Simulated descendant sequence (same length as its anchor, ACGT only). |
| `labels` | Same as `original_label` — included so the file loads directly via `nn_proj.common.datasets.load_local_dataset` (which requires a `label`/`labels` column). **Do not treat this as a validated label** at large branch lengths — see Limitations. |
| `anchor_id` | The real anchor's `name` field from `promoter_all` (e.g. `chr20:3148108-3148408\|1`). |
| `original_label` | The anchor's real `promoter_all` label (0/1), carried through for reference / accuracy-collapse tracking only. |
| `branch_length` | Simulated evolutionary distance from the anchor, in substitutions/site (one of the 8 grid values). |
| `realized_identity_to_anchor` | Fraction of positions identical to the anchor, computed directly by exact character comparison (not read from AliSim's internal logs). |

`data_gen/promoter_alisim/csv_data/promoter_alisim_params.json` records the
exact run parameters (seed, branch-length grid, kappa, alpha, empirical base
frequencies, IQ-TREE binary path and version) for provenance.

`data_gen/promoter_alisim/pilot/` contains a 5-anchor x 8-branch-length (40
row) pilot run (`pilot_alisim.csv` + `pilot_alisim_params.json`), plus the
raw per-anchor AliSim scratch files (`iqtree_work/anchor_XXXX/`: root FASTA,
star tree, IQ-TREE log, and simulated FASTA) kept for manual inspection. The
full 3200-row run's scratch files are not kept (they are deterministically
reproducible from the same seed).

## Pilot sanity-check results

Ran first (5 anchors x 8 branch lengths = 40 sequences) before the full grid,
per the task's fallback/verification procedure. All checks in `verify()`
passed:

- ACGT-only alphabet: yes, for all 40 sequences.
- Sequence length: every simulated sequence matches its anchor's length
  (300 bp, `promoter_all`'s fixed sequence length).
- `branch_length == 0.0` rows: exactly identical to the anchor
  (`realized_identity_to_anchor == 1.0`), for every anchor.
- Identity vs. branch length tracks sensibly (mean identity across the 5
  pilot anchors, monotonically decreasing):

  | branch_length | mean realized_identity_to_anchor |
  |---|---|
  | 0.00 | 1.000 |
  | 0.02 | 0.979 |
  | 0.05 | 0.958 |
  | 0.10 | 0.921 |
  | 0.20 | 0.854 |
  | 0.40 | 0.712 |
  | 0.80 | 0.607 |
  | 1.50 | 0.485 |

The full 3200-row run reproduces essentially the same curve (see
`csv_data/promoter_alisim.csv`; mean identity by branch length: 1.000,
0.981, 0.954, 0.912, 0.843, 0.736, 0.609, 0.501).

## Reproducing

Run with a Python environment that has `datasets`, `transformers`, `numpy`,
and `pandas` installed — this was originally run with the repo's
`glm_epinet_venv` (`/scratch/home/glh52/venvs/glm_epinet_venv`, `datasets`
4.0.0 / `transformers` 4.55.2). Requires the IQ-TREE2 binary described above
(auto-detected, or override with `--iqtree-bin` / `$ALISIM_IQTREE_BIN`).

```bash
# small pilot (5 anchors x 8 branch lengths = 40 rows)
python data_gen/promoter_alisim/make_alisim_data.py --pilot

# full grid (400 anchors x 8 branch lengths = 3200 rows)
python data_gen/promoter_alisim/make_alisim_data.py
```

Both modes run `verify()` automatically and raise an `AssertionError` /
`RuntimeError` if any sanity check fails (alphabet, length, exact
`t=0.0`/`ref` reproduction, or non-monotonic mean identity).

## Limitations

- **No alignment-based model fitting.** kappa and alpha are literature
  defaults, not fit to `promoter_all`, because its sequences are
  non-orthologous and fitting a phylogenetic substitution model to them via
  a tool like ModelFinder would misapply an alignment/common-ancestry
  assumption that does not hold here. Only base frequencies (which require
  no such assumption) are corpus-derived.
- **No indels.** Substitutions only; every simulated sequence is exactly the
  same length as its anchor. Indel simulation (AliSim supports this via
  `--indel-size`/`--indel`) is a natural future extension if variable-length
  OOD severity is ever needed.
- **Label validity is not asserted at large branch lengths.** `labels` /
  `original_label` simply carries through the anchor's real `promoter_all`
  label for reference and for tracking how model accuracy collapses as
  branch length increases. At large `t` (e.g. 0.8, 1.5) the simulated
  sequence has diverged far enough that there is no claim its true
  regulatory/promoter status still matches the original label — this
  dataset is meant for **epistemic-uncertainty (OOD-detection) evaluation**,
  not as a validated classification benchmark at high severity.
- **A single fixed kappa/alpha for all anchors and branch lengths.** Real
  mammalian genomic sequence has rate heterogeneity across loci that a
  single global kappa/alpha does not capture; this is a deliberate
  simplicity/reproducibility tradeoff consistent with the task's design
  spec, not an oversight.
- **`realized_identity_to_anchor` measures raw character identity**, not a
  model-based evolutionary distance estimate — it is a direct, assumption-free
  empirical readout, by design (see task's output-schema requirement), and
  will not exactly equal the nominal `branch_length` for any individual
  sequence (branch length is an expected number of substitutions/site).

## Wiring into the experiment grid

Added as a `tests:` entry under the existing `promoter_all` training task in
`configs/experiments.yaml`, **not** a new standalone training task:

```yaml
promoter_all:
  ...
  tests:
    promoter_all: InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/promoter_all
    enhancers: InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/enhancers
    promoter_alisim: data_gen/promoter_alisim/csv_data/promoter_alisim.csv
```

**Rationale (judgment call):** this dataset is an eval-only, graded
OOD-severity axis for models already trained on `promoter_all` — it shares
`promoter_all`'s label space and was built to *evaluate* uncertainty
calibration as a function of `branch_length`, not to train a new model. It
therefore fits the existing `tests:` mechanism (one CSV, sliced by
`branch_length` during analysis) rather than warranting its own
`training_tasks:` entry, which would imply a distinct training recipe/labels
that don't exist here.

## Dense grid follow-up (16-point branch-length axis)

The original 8-point grid (above) showed a weak, non-monotonic
`U_epistemic` signal for `mc_dropout`/`evidential` (peaking around
branch_length 0.2-0.4, declining toward 1.5) -- too coarse to tell whether
that was real saturation or noise. `make_alisim_data_dense.py` reruns the
identical anchor sampling and AliSim simulation procedure (same 400 anchors,
seed 20260907, same HKY+F+G4 model) over a denser
**16-point branch-length grid**:
`{0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5,
2.0, 3.0}` -- adding resolution in the 0.1-1.0 region where the
non-monotonic behavior appeared, and extending to 2.0/3.0 to check for a
saturation plateau. Output: `csv_data/promoter_alisim_dense.csv` (400
anchors x 16 branch lengths = 6400 rows, same columns as the 8-point CSV
above) + `csv_data/promoter_alisim_dense_params.json`.

## Uncertainty evaluation

`data_gen/promoter_alisim/uncertainty_eval/` runs three UQ methods
(`mc_dropout`, `conv_epinet`, `evidential`) over the full 3200-row grid and
asks the question this dataset exists to answer: does `U_epistemic` rise
with branch length, while `U_aleatoric` stays comparatively flat (real
evolutionary divergence should read as *novelty*, not *label ambiguity*)?
Contents:

- `run_uncertainty_eval.py` -- the reproducible script.
- `per_example_uncertainty.csv` -- all 9600 rows (3200 x 3 methods):
  `anchor_id, branch_length, realized_identity_to_anchor, original_label,
  method, U_aleatoric, U_epistemic, pred, labels, correct`.
- `results_summary.md` -- the dose-response table, Spearman correlations,
  and endpoint Mann-Whitney tests below, plus `results_summary_dose_response.csv`
  / `results_summary_spearman.csv` / `results_summary_mwu_endpoints.csv` (the
  same three tables, split out for machine reading).

**Checkpoints (judgment call, flagged per task spec):** all three methods
reuse `checkpoints/seed_1/DNABERT2/label_noise_r00/{base,epinet,evidential}`
-- DNABERT2 fine-tuned/epinet-trained/evidential-trained on the real
`promoter_all` task (0% injected label noise). This is the *same* checkpoint
family the AliSim anchors themselves were sampled from the test split of, so
it is the natural, consistent choice; no new checkpoints were trained for
this eval.

**Epinet fix used:** `nn_proj/models/epinet/epinet.py` was replaced
wholesale with the version from branch `worktree-fix-epinet-batch-z`
(`git show worktree-fix-epinet-batch-z:nn_proj/models/epinet/epinet.py`)
before running `conv_epinet`. The critical fix is `GaussianIndexer` drawing
one `z` sample per batch *example* instead of one `z` shared across the
whole batch -- without it, every one of the K "posterior draws" for a given
sample index would be identical across a batch's examples, collapsing
exactly the per-example epistemic spread this eval measures. (That branch's
version also carries a few unrelated speedups/fixes -- e.g. `forward_multi`
computing the conv prior's frozen basis once per batch instead of once per
sample, and `predict()` accepting a caller-supplied `metadata_cols` -- taken
as-is since the task specified using that branch's file wholesale.)

**Environment note (real 3-way version conflict, resolved empirically):**
DNABERT2's remote code (`bert_layers.py`) imports the *real*
`transformers.models.bert.modeling_bert.BertPreTrainedModel` as its base
class rather than a local one, so its `config_class` attribute is the
built-in `BertConfig`, not DNABERT2's own `configuration_bert.BertConfig`.
Newer `transformers` (empirically, 4.33.3 -- confirmed by testing, not just
inferred) added a consistency check inside
`AutoModelForSequenceClassification.from_pretrained` that raises `ValueError:
The model class you are passing has a config_class attribute that is not
consistent with the config class you passed` for exactly this mismatch.
Older `transformers` (empirically, 4.29.2, matching what
`/scratch/home/glh52/venvs/DNAbert_venv` already had pinned, Python 3.8)
does not have this check and loads fine. Since the `nn_proj.models.epinet`
package needs Python >= 3.10 (`feature_fns.py`'s `torch.Tensor | None`
annotation is evaluated eagerly -- no `from __future__ import annotations`
in that file -- so it hard-fails to parse on 3.8), the working combination
had to satisfy both constraints at once:
**`/scratch/home/glh52/venvs/aleatoric_boundary_venv` (Python 3.10.12),
with `transformers` pinned down from its stock 4.33.3 to 4.29.2** (the task
brief suggested trying "4.29-4.35"; 4.29.2 specifically is the version
verified end-to-end here -- a full forward pass through `base`, `epinet`,
and `evidential` checkpoints was run before committing to this environment,
per the task's own verification requirement). `scikit-learn` and `tabulate`
were also `pip install`ed into that venv (needed transitively by
`nn_proj.common.utils` and for `DataFrame.to_markdown`, respectively).

**Other judgment calls:**
- The AliSim CSV and its own README (`data_gen/promoter_alisim/README.md`,
  this file) were generated by a sibling agent in a different git worktree
  earlier in the same session; worktrees don't share untracked files with
  each other, so both were copied into this worktree before running the
  eval. Likewise `nn_proj/models/evidential/` (untracked, already built
  earlier in the session in the main checkout) was copied into this
  worktree's `nn_proj/models/` so `evidential` inference could reuse it
  as-is rather than reimplementing it.
- `checkpoints/` is itself gitignored local-run-artifact storage that only
  physically exists under the main repo checkout, not under any worktree --
  `run_uncertainty_eval.py` therefore references it via an absolute path
  (`/scratch/home/glh52/glm-epinet-pyt/checkpoints/...`) rather than a
  path relative to the script.
- K=10 index/dropout samples for `mc_dropout` and `conv_epinet` (task spec).
  `evidential` draws no samples (single Dirichlet forward pass); per its own
  `predict_evidential` docstring and
  `data_gen/label_noise/decomp_compare_label_noise_evidential.py`'s
  convention, **vacuity** (`num_classes / sum(alpha)`) is read as its
  epistemic-analogue column below, not its (identically-zero, K=1)
  `U_epistemic`.

### Results

**Dose-response (mean U_epistemic / U_aleatoric / accuracy by branch_length, n=400/cell):**

| method | branch_length | U_epistemic | U_aleatoric | accuracy |
|---|---:|---:|---:|---:|
| conv_epinet | 0.00 | 0.0660 | 0.3297 | 0.8725 |
| conv_epinet | 0.02 | 0.0644 | 0.3336 | 0.8650 |
| conv_epinet | 0.05 | 0.0659 | 0.3305 | 0.8875 |
| conv_epinet | 0.10 | 0.0687 | 0.3533 | 0.8500 |
| conv_epinet | 0.20 | 0.0783 | 0.3740 | 0.8225 |
| conv_epinet | 0.40 | 0.0800 | 0.4096 | 0.7125 |
| conv_epinet | 0.80 | 0.0866 | 0.4225 | 0.6175 |
| conv_epinet | 1.50 | 0.0847 | 0.4206 | 0.5575 |
| evidential (vacuity) | 0.00 | 0.3534 | 0.6374 | 0.8850 |
| evidential (vacuity) | 0.02 | 0.3549 | 0.6386 | 0.8775 |
| evidential (vacuity) | 0.05 | 0.3451 | 0.6297 | 0.8875 |
| evidential (vacuity) | 0.10 | 0.3597 | 0.6436 | 0.8600 |
| evidential (vacuity) | 0.20 | 0.3781 | 0.6622 | 0.7950 |
| evidential (vacuity) | 0.40 | 0.3877 | 0.6704 | 0.6825 |
| evidential (vacuity) | 0.80 | 0.3697 | 0.6534 | 0.5750 |
| evidential (vacuity) | 1.50 | 0.3381 | 0.6277 | 0.5375 |
| mc_dropout | 0.00 | 0.0251 | 0.3841 | 0.8675 |
| mc_dropout | 0.02 | 0.0282 | 0.3873 | 0.8625 |
| mc_dropout | 0.05 | 0.0304 | 0.3834 | 0.8575 |
| mc_dropout | 0.10 | 0.0345 | 0.4176 | 0.8225 |
| mc_dropout | 0.20 | 0.0450 | 0.4286 | 0.7325 |
| mc_dropout | 0.40 | 0.0375 | 0.3714 | 0.5975 |
| mc_dropout | 0.80 | 0.0315 | 0.3140 | 0.5350 |
| mc_dropout | 1.50 | 0.0268 | 0.2794 | 0.5075 |

**Spearman (branch_length, continuous, vs. score; n=3200):**

| method | score | rho | p |
|---|---|---:|---:|
| conv_epinet | U_epistemic | 0.108 | 7.9e-10 |
| conv_epinet | U_aleatoric | 0.127 | 5.1e-13 |
| evidential | vacuity | 0.026 | 0.138 (n.s.) |
| evidential | U_aleatoric | 0.019 | 0.275 (n.s.) |
| mc_dropout | U_epistemic | 0.055 | 0.0020 |
| mc_dropout | U_aleatoric | -0.055 | 0.0020 |

**Endpoint check (Mann-Whitney U, t=1.5 > t=0.0, one-sided):**

| method | score | p | mean(t=0) | mean(t=1.5) |
|---|---|---:|---:|---:|
| conv_epinet | U_epistemic | 2.3e-05 | 0.0660 | 0.0847 |
| conv_epinet | U_aleatoric | 8.8e-08 | 0.3297 | 0.4206 |
| evidential | vacuity | 0.316 (n.s.) | 0.3534 | 0.3381 |
| evidential | U_aleatoric | 0.364 (n.s.) | 0.6374 | 0.6277 |
| mc_dropout | U_epistemic | 0.018 | 0.0251 | 0.0268 |
| mc_dropout | U_aleatoric | 0.998 (n.s., wrong direction) | 0.3841 | 0.2794 |

Accuracy against `original_label` collapses from ~0.87-0.89 at t=0.0 to
~0.51-0.56 at t=1.5 for every method -- expected drift toward chance as
sequences diverge, reported here only as accuracy-collapse tracking, not as
a validated-label claim (see this README's Limitations section above: label
validity is not asserted at large branch lengths).

### Interpretation: the epistemic-rises / aleatoric-flat hypothesis held only partly, method-dependent

- **conv_epinet**: `U_epistemic` rises significantly with branch length
  (Spearman rho=0.108, p=7.9e-10; endpoint MWU p=2.3e-5) -- the hypothesis's
  epistemic half holds. But `U_aleatoric` rises *at least as strongly*
  (rho=0.127, p=5.1e-13; endpoint MWU p=8.8e-8) -- **this disconfirms the
  "aleatoric stays flat" half**. This replicates, on a continuous and
  independently-built axis, the same conv_epinet aleatoric-leakage-under-
  input-novelty confound this project's own prior discrete-severity work
  found (`data_gen/label_noise/decomp_compare_ood_severity.py`) -- not a
  surprise per the task brief, but a straightforward confirmation that the
  confound isn't an artifact of that earlier dataset's shuffled/random-DNA
  construction.
- **evidential**: neither vacuity (epistemic-analogue) nor its own
  `U_aleatoric` show a statistically significant relationship with branch
  length across the full range (Spearman p=0.138 and p=0.275; endpoint MWU
  p=0.316 and p=0.364) -- both terms are essentially flat overall, with a
  mild, non-monotonic hump peaking around t=0.4 and *falling back down* by
  t=1.5 (vacuity: 0.353 -> 0.388 -> 0.338). This **disconfirms the
  hypothesis for evidential**: accuracy still collapses toward chance
  (0.885 -> 0.538) exactly as for the other methods, but evidential's own
  uncertainty signal does not track that collapse in a monotonic,
  significant way over the full grid.
- **mc_dropout**: `U_epistemic` rises with weak but significant magnitude
  (rho=0.055, p=0.0020; endpoint MWU p=0.018) and is non-monotonic
  (peaks at t=0.2, 0.045, then declines toward 0.027 at t=1.5) -- a partial,
  attenuated confirmation. `U_aleatoric` moves in the *opposite* direction
  overall (rho=-0.055, p=0.0020) and the endpoint comparison goes the wrong
  way entirely (mean 0.384 at t=0.0 down to 0.279 at t=1.5, p=0.998 for
  "hi > lo"). This is a different failure mode than conv_epinet's leakage:
  rather than aleatoric rising alongside epistemic, both signals shrink back
  down at the most extreme branch length, plausibly because near-saturated
  divergence (~50% identity, close to the 25% stationary-distribution floor)
  produces inputs the model settles into confidently -- if wrongly --
  classifying, rather than inputs it visibly hedges on via dropout
  variance. Whatever the mechanism, the aleatoric channel does not stay
  flat here either; it moves, just in a way that happens not to look like
  conv_epinet's leakage.

**Net**: the epistemic-rises direction held (to varying, generally modest,
degrees) for two of three methods (conv_epinet, mc_dropout) and did not hold
for evidential; the aleatoric-flat half of the hypothesis did not clearly
hold for *any* of the three methods -- each showed a significant or at
least non-trivial *aleatoric* trend of its own (rising for conv_epinet,
falling for mc_dropout, flat-but-noisy/non-monotonic for evidential).
Reported plainly per the task brief: this is a disconfirming result for the
"aleatoric stays flat under real evolutionary novelty" half of the
hypothesis across the board, consistent with this project's existing
concern (documented elsewhere in this repo) that these decompositions'
aleatoric channels are not cleanly isolated from input novelty in practice.

## Dense-grid uncertainty evaluation: extended to the full 8-method roster

`data_gen/promoter_alisim/uncertainty_eval/dense_grid/` reruns the same
question (does `U_epistemic` rise with branch length while `U_aleatoric`
stays flat?) on the 16-point dense grid above, and extends it from the
original 3 UQ methods to the **full method roster this project uses
elsewhere** -- 8 method families, 9 configurations in total (`ensemble` is
evaluated at both K=5 and K=3):

| # | method(s) | mechanism | checkpoint / model |
|---|---|---|---|
| 1 | `conv_epinet` | epinet index-sample draws (K=10) | `checkpoints/seed_1/DNABERT2/label_noise_r00/epinet` |
| 2 | `mc_dropout` | stochastic dropout forward passes (K=10) | `checkpoints/seed_1/DNABERT2/label_noise_r00/base` |
| 3 | `evidential` | Dirichlet evidence, single forward pass (vacuity as epistemic-analogue) | `checkpoints/seed_1/DNABERT2/label_noise_r00/evidential` |
| 4 | `laplace` | last-layer diagonal Laplace, K=16 posterior samples, post-hoc | `checkpoints/seed_1/DNABERT2/label_noise_r00/base` |
| 5 | `ensemble_k5` | deep ensemble, 5 independently fine-tuned checkpoints | `checkpoints/seed_{1,2,3,4,42}/DNABERT2/label_noise_r00/base` |
| 6 | `ensemble_k3` | deep ensemble, 3 of the same 5 checkpoints | `checkpoints/seed_{1,2,3}/DNABERT2/label_noise_r00/base` |
| 7 | `cnn_mc_dropout` | from-scratch 1D-CNN, stochastic dropout (K=16) | freshly trained (see below) |
| 8 | `cnn_ensemble` | from-scratch 1D-CNN, 5 independently-initialized-and-trained models | freshly trained (see below) |
| 9 | `rf_kmer` | k-mer (k=6) count features + Random Forest (200 trees), per-tree class probabilities as K samples | freshly trained (see below) |

Methods 1-3 are unchanged from the original 8-point-grid run (see
`data_gen/label_noise/decomp_compare_ood_severity*.py` /
`decomp_compare_label_noise_laplace.py` for the exact templates the new
methods 4-9 were adapted from -- built for this project's
real/shuffled/random_dna label-noise severity axis and here swapped onto
the 16 branch-length values instead). Contents:

- `run_uncertainty_eval_dense.py` -- unchanged, produces methods 1-3.
- `run_new_methods_dense.py` -- new, produces methods 4-9; appends its rows
  into the same `per_example_uncertainty.csv` (dropping/replacing any
  previous rows for the same method names on a re-run, so it is safe to
  re-run without duplicating rows) and regenerates all four
  `results_summary_*.csv` files plus `results_summary.md` from the
  **combined** dataframe (all 9 configurations), not just the new ones.
- `per_example_uncertainty.csv` -- all 57,600 rows (6400 rows x 9
  configurations), same schema as the original: `anchor_id, branch_length,
  realized_identity_to_anchor, original_label, method, U_aleatoric,
  U_epistemic, pred, labels, correct`.
- `results_summary.md` (+ its four component CSVs) -- the combined
  dose-response table, Spearman correlations, endpoint Mann-Whitney tests,
  and shape characterization (peak location, %rise-to-peak,
  %decline-from-peak, plateau-vs-still-rising-vs-declining classification)
  across all 9 configurations, plus a data-driven interpretation section
  (grouping methods by shape, classifying each method's own aleatoric trend)
  generated directly from the summary tables on every run.

### Judgment calls for the 6 new configurations

- **Laplace, ensembles: post-hoc / no new training beyond what's already
  checkpointed.** `laplace` fits its diagonal GGN on the seed-1 DNABERT2
  base checkpoint's own training data
  (`data_gen/label_noise/csv_data_r00/train.csv`, the clean r00 promoter_all
  split that checkpoint was actually fine-tuned on), capped at 2000 examples
  and `prior_precision=1.0`, K=16 posterior samples -- identical
  hyperparameters to `decomp_compare_label_noise_laplace.py`. The ensembles
  reuse the 5 existing DNABERT2 `label_noise_r00` base checkpoints (seeds 1,
  2, 3, 4, 42) with no retraining; `ensemble_k3` is simply the first 3 of
  those 5 members (seeds 1, 2, 3) evaluated as a 3-way ensemble, computed in
  the same forward pass as `ensemble_k5` (5 models' logits are gathered once
  per batch, then both a 5-way and a 3-way subset are stacked from the same
  logits -- not two separate inference passes).
- **CNN and RF: retrained fresh on standard `promoter_all`/`label_noise_r00`
  training data**, per `decomp_compare_ood_severity_cnn.py` /
  `decomp_compare_ood_severity_rf.py`'s convention (those scripts always
  train fresh rather than loading a saved model -- there is no saved
  from-scratch CNN or RF checkpoint anywhere in this repo to reuse). Both
  train on `data_gen/label_noise/csv_data_r00/train.csv` (30,000 sequences,
  0% injected label noise -- the same clean promoter_all split the DNABERT2
  `label_noise_r00` checkpoints were fine-tuned on, for a fair comparison
  across the whole roster) and are evaluated zero-shot on the AliSim dense
  grid, exactly as the DNABERT2-based methods are.
  - `cnn_mc_dropout`: one `SmallCNN` (`data_gen/label_noise/cnn_scratch.py`
    -- a DeepBind/DeepSEA-lite one-hot-conv architecture, no pretraining),
    seed 1, dropout_p=0.3, K=16 stochastic forward passes at eval.
  - `cnn_ensemble`: 5 independently-initialized-and-trained `SmallCNN`
    instances (seeds 1, 2, 3, 4, 42) -- a *stronger* independence condition
    than the DNABERT2 ensemble, since none of the 5 CNNs share any
    pretrained initialization at all.
  - `rf_kmer`: `CountVectorizer(analyzer="char", ngram_range=(6,6))` k-mer
    features + `RandomForestClassifier(n_estimators=200,
    min_samples_leaf=20)` (the non-default `min_samples_leaf=20` avoids the
    sklearn default of pure/single-example leaves, which would otherwise
    collapse per-tree aleatoric uncertainty to near-zero as a training
    artifact -- see `decomp_compare_label_noise_rf.py`). All 200 trees'
    per-tree class probabilities are stacked as the K=200 "samples" fed to
    `compute_uncertainty`, identical to the label-noise-axis RF baseline.
- **Epinet fix applied wholesale to this checkout too**, for consistency
  with the original 3-method run (`nn_proj/models/epinet/epinet.py`
  replaced with the branch carrying the per-example-`z` fix) -- though none
  of the 6 new methods touch epinet code themselves, this keeps
  `conv_epinet`'s already-computed rows (reused unchanged in the combined
  table) consistent with what would be produced if the whole pipeline were
  re-run from scratch here.
- **Schema consistency**: all 9 configurations report the same
  entropy-based BALD decomposition (`compute_uncertainty` from
  `nn_proj.common.utils`) as `U_aleatoric`/`U_epistemic`, matching the
  original 3-method schema exactly -- not the separate variance-based
  decomposition (`nn_proj.common.variance_decomp`) that some of the
  label-noise-axis templates also compute (as `var_U_epistemic`/
  `var_U_aleatoric`) alongside the BALD version. That second decomposition
  was left out of this combined table to keep one schema across all 9 rows;
  it is available by rerunning any of the individual template scripts if a
  variance-based cross-check is ever needed.

### Environment

Same venv as the original 3-method run:
`/scratch/home/glh52/venvs/aleatoric_boundary_venv` (Python 3.10.12). At the
time this extension was run that venv had `transformers==4.30.2` (not
`4.29.2` as originally pinned) -- re-verified end-to-end before use:
DNABERT2's `AutoConfig`/`AutoModelForSequenceClassification.from_pretrained`
load without the `config_class` consistency error described in the original
3-method README section at this version too, so no further pin was needed
for this extension. `scikit-learn` (1.7.2) was already present for the RF
baseline; no other new dependencies.

### Results

See `uncertainty_eval/dense_grid/results_summary.md` for the full combined
dose-response / Spearman / endpoint-MWU / shape-characterization tables and
data-driven interpretation across all 9 configurations. Headline findings:

- **Three distinct shape families, not one universal pattern.** Grouping by
  the shape-characterization column: `conv_epinet`, `laplace`, and `rf_kmer`
  (3/9) keep rising with no interior peak through t=3.0; `cnn_mc_dropout`
  and `cnn_ensemble` (2/9) rise sharply then plateau within 3% of their peak;
  `mc_dropout`, `evidential`, `ensemble_k3`, and `ensemble_k5` (4/9) rise
  then decline back down after peaking around branch_length 0.2-0.6 -- the
  hump-shaped pattern first seen for 2/3 methods in the original run now
  extends to *both* DNABERT2 deep-ensemble configurations as well.
- **The two non-pretrained baselines (CNN, RF) show the strongest, most
  significant epistemic responses of the whole roster** -- `rf_kmer` has by
  far the largest Spearman rho (0.563, vs. 0.12-0.19 for every
  DNABERT2-based method) and the two CNN variants show the largest relative
  rises (83-105% above baseline). A from-scratch model with no pretrained
  representation to fall back on appears to track raw compositional/k-mer
  novelty more directly than a fine-tuned transformer does.
- **The aleatoric-stays-flat half of the hypothesis fails for essentially
  the whole roster**: 7 of 9 configurations show `U_aleatoric` rising
  significantly alongside `U_epistemic` (conflation), `mc_dropout` moves in
  the *wrong* direction (falls as branch length increases, alongside
  accuracy collapsing), and only `evidential` comes close to a flat
  aleatoric trend (nominally significant at n=6400 but a negligible effect
  size next to every other method). This holds regardless of UQ mechanism
  (dropout, epinet, ensembling, Laplace, classical RF) or pretraining status
  -- reported plainly as a roster-wide, not method-specific, limitation of
  this project's aleatoric/epistemic decomposition under real evolutionary
  input novelty.
