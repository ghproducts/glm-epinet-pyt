# Reproducing the results

Every number, table and figure in the manuscript is produced by the commands
below. Nothing is transcribed by hand.

The pipeline has four stages. Stages 1–2 need GPUs and take days; stages 3–4
run on a laptop in about two minutes from the released prediction files.

```
1. data_gen/     build the datasets                (once, ~1 day + downloads)
2. scripts/      train and run inference           (2160 jobs, GPU)
3. nn_proj.analysis.aggregate    predictions -> tables      (~2 min)
4. nn_proj.analysis.figures      tables -> figures          (~30 s)
```

If you only want to check the analysis, skip to stage 3 with the released
`results/` archive.

---

## 1. Datasets

### Regulatory tasks

Downloaded automatically from HuggingFace
(`InstaDeepAI/nucleotide_transformer_downstream_tasks_revised`). Nothing to do.

### Metagenomic gene classification

Downloaded automatically from `MsAlEhR/scorpio-gene-taxa`.

### Simulated long reads

Three steps, in order. The output of step 1 is committed to this repository, so
steps 1 and 2 only need rerunning to change the split parameters.

```bash
# 1. Assign species to splits. Deterministic given --seed.
#    Outputs are already committed under data_gen/pbsim/splits/.
python data_gen/pbsim/make_splits.py \
    --lineage data_gen/pbsim/full_basic_lineage.csv \
    --sequences /path/to/refseq/genomes \
    --seed 123

# 2. Simulate reads with PBSim3 (requires `pbsim` on PATH).
LENGTH_MIN=10000 LENGTH_MAX=11000 DEPTH=1 \
    bash data_gen/pbsim/generate_pbsim_reads.sh

# 3. Convert FASTQ to the CSV format the training code reads.
python data_gen/pbsim/reads_to_csv.py --seed 123
```

`make_splits.py` verifies its own output before writing: `id_novel_genus` must
share no genus with training while every one of its families is present in
training, `ood_novel_family` must share no family with training, and no species
may appear in two splits. It refuses to write if any of these fail.

> **On read length:** reads are simulated at 10–11 kbp and then capped at 6 kbp
> before the models see them, which is the 6 kbp the manuscript reports.
> `configs/experiments.yaml` records `seq_length: 6000`, and each backbone's
> `model_max_length` follows from it.
>
> The tool comparison does *not* apply that cap — the stored Kraken2 output has
> median read length 10,745 bp — so Kraken2 and MMseqs2 were given roughly 1.8x
> more sequence per read than the models. Truncate to 6 kbp before rerunning the
> tools, or state the asymmetry. See `docs/FINDINGS.md` item 6.

Reference genomes are downloaded per species taxid into `sequences/<taxid>/`.
The taxonomy comes from Woltka's RefSeq snapshot of 2023-07-24.

---

## 2. Training and inference

`configs/experiments.yaml` holds the whole grid: 4 backbones x 8 training tasks
(including one per taxonomic rank) x 5 seeds, then 4 uncertainty methods x
every test set. `scripts/run_grid.py` expands it.

```bash
# Inspect the plan. Nothing runs without --run.
python scripts/run_grid.py --dry-run
# -> train: 160, epinet: 160, temperature: 160, inference: 1680 (total 2160)

# Run one slice
python scripts/run_grid.py --stage train --backbone DNABERT2 --seed 1 --run

# Run everything, resuming after an interruption
python scripts/run_grid.py --stage all --skip-existing --continue-on-error --run

# Or emit a job list for a scheduler
python scripts/run_grid.py --dry-run --format plain > jobs.txt
```

Stages must run in order: `train` -> `epinet` -> `temperature` -> `inference`.
The temperature stage writes `checkpoints/temperatures.json`, and the
inference stage reads it for the `base_scaled` runs. **If that file is missing,
`base_scaled` silently falls back to T = 1.0 and is then identical to `base`.**
The driver warns when a temperature cannot be parsed.

The four shell scripts in `scripts/` still work for single one-off runs, but
they encode one cell of the grid at a time and must be edited between runs.
Prefer `run_grid.py` for anything reported.

### `model_max_length`

Derived from each tokenizer's compression rather than hard-coded:

| backbone | tokens per base | promoter_all (300 bp) |
|---|---|---|
| NT v2 | 1/6 (6-mer) | 51 |
| DNABERT-2 | 0.25 (BPE) | 75 |
| HyenaDNA | 1 (character) | 300 |
| CARMANIA | 1 (character) | 300 |

These reproduce the constants in the original shell templates; a regression
test pins the DNABERT-2 value at 75.

---

## 3. Predictions to tables

```bash
python -m nn_proj.analysis.aggregate results/ -o tables/
```

Writes:

| file | contents |
|---|---|
| `calibration.csv` | per seed: error, ECE, NLL, Brier, AURC/AUCO, mean uncertainties |
| `calibration_summary.csv` | the above as mean ± std across seeds, with `n_seeds` |
| `ood_detection.csv` | per seed: AUROC and Δ vs base, per ID/OOD pair and score |
| `ood_detection_summary.csv` | the above aggregated across seeds |
| `ece_bin_sensitivity.csv` | ECE across bin counts and binning schemes |
| `task_registry.csv` | the shift-category specification, as a table |
| `missing_runs.csv` | registered cells with no prediction file |
| `row_count_anomalies.csv` | runs with fewer rows than the rest of their test set |

Read the last two before trusting a figure. On the released prediction set they
report 3 missing runs and 4 short files; see `docs/FINDINGS.md`.

Aggregation applies three rules that the original notebooks did not:

1. **`n_seeds` travels with every cell.** A cell backed by two seeds is never
   presented like one backed by five.
2. **Degenerate scores are dropped.** The epistemic component of a
   deterministic model is analytically zero but computed as a difference of
   entropy estimates, landing within ~4e-8 of zero. Ranking on that noise
   yields an AUROC that looks meaningful and is not.
3. **Calibration is only reported where it is interpretable.** Pairs that
   change the prediction target carry `supports_calibration = False`. Their
   error and ECE are still computed and stored, so the filtering decision is
   visible and reversible, but figures exclude them by default.

---

## 4. Tables to figures

```bash
python -m nn_proj.analysis.figures tables/ -o plots/ --results results/
```

| figure | manuscript | source |
|---|---|---|
| `regulatory_scatter.pdf`, `metagenomic_scatter.pdf` | Figs 4, 7 | `calibration_summary.csv` |
| `reliability_*.pdf` | Figs 5, 8 | prediction files + `calibration_summary.csv` |
| `auroc_*.pdf` | Figs 6, 9 | `ood_detection_summary.csv` |
| `risk_coverage_*.pdf` | new | prediction files |
| `ece_sensitivity_*.pdf` | new | `ece_bin_sensitivity.csv` |

`--results` is needed only for the reliability and risk-coverage figures, which
read per-example predictions.

Two changes from the published versions worth knowing about:

* Reliability legends show **mean ± std across seeds with the seed count**. The
  published legends were single-seed values; see `docs/FINDINGS.md` item 1.
* AUROC heatmaps carry the **absolute base AUROC** beside each row. A ΔAUROC of
  +0.08 over a chance-level 0.48 baseline is a different claim from +0.08 over
  0.90, and the published heatmaps did not let a reader tell them apart.

---

## Shift categories

`configs/tasks.yaml` specifies, for every train/test pair: the prediction
target, the label spaces, which biological unit is held out, the shift
category, and which metrics the pair supports.

Categories are assigned by **what is held out of training**, not by
thresholding a similarity statistic:

| category | rule |
|---|---|
| ID | held-out split of the same corpus, same task, same label space |
| Near-ID | same target and label space; a unit *below* the label rank is held out |
| Near-OOD | same target; the labelled unit itself is novel, so error is 1.0 by construction |
| OOD | outside the trained domain, or the target-defining variable held out |

BLAST statistics are descriptive evidence about residual similarity, not the
assignment rule. They are not monotone in the category ordering: near-OOD
sequences can retain high local similarity through conserved regions. Treating
similarity as the criterion would conflate "distant in sequence space" with
"novel with respect to the label".

To regenerate the BLAST table:

```bash
python -m nn_proj.common.blast \
    --train_data_path data_gen/pbsim/csv_data/reads_train.csv \
    --test_data_path  data_gen/pbsim/csv_data/reads_id_novel_genus.csv \
    --taxa_rank family --taxa_df data_gen/pbsim/full_basic_lineage.csv \
    --seed 1 --outdir blast_eval/pbsim_family_id_novel_genus
```

> **`blast.py` is not yet fixed.** As it stands the script applies no rank
> mapping and class-encodes the train and test files independently, so the
> comparison runs across mismatched label spaces — which is why the published
> `novel_genus` precision is 0.0. It also hard-codes the split seed to 42 while
> training splits on `data_seed`. The command above shows the interface the
> fixed version should have; see `docs/MODEL_CODE_FIXES.md` item 1.1.
>
> Until then, `scripts/check_tool_panels.py` gives the same statistics from the
> stored MMseqs2 searches, which carry labels inside the sequence IDs and so are
> not affected.

---

## Tool comparison

```bash
python data_gen/tools/tool_confidence.py kraken2 kraken_output.txt \
    --labels data_gen/pbsim/csv_data/reads_id_novel_genus.csv \
    --lineage data_gen/pbsim/full_basic_lineage.csv --rank family \
    -o results/tools/kraken2/pbsim_family/id_novel_genus_family
```

Output lands in the standard prediction format, so `nn_proj.analysis` treats
tools and models identically.

To recompute the published tool panels and see what they depend on:

```bash
python scripts/check_tool_panels.py \
    --kraken2 <dir>/kraken2 --mmseqs <dir>/mmseqs
python scripts/check_tool_panels.py \
    --kraken2 <dir>/kraken2 --mmseqs <dir>/mmseqs --taxid-map full
```

The MMseqs2 rows reproduce exactly under the published settings. The Kraken2
rows do not, and swing by up to 40 percentage points between the two
`--taxid-map` settings, because that choice decides whether 5% or 100% of the
reads are retained. See `docs/FINDINGS.md` item 9 before quoting them.

These scores are uncalibrated by construction and were never intended as
probabilities of correctness. Reporting their raw ECE next to a model's is close
to a tautology. Either calibrate them on the same validation split, or compare
on ranking ability (AUROC, risk-coverage) — which is what they are built for.
The module docstring covers both.

---

## Repairing prediction files

Ten prediction files from the seed-42 DNABERT-2 `gene_taxa` runs have a NumPy
repr interleaved into the CSV. The loader repairs them transparently; to fix
them on disk:

```bash
python -m nn_proj.analysis.repair results/            # report
python -m nn_proj.analysis.repair results/ --write    # rewrite, keeping .corrupt backups
```

Eight recover exactly. Three `base_scaled` files lost rows beyond recovery and
should be re-run; `row_count_anomalies.csv` flags them on every aggregation.

---

## Tests

```bash
python -m pytest tests/ -q
```

Checks that every backbone entry point imports, that the argument parsers
accept the flags the grid driver emits, and that the grid expands to the
expected shape. It does not train anything — it catches the failure mode where
a refactor drops a flag and a run dies hours in.

---

## Environment

Python 3.11+ (developed on 3.11, tested on 3.14), PyTorch 2.x, one A100 for
training. Analysis needs only pandas, numpy, scikit-learn, matplotlib, pyyaml.

```bash
pip install -r requirements.txt
```
