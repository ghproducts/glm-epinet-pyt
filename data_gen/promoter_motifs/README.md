# Promoter motif composition

A regulatory-sequence distribution-shift benchmark built from
`human_nontata_promoters` (via `genomic-benchmarks`), where in-distribution
(ID) and out-of-distribution (OOD) test sets differ **only** in which
transcription-factor (TF) motif *combinations* a sequence carries. Train and
every test set share the same binary promoter/non-promoter task and the same
label space — unlike the cross-task regulatory pairs in
`configs/tasks.yaml` (`promoter_all -> enhancers`, etc.), this is a genuine
same-target shift.

Ported from a Drexel DSCI 691 course project
(`.docs/DSCI-691-Project/`, not tracked in this repository) by this
manuscript's corresponding author, generalized here to run from committed,
documented steps instead of a personal-path notebook. See
`docs/REVISION_PLAN.md` in `.docs/` for why this dataset was pulled in
(it targets Reviewer 2's point 1 — task-mismatched OOD evals aren't real
distribution shift — and Reviewer 1's point 3 — the near/far-OOD hypothesis
was asserted, never tested).

## Pipeline

Five stages. Stages 1 and 3 need external tools/downloads; stages 2, 4, and 5
are deterministic Python, committed here.

```
1. download_promoters.py   corpus download + FASTA export      (network)
   -- external: JASPAR download, AME enrichment, FIMO scan --  (MEME suite)
2. select_motifs.py        extract the 6 selected JASPAR motifs (deterministic)
   -- external: FIMO scan of promoters_all.fa against the selected motifs --
3. annotate_motifs.py      join FIMO hits onto the corpus       (deterministic)
4. make_splits.py          assign train/ID/matched-ID/OOD        (deterministic, verified)
5. build_csvs.py           manifest + corpus -> model-ready CSVs (deterministic)
```

Stage 4's output (`splits/motif_splits.csv`) is committed — it is the exact
split behind the released `.docs/DSCI-691-Project/results/baseline_results.csv`
numbers, re-derived here from those released CSVs rather than from a fresh
rerun (see **Regenerating vs. using the release**, below). Stage 5's output
(`csv_data/`) is not committed, the same way `data_gen/pbsim/csv_data/` isn't:
regenerate it from the manifest with `build_csvs.py`.

### 1. Download the corpus

```bash
python data_gen/promoter_motifs/download_promoters.py \
    --out-dir data_gen/promoter_motifs
```

Downloads `human_nontata_promoters` (version 0) via `genomic-benchmarks` and
writes `promoters_all.csv` (`sequence_id, sequence, label, label_name,
source_split`) plus per-class FASTA files for the AME step below.
`source_split` is the dataset's own train/test split, distinct from — and not
used by — the train/ID/OOD splits this module builds.

### 2. Select motifs

```bash
mkdir -p data_gen/promoter_motifs/motifs
cd data_gen/promoter_motifs/motifs
wget -O jaspar_core_vertebrates.meme \
    https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt
# 1019 motifs (MEME suite / AME / FIMO version 5.5.9, via bioconda)

mkdir -p ../../../results/ame_promoter_enrichment_text
ame --text \
    --control ../promoters_negative_train.fa \
    ../promoters_positive_train.fa \
    jaspar_core_vertebrates.meme \
    > ../../../results/ame_promoter_enrichment_text/ame.tsv
```

AME ranks JASPAR motifs by enrichment in promoter vs. non-promoter sequences.
The six below were hand-picked from the top of that ranking for enrichment
*and* diversity — many of the highest-ranked motifs are just CpG-content
artifacts (C-next-to-G is globally depleted in the genome because it
spontaneously deaminates to T, so anywhere that depletion is suppressed —
promoters included — looks "enriched" for almost any GC-rich motif; that's
not a meaningful regulatory signal on its own):

```bash
python data_gen/promoter_motifs/select_motifs.py \
    --jaspar-meme data_gen/promoter_motifs/motifs/jaspar_core_vertebrates.meme \
    --out data_gen/promoter_motifs/motifs/selected_promoter_motifs.meme
```

| JASPAR ID | name |
|---|---|
| MA0516.3 | SP2 |
| MA2328.1 | ZBED4 |
| MA0162.5 | EGR1 |
| MA0759.3 | ELK3 |
| MA1122.2 | TFDP1 |
| MA2546.1 | ZNF131 |

### 3. Scan and annotate

```bash
mkdir -p data_gen/promoter_motifs/fimo_out
fimo --oc data_gen/promoter_motifs/fimo_out --thresh 1e-4 \
    data_gen/promoter_motifs/motifs/selected_promoter_motifs.meme \
    data_gen/promoter_motifs/promoters_all.fa

python data_gen/promoter_motifs/annotate_motifs.py \
    --promoters data_gen/promoter_motifs/promoters_all.csv \
    --fimo data_gen/promoter_motifs/fimo_out/fimo.tsv \
    --out data_gen/promoter_motifs/promoters_annotated.csv
```

Writes one presence flag per motif (`SP2`, `ZBED4`, `EGR1`, `ELK3`, `TFDP1`,
`ZNF131`), plus `num_selected_motifs` (their sum) and `motif_combo` (the
sorted `+`-joined names of the motifs present, or `none`).

### 4. Assign splits

```bash
python data_gen/promoter_motifs/make_splits.py \
    --annotated data_gen/promoter_motifs/promoters_annotated.csv \
    --outdir data_gen/promoter_motifs/splits
```

The operational definition of the shift. Every eligible sequence goes to
exactly one of four splits:

| split | rule |
|---|---|
| `train` | 80% of the non-held-out eligible pool, stratified by label |
| `test_ID` | the other 20% |
| `test_matched_ID` | resampled from `test_ID` to match `test_OOD`'s per-label, per-motif-*count* distribution — a matched control, not a shift condition |
| `test_OOD` | every sequence carrying one of four held-out motif *combinations* |

**Held-out combinations** (excluded from training entirely):
`EGR1+SP2`, `SP2+TFDP1`, `SP2+ZNF131`, `ZBED4+ZNF131`.

**Eligible combinations** (everything else is dropped — insufficient support
in one label or the other): `none`, `SP2`, `EGR1`, `ZBED4`, `ZNF131`, `ELK3`,
`TFDP1`, `SP2+ZBED4`, plus the four held-out ones above and
`SP2+ZBED4+ZNF131`, `EGR1+SP2+ZBED4+ZNF131` (multi-motif combinations that
share support with both groups).

Both the 80/20 train/ID split and the matched-ID resampling are seeded
(`--seed`, default 42) for reproducibility. `verify()` checks the invariants
this split depends on and refuses to write output if any fail — see
**Verification** below for what it actually checks and the one thing it
currently only warns about.

### 5. Build model-ready CSVs

```bash
python data_gen/promoter_motifs/build_csvs.py \
    --manifest data_gen/promoter_motifs/splits/motif_splits.csv \
    --annotated data_gen/promoter_motifs/promoters_annotated.csv \
    --outdir data_gen/promoter_motifs/csv_data
```

Joins the committed manifest back onto the sequence text and writes
`train.csv`, `test_ID.csv`, `test_matched_ID.csv`, `test_OOD.csv` with the
columns `nn_proj.common.datasets.load_local_dataset` expects (`sequence`,
`label`) plus `sequence_id`, `motif_combo`, and `num_selected_motifs` carried
through as metadata — once `MODEL_CODE_FIXES.md` item 2.3 (metadata
passthrough in `predict()`) lands, these are what let per-motif-combo
stratification (Reviewer 2 point 3, Reviewer 1 point 3) run directly off the
prediction CSVs.

## Regenerating vs. using the release

`splits/motif_splits.csv` was **not** produced by running `make_splits.py`
against a fresh download — it was derived directly from the four CSVs
released with the DSCI-691 project
(`train.csv`, `test_ID.csv`, `test_motif_matched_ID.csv`, `test_OOD.csv`,
originally under `.docs/DSCI-691-Project/dataset_g/`), which are the exact
split behind `baseline_results.csv`'s published KNN/CNN/BiLSTM numbers.
`make_splits.py`'s stratified 80/20 draw depends on the row order of its
input, which a fresh download-and-annotate run will not reproduce bit-for-bit
even with the same `--seed` — so re-running the pipeline from scratch
regenerates a *statistically equivalent* split (same rule, same combination
sets, same class balance), not an *identical* one. This is the same tension
`data_gen/pbsim/splits/` resolves the same way: the committed split table is
the authoritative one behind published numbers; the script is how you'd build
a new one, not how you'd reproduce this one exactly.

`make_splits.py --check` re-runs `verify()` against an existing manifest
without recomputing anything, which is what was actually used to validate the
ported logic here: `verify()` passes against `splits/motif_splits.csv` as
released.

## Verification

`make_splits.py`'s `verify()` checks, and fails closed if any is violated:

- **No held-out combination leaks into training.** All four held-out
  combinations have zero rows in `train`.
- **No sequence appears in more than one split**, except the expected,
  by-construction overlap between `test_ID` and `test_matched_ID` (the
  matched control is drawn *from* the ID test pool, not disjoint from it —
  don't concatenate the two and treat every row as an independent test
  observation).
- **Motif-count distributions match** between `test_matched_ID` and
  `test_OOD`, per label — this is the property the matched control exists to
  guarantee.

One thing it only **warns** about, loudly, because it's a real limitation of
the released split rather than a bug to fail on:

> **`test_matched_ID` is heavily resampled with replacement.** The 1,569 rows
> released as the matched-ID control set contain only **122 unique
> sequences**, repeated up to **25 times each**. This happens because the
> matching pool for `num_selected_motifs == 2` is restricted to sequences
> carrying exactly the one eligible two-motif combination (`SP2+ZBED4`) — a
> small pool relative to the 1,569 rows needed to match `test_OOD`'s size, so
> `pool.sample(..., replace=True)` fires for nearly every group. Treat
> `test_matched_ID` as **~122 independent observations, not 1,569** for any
> analysis sensitive to sample size (significance tests, CI width); metrics
> that only need a distribution (mean ECE, mean error) are less affected but
> still correlated within each repeated sequence's copies. Fixing this
> properly (widening the matching criterion, or accepting a smaller matched
> set) is scoped as follow-up work, not done here — see `.docs/REVISION_PLAN.md`.

## Known limitations carried over from the source project

- Motif selection (the 6 JASPAR IDs) was a manual choice from the AME
  ranking, not an automated procedure — documented above, not algorithmic.
- `promoters_all.csv`'s `source_split` (the `human_nontata_promoters`
  dataset's own train/test division) is unrelated to and not aligned with
  this module's train/test_ID/test_OOD split; a sequence's `source_split`
  value is provenance only.
- Sequences are a fixed 251bp (the `human_nontata_promoters` sequence
  length); no length-based confound between splits.
