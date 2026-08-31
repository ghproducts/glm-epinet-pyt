# Discrepancies found while rebuilding the analysis pipeline

Recorded during consolidation of the analysis and dataset-generation code into
this repository. Each item was found by recomputing a published number from the
raw prediction files.

**None of the model-code fixes these findings imply have been applied.**
`nn_proj/models/` and `nn_proj/common/` are untouched on this branch; the
changes are specified in [MODEL_CODE_FIXES.md](MODEL_CODE_FIXES.md).

## 1. Reported ECE values are single-seed, not 5-seed means

The manuscript states results are averaged across 5 seeds. The ECE values in the
Fig 5 reliability-plot legends match **seed 42 alone**, to within 0.04
percentage points in every case checked.

| backbone | pair | method | paper (seed 42) | 5-seed mean |
|---|---|---|---|---|
| NT | promoter_all→enhancers | base | 23.5 | 17.9 |
| NT | promoter_all→enhancers | epinet | 17.9 | 12.2 |
| DNABERT2 | promoter_all→enhancers | base | 23.0 | 20.2 |
| DNABERT2 | promoter_all→enhancers | epinet | 17.4 | 13.1 |

Seed 42 is systematically the worst seed for the OOD baseline (23.5 vs 15.2–17.3
for seeds 1–4), so the published baseline ECE is inflated.

This was confirmed independently on three figures. Figures 5, 8 and 9 all match
seed 42 rather than the 5-seed mean: Figure 8 matches it exactly in 23 of 24
cells (the 24th is item 8 below), and Figure 9 reproduces to three decimals.

Figure 4 is the exception and *is* the 5-seed mean — its 96 hard-coded values
match recomputation to a mean of 0.03 pp. So the manuscript mixes two seed
conventions across its own figures.

The direction of the finding survives and slightly strengthens: the epinet ECE
reduction on NT is −5.6 pp of an inflated baseline (24% relative) as published,
versus −5.7 pp of the correct baseline (32% relative). CARMANIA and HyenaDNA were
unaffected because their seed-42 values happen to sit near their means.

**Action:** regenerate every reported ECE from `tables/calibration_summary.csv`,
which reports mean ± std with the seed count attached.

## 2. Figures 4, 6, 7 and 9 were built from hand-transcribed constants

`data_viz.ipynb` and `AUROC_plots.ipynb` contained the ECE, error and AUROC
values as Python literals typed into the notebook rather than read from the
prediction files. `data_viz.ipynb` carried two conflicting copies of the same
table in adjacent cells (e.g. NT promoter_all base ECE as both 3.9 and 4.0).

**Action:** all four figures are now generated from `tables/*.csv` by
`nn_proj.analysis.figures`. No number in a figure is typed by hand.

## 3. Ten prediction files were corrupted by an interleaved array repr

The seed-42 DNABERT2 `gene_taxa` runs have a multi-line NumPy repr written into
the CSV between genuine rows, inflating `gene_taxa/test` from 140,524 to
1,404,362 lines. Pandas reads the affected columns as object dtype.

Eight of the ten files recover exactly (140,524 / 11,800 / 43,417 rows). Three
`base_scaled` files lost rows that were fragmented beyond recovery:

| file | recovered | expected | lost |
|---|---|---|---|
| base_scaled/gene_taxa/taxa_out | 11,206 | 11,800 | 594 (5.0%) |
| base_scaled/gene_taxa/gene_out | 42,580 | 43,417 | 837 (1.9%) |
| base_scaled/gene_taxa/test | 140,411 | 140,524 | 113 (0.08%) |

**Action:** `nn_proj.analysis.repair` detects and repairs these on load and on
disk. `ResultsIndex.row_count_check()` flags any run whose row count disagrees
with the rest of its test set, so the three partial files cannot silently
contribute a biased mean. Those three cells should be re-run.

## 4. Three prediction runs are missing entirely

`NT_transformer / mc_dropout / pbsim_phylum / ood_nonbacterial_phylum` has no
output for seeds 1, 3 and 4. That cell rests on 2 of 5 seeds.

**Action:** surfaced by `ResultsIndex.missing()` and written to
`tables/missing_runs.csv` on every aggregation run.

## 5. Table 2's precision column compares mismatched label spaces

`nn_proj/common/blast.py` loaded datasets without applying the taxonomic rank
mapping used everywhere else, and class-encoded train and test files
independently, so train label index *k* and test label index *k* denoted
different taxa. The reported `novel_genus` precision of 0.0 is an artefact of
that mismatch, not a measurement.

**Action:** not yet fixed — `blast.py` is untouched on this branch. The change
is specified in `docs/MODEL_CODE_FIXES.md` item 1.1. Confirmed empirically
against MMseqs2 in item 10 below.

## 6. Simulated read length — resolved, but it creates an asymmetry

`generate_pbsim_reads.sh` simulates 9.6–11.6 kbp reads while the manuscript says
6 kbp. Both are right: the reads are simulated long and then capped at 6 kbp by
the experiment scripts on the compute server before the models see them. The
manuscript's 6 kbp describes the model input, which is the relevant number.
`configs/experiments.yaml` records `seq_length: 6000`, and each backbone's
`model_max_length` follows from it.

**But the tool comparison does not apply the cap.** The stored Kraken2 output
shows read lengths of 9,589–11,571 bp (median 10,745), so Kraken2 and MMseqs2
ran on the full reads while the models saw the first ~6 kbp — roughly 1.8x more
sequence for the tools. Figure 10 compares them directly.

**Action:** either truncate to 6 kbp before rerunning the tools, or state the
asymmetry. `data_gen/tools/tool_confidence.py` reports the read length it sees
and notes this in its docstring.

## 7. Figure 6 reports a value that is mathematically impossible

The Temp Scaling column for `promoter_all -> enhancers` reads -0.018. Dividing
logits by T > 0 is a strictly monotone map of the logit gap, and for a binary
task both the top-class probability and the entropy are monotone functions of
that gap. The ranking is therefore unchanged, and AUROC — which depends only on
the ranking — is exactly invariant. The value must be 0.000.

Recomputation gives exactly 0.000, identical to four decimal places across all
five seeds. The other two binary regulatory rows are reported as +0.000 and
-0.002, as expected. `enhancers_types -> splice_sites_all` is a 3-class task
where a small change is legitimate, and there the reported +0.002 matches the
recomputed +0.001.

Every other cell of Figure 6 reproduces to within 0.001.

## 8. Figure 8 contains a transcription error that changed a headline count

The Base Scaled ECE for DNABERT-2, phylum rank, `id_novel_genus` is printed as
16.6%. No run produces that value: across the five seeds the cell ranges
5.97–7.52% (seed 42: 5.97%). The other 23 values in that figure match seed 42
exactly.

It matters because it flips the sign of the comparison. At 16.6% the cell counts
as temperature scaling degrading calibration against a base of 7.8%; at the
correct ~6.0% it counts as an improvement. Recomputing the manuscript's tally
over the same scope (4 backbones x 4 ranks on `id_novel_genus`) gives **11/16,
max +16.4 pp** against the reported **12/16, max +16.5 pp** — a one-cell
difference in exactly the place this error sits.

The companion OOD claim (9/12, max +16.8 pp) could not be reproduced under any
scoping tried; the closest comparable counts are 11/16 and 23/32. The
denominator of 12 is unexplained and should be re-derived from
`tables/calibration_summary.csv`.

## 9. Figure 10 — MMseqs2 reproduces exactly, Kraken2 does not

**MMseqs2 panels reproduce exactly.** Percent identity matches on all four
checked cells to 0.0 pp ECE and three decimal places of slope; query coverage
matches on ECE to 0.0 pp in all four.

| panel | score | paper ECE | recomputed | paper slope | recomputed |
|---|---|---|---|---|---|
| promoter_all→promoter_all | pident | 10.3 | 10.3 | -0.175 | -0.175 |
| promoter_all→promoter_all | qcov | 30.8 | 30.8 | +0.264 | +0.264 |
| promoter_all→enhancers | pident | 15.3 | 15.3 | -0.045 | -0.045 |
| promoter_all→enhancers | qcov | 26.3 | 26.3 | +0.713 | +0.713 |
| class/id_novel_genus | pident | 18.7 | 18.7 | -1.789 | -1.797 |
| phylum/ood_novel_family | pident | 17.2 | 17.2 | -1.550 | -1.550 |

Two caveats on how these were computed. The MMseqs frames keep **every hit
row**, not the best hit per query, so a query with ten alignments contributes
ten rows and is weighted ten times as heavily; this is not a per-sequence
calibration measurement. And queries with no hit at all are simply absent — for
`promoter_all` that is roughly 60% of the test set — which removes exactly the
sequences the tool is least able to classify. Both choices flatter the tool.

**Kraken2 does not reproduce, and the numbers are not well determined.** The
result hinges on an undocumented choice: whether a Kraken2 taxid is mapped to
the target rank using only the lineage table's `species` column, or every
lineage column. Kraken2 assigns reads to internal nodes, so the choice decides
whether most reads are kept or discarded:

| cell | mapping | reads kept | ECE % | paper |
|---|---|---|---|---|
| class/id_novel_genus | species column only | 272 / 3030 | 32.8 | 27.6 |
| class/id_novel_genus | all lineage columns | 3020 / 3030 | 14.6 | 27.6 |
| class/ood_novel_family | species column only | 103 / 1940 | 45.4 | 41.7 |
| class/ood_novel_family | all lineage columns | 1937 / 1940 | 5.1 | 41.7 |
| phylum/ood_novel_family | species column only | 103 / 1940 | 34.5 | 32.2 |
| phylum/ood_novel_family | all lineage columns | 1937 / 1940 | 5.2 | 32.2 |

The same input file yields ECE anywhere from 5.1% to 45.4% — a nine-fold range
— depending on this one decision. The published values sit closest to the
variant that discards 91–95% of the reads, but are still 2.7–3.7 pp above it,
so neither reading reproduces them.

**Action:** the Kraken2 panels should not be reported until the mapping rule is
fixed and stated, and until the discarded-read fraction is reported alongside
the ECE. An ECE computed on the 5% of reads a tool could confidently place is
not comparable to a model's ECE over the whole test set.

## 10. Table 2's precision artefact, confirmed with a second aligner

Table 2 could not be regenerated: no BLAST binaries are available here and no
BLAST output was retained. But the stored MMseqs2 searches carry the same
structure (`query|label`, `target|label`), so the same statistics can be
computed with a different aligner as an independent check.

Where the label spaces genuinely coincide, the two aligners agree closely on
best-hit precision:

| pair | BLAST (published) | MMseqs2 |
|---|---|---|
| promoter_all→promoter_all | 85.9 | 85.5 |
| splice_sites_acceptors→acceptors | 68.3 | 67.5 |
| splice_sites_acceptors→donors | 68.0 | 66.7 |
| gene_taxa→test | 99.6 | 99.9 |
| gene_taxa→taxa_out | 13.0 | 13.7 |
| **pbsim→id_novel_genus** | **0.0** | **20.6** |

Every row agrees to within 1.5 points except the one predicted to be broken.
`pbsim→id_novel_genus` is the Near-ID condition, where by construction the
family labels *are* shared with training, and MMseqs2 recovers 20.6% best-hit
precision. The published 0.0 is the label-space artefact of item 5, now
confirmed empirically rather than only read off the code.

By contrast `ood_novel_family` and `ood_nonbacterial` give 0.0 under both
aligners, which is correct: those families are held out, so no best hit can
match.

**The similarity gradient does not hold in either aligner.** Both agree that
the regulatory rows are not ordered by shift category: `splice_sites_acceptors`
ID has qcov 43.0 (BLAST) / 37.8 (MMseqs) against its Near-ID counterpart at
43.2 / 37.5 — indistinguishable — and `promoter_all` ID has a *lower* hit rate
than its Near-OOD counterpart under both (15.7 < 24.0, 23.0 < 35.3). The
taxonomic rows are ordered correctly under both. This supports treating the
alignment statistics as descriptive rather than as the category criterion,
which is how `configs/tasks.yaml` now frames them.

**Action:** regenerate Table 2 with the fixed `blast.py` once BLAST is
available. Expect the `pbsim→id_novel_genus` precision to move from 0.0 to a
non-zero value, and consider dropping the precision column for pairs whose
label spaces are disjoint by construction, where 0.0 carries no information.

## Verification status

What was checked against the published figures, by recomputing from the raw
prediction files:

| artefact | status |
|---|---|
| Fig 4 (regulatory ECE vs error) | reproduces; 96 values, mean 0.03 pp; is the 5-seed mean |
| Fig 5 (regulatory reliability) | reproduces against seed 42, not the 5-seed mean |
| Fig 6 (NT AUROC heatmap) | reproduces to ≤0.001 in 20 of 21 cells; see item 7 |
| Fig 7 (metagenomic ECE vs error) | regenerated; not separately checked against published values |
| Fig 8 (metagenomic reliability) | reproduces against seed 42 in 23 of 24 cells; see item 8 |
| Fig 9 (CARMANIA AUROC heatmap) | reproduces to three decimals |
| Fig 10, MMseqs2 panels | reproduces exactly (ECE to 0.0 pp, slope to 3 dp) |
| Fig 10, Kraken2 panels | **does not reproduce**; result spans 5.1–45.4% ECE on the same file depending on an undocumented mapping choice. See item 9 |
| Table 1 (model descriptions) | nothing to compute |
| Table 2 (BLAST statistics) | **not regenerable here** (no BLAST binaries, no retained output). Cross-checked with MMseqs2: precision agrees to ≤1.5 points on every pair except `pbsim→id_novel_genus`, where the published 0.0 is confirmed as an artefact against 20.6. See item 10 |

Three different seed conventions appear across the manuscript's own figures:
Fig 4 is the 5-seed mean, Figs 5/8/9 are seed 42, and Fig 10 uses seed 1
(`inference_results_1` in the source notebook).
