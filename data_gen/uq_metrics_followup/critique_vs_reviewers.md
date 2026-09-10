# Skeptical cross-check: this session's work vs. the actual PLOS reviews

Source for reviewer text: `.docs/PLOS_review.html` (read directly, not via the
project's own paraphrase). Source for "what this session actually did": the
`worktree-fix-epinet-batch-z` branch — commit `fb38241` (the per-example-`z`
fix to `nn_proj/models/epinet/epinet.py`), `data_gen/promoter_alisim/README.md`,
`data_gen/aleatoric_boundary/README.md`, and `data_gen/uq_metrics_followup/`
(`README.md`, `results_summary.md`, `paper_tables.tex`). Cross-referenced
against `.docs/REVISION_PLAN.md`, `.docs/RESUBMISSION_PLAN.md` (point numbers
R1-x/R2-x below match these two files), `.docs/FINDINGS.md`,
`.docs/MODEL_CODE_FIXES.md`, `.docs/ALEATORIC_EPISTEMIC_CROSSCHECK.md`,
`.docs/ALL_METHODS_OOD_SEVERITY.md`.

**Framing note before the point-by-point**: `RESUBMISSION_PLAN.md` was last
updated 2026-09-02. Everything this session did — the batch-`z` fix, both new
datasets, `uq_metrics_followup/` — postdates it and is **not reflected
anywhere in that plan's status table**. Any claim that a point is "resolved"
per that document should be read as stale by construction; this report
supersedes it for the items touched below.

---

## Reviewer #1

### R1-main / R1-practical-value — lacks biological insight, limited practical merit of proposed methods
**Status: Still open.** Nothing this session touches biological interpretation
or makes a stronger practical case for epinets/UQ over plain fine-tuning. If
anything, the new findings below (decomposition doesn't separate cleanly even
for true ensembles) cut the other way — toward *less* practical merit, not
more. This is a rewrite/reframing problem (R2-4/R2-10 territory), not one any
new dataset can fix.

### R1-1 (line 157-158) — Near-ID/Near-OOD/OOD/far-OOD terms undefined
**Status: Still open**, for the manuscript's original categories (taxonomic,
gene-taxa, regulatory cross-task). The two new datasets this session built
(`promoter_alisim`, `aleatoric_boundary`) sidestep the problem rather than
solving it: both define a *new*, cleanly continuous severity axis
(branch length; margin quartile) for a *task the manuscript doesn't use this
way* (`promoter_all`, same-task/same-label-space). That's a good template for
what a defensible axis looks like, and could inform how to answer R1-1 for the
original categories, but it is not itself an answer to "what's the
ID/Near-ID/Near-OOD/OOD rule for `novel_genus`, `promoter_all→enhancers`,
etc." No such rule was written this session. `RESUBMISSION_PLAN.md` item #1's
own three-part plan (cite `make_splits.py` for taxonomic, one sentence for
gene-taxa, **write a new criterion for regulatory cross-task pairs** — flagged
there as real new intellectual work) is untouched.

### R1-2 (line 204-211) — how was "accuracy" defined for OOD tasks (affects ECE)
**Status: Still open.** Not addressed by this session's work. The new
datasets sidestep this too, by construction — `promoter_alisim` and
`aleatoric_boundary` both keep the *same* label space as `promoter_all`
throughout, so "accuracy" is unambiguous there — but that says nothing about
how accuracy was defined for the manuscript's actual cross-task and
taxonomic-domain pairs, which is what the reviewer asked about.

### R1-3 (line 466 onward) — far-OOD can be as hard as near-OOD; no measured distance axis
**Status: Partially resolved, but on the wrong data to close the point as
scoped.** `promoter_alisim`'s branch-length axis and `aleatoric_boundary`'s
margin-quartile axis are exactly the kind of "actual measured axis instead of
a qualitative label" the reviewer and `RESUBMISSION_PLAN.md` item #2 call for
— continuous, monotonic (`realized_identity_to_anchor` from 1.0 to ~0.50;
margin from ~1.0 to ~0.0), with built-in sanity checks (accuracy collapses
monotonically with both). This is genuinely good methodology. But it is built
entirely on `promoter_all`/DNABERT2 — a *new* case study, not a rerun of the
manuscript's actual flagged pairs (the taxonomic/pbsim and gene-taxa domains
where the original "far-OOD as hard as near-OOD" concern was raised).
`RESUBMISSION_PLAN.md` item #2 explicitly requires rerunning the *original*
BLAST-based hit-rate/coverage analysis against error/ECE/AUROC on the
manuscript's own train/test pairs; that has still not been done. Same
category of gap as R1-1: the session proved the *method* for building a
measured-distance axis, on data that didn't exist in the original manuscript,
and has not yet applied it (or an equivalent) to the pairs the reviewer
actually flagged.

### R1-4 / #8 — Fig 9 base AUROC <0.5 for `novel_genus` vs `ood_nonbacterial`, suspected training bug
**Status: Newly complicated. The code fix is real; the reviewer's actual empirical
question is still unanswered.**

Verified by reading the diff (`git show fb38241`, not just the commit
message): `GaussianIndexer.forward()` now returns `[B, Dz]` (one `z` per
batch example) instead of `[Dz]` (one shared by the whole batch), threaded
correctly through `ProjectedMLP` (`einsum('bcd,bd->bc', ...)` replacing
`einsum('bcd,d->bc', ...)`), `FixedConv1DPriorEnsemble`'s new `basis()`/
`contract()` split, and both the single- and multi-sample paths of
`EpinetWrapper.forward()`. This is a genuine, correctly-implemented fix for
a real defect, not a cosmetic change. The commit's own toy-model verification
(a small from-scratch conv classifier, not a pretrained GLM) is methodologically
sound as a check that the *mechanism* works: per-example `z` now visibly
varies row-to-row, OOD AUROC on a synthetic homopolymer-vs-real task is
0.94-1.00, and run-to-run AUROC variance drops ~10x vs. a monkeypatched
reproduction of the old shared-`z` behavior.

But this is exactly where `RESUBMISSION_PLAN.md`'s own two-stage plan for
item #8 gets skipped over, not completed:

- **Stage (a)** was specified as: "cheap targeted confirmation — patch
  inference-time sampling on the ~8-10 flagged cells only, no retraining, see
  if AUROC moves." **This was never done.** The flagged cell is Fig 9's
  CARMANIA `id_novel_genus → ood_nonbacterial` AUROC of 0.482, a taxonomic
  pbsim-domain result. Nothing in this session touches CARMANIA, the
  taxonomic domain, or that checkpoint family — the fix was validated on a
  toy model and then deployed on two brand-new DNABERT2/`promoter_all`
  datasets that did not exist in the original manuscript. The actual number
  the reviewer complained about has not moved, because it has not been
  recomputed under the fix at all.
- **Stage (b)** (decide whether to propagate the fix and regenerate every
  epinet result in the manuscript) is explicitly still an open decision in
  `RESUBMISSION_PLAN.md`, and nothing this session did resolves that decision
  — if anything it makes the case for propagating *stronger* (a real batch-
  correlation bug existed and inflated MC noise) while making the *cost* of
  propagating clearer too (every original checkpoint this would need to rerun
  against no longer exists on disk — see R2-9 below), which likely pushes the
  decision toward "document as a limitation" rather than "regenerate."

**So: is R1-4/#8 honestly "resolved"?** No — it is resolved *prospectively*
(future epinet work in this repo will not have this bug), but the manuscript's
own reported number that triggered the concern is untouched, and the specific
confirmation step the project's own plan called for as the minimum bar was
substituted with a toy-model demonstration on a different architecture and a
different task domain. Calling this "R1-4 fixed" in a response-to-reviewers
letter would overstate what was actually verified. The honest framing is: "we
found and fixed a real bug in how the index sample is drawn; we did not
re-verify it against the specific cell the reviewer flagged, because that
checkpoint no longer exists; we now document the batch-shared-z mechanism as
a limitation of the originally-reported numbers." That is a materially weaker
claim than "fixed."

There's a further complication: the fix's own downstream results (see R2-5
below) show that even *with* per-example `z` — and even for architectures
that never had this bug at all (true independent ensembles) — the epistemic
signal still doesn't cleanly separate from aleatoric or from margin/difficulty.
That's evidence the shared-`z` bug was not the primary cause of the
decomposition's poor behavior generally, even if it was a real contributor to
noise in any single AUROC comparison like Fig 9's. This narrows what fixing
it actually buys the manuscript's broader UQ claims.

---

## Reviewer #2

### R2-1 — cross-task OOD evals change the prediction target
**Status: Still open for the manuscript's actual pairs; a second good
same-target replacement now exists but isn't wired in.** `promoter_alisim`
and `aleatoric_boundary` are both same-task, same-label-space shift axes
(exactly what R2-1 asks for), joining `promoter_motifs_v3` as a third such
dataset built this project-cycle. Three same-target replacement datasets now
exist and none of them has been inserted into the manuscript text, and
`REVISION_PLAN.md`'s original decision (keep the 3 cross-task pairs as
AUROC-only, add one same-target replacement with a limitations paragraph) has
not been revisited to account for having three candidates instead of one, or
updated to prefer the newer, better-controlled ones over `promoter_motifs_v3`
(which per `RESUBMISSION_PLAN.md` item #3 had known weaknesses: small
resample, epistemic detector at-or-below chance). Arguably `aleatoric_boundary`
and `promoter_alisim` are *more* defensible replacements than
`promoter_motifs_v3` (larger n, continuous graded axis with monotonic sanity
checks) — but no decision has been made about which one(s) to actually use,
and the manuscript itself hasn't changed.

### R2-2 — ID/Near-ID/Near-OOD/OOD not reproducibly defined
**Status: Still open**, same underlying gap as R1-1. Not touched this session
beyond, again, providing a good template (continuous branch-length / margin
severity) for a task outside the manuscript's actual disputed categories.

### R2-3 — improved ECE doesn't establish adaptive uncertainty; need accuracy-stratified-by-uncertainty
**Status: Partially addressed, but on the wrong slice of data to close the
point as the reviewer scoped it.** `aleatoric_boundary`'s margin-quartile
accuracy table (Q1 0.674 → Q4 0.992, monotonic, computed independent of any
method's own uncertainty score) is a textbook version of exactly what R2-3
asks for. But `RESUBMISSION_PLAN.md` item #7 is explicit that the reviewer's
"60-85% error even when ECE improves" complaint is about the manuscript's
*original hard taxonomic cells* — `promoter_all` (86-99% accuracy across
quartiles) is nowhere near that regime. This session's accuracy-stratified
analysis is real, competently done, and reusable as a template, but it
evaluates the easy case, not the case the reviewer actually flagged as
concerning. Item #7's original scope ("genuinely not done yet, no new code
needed, just new analysis on existing prediction outputs" — i.e., run the
already-existing risk-coverage infra on the hard taxonomic cells) remains
undone.

### R2-4 — none of the methods work reliably under shift; narrow the claims
**Status: Reinforced, not newly resolved, and the honest direction of travel
is toward a stronger version of the reviewer's complaint than before.** No
rewrite happened this session (this was always scoped as writing-only). But
the new evidence base makes the "narrow the claims" argument considerably
easier to make and harder to avoid: two new, independently-designed axes both
show every method's uncertainty signal responding to changes it theoretically
should not respond to (see R2-5), `rf`/`laplace` are consistently the
*best*-behaved methods in the new decomposition tables while `conv_epinet`
(the manuscript's headline UQ method) and true deep ensembles are among the
*worst*-conflating. If this material makes it into the manuscript, it
strengthens R2-4's case, it does not answer it — the abstract/title still
need the narrowing pass `RESUBMISSION_PLAN.md` item #4 already scoped.

### R2-5 — aleatoric/epistemic decomposition insufficiently justified
**Status: Substantially reinforced with strong new evidence — but this
surfaces a problem for other parts of the manuscript, not just a clean
resolution of R2-5 in isolation.**

This is the strongest actual result from this session. Two independently
designed, well-controlled axes were tested:
- `promoter_alisim` (epistemic-only manipulation: real evolutionary distance
  from a fixed anchor, label never manipulated) — `U_epistemic` should rise,
  `U_aleatoric` should stay flat. Result: epistemic barely moves for most
  methods (only `rf` shows a real, clean rise, ρ=0.563); aleatoric rises
  *at least as much* as epistemic for the epinet, and both drift for several
  methods — the "aleatoric stays flat" half fails for essentially the whole
  9-method roster.
- `aleatoric_boundary` (aleatoric-only manipulation: margin-based difficulty
  on real, fully in-distribution sequences, no input novelty at all) —
  `U_aleatoric` should rise, `U_epistemic` should stay flat. Result:
  aleatoric behaves correctly for every method (ρ = -0.66 to -0.99), but
  epistemic rises significantly for **all 9** method-configurations tested,
  including two genuine independently-trained deep ensembles
  (`ensemble_k5`/`ensemble_k3`), which show the *worst* relative epistemic
  inflation of the entire roster (~275-460x rise, ρ up to -0.914) — nearly as
  strong as their own aleatoric response.

That ensemble result is important and is the sharpest evidence yet for R2-5:
deep ensembles are the textbook case where the formal posterior-predictive
decomposition is supposed to hold up best (independently trained models, no
shared-weights sampling artifact), and even they fail to keep epistemic flat
under a pure label-difficulty manipulation. This directly answers — in the
reviewer's favor — the question of whether the entanglement is a sampling-
mechanism artifact (epinet index draws, dropout masks) or something more
fundamental about applying this decomposition to fine-tuned classifiers on
real data. **It is not merely a sampling-mechanism artifact.**

**Where this newly complicates things, not just closes R2-5**: this result
sits in tension with `.docs/ALEATORIC_EPISTEMIC_CROSSCHECK.md`'s prior,
narrower conclusion from earlier this project-cycle — that the confound
"tracks resampling *one* fitted model... and is largely absent when the K
samples come from independently trained models." This session's
`aleatoric_boundary` extension directly contradicts that narrowing: on the
margin-quartile axis, the independently-trained ensembles are the *worst*
performers, not the best. Neither this new critique document nor
`RESUBMISSION_PLAN.md`/`REVISION_PLAN.md` reconciles the two findings — a
reader who only saw the earlier crosscheck doc would draw the wrong
conclusion about ensembles being a fix. This should be flagged explicitly in
any manuscript text that cites the earlier, more optimistic narrowing.

More broadly: **if the decomposition doesn't work even for true ensembles**,
that retroactively weakens any place elsewhere in the manuscript (or in this
session's own `promoter_alisim`/`ALL_METHODS_OOD_SEVERITY.md` results) that
reports an "epistemic AUROC" or "epistemic rises with severity" finding as if
it specifically indicates novelty-detection rather than just being a noisier,
correlated copy of total/aleatoric uncertainty. `ALL_METHODS_OOD_SEVERITY.md`
reports `conv_epinet`'s epistemic component as "the best-behaved epistemic
signal of the three sampling-based methods" on a severe (shuffled/random-DNA)
shift — that finding is not contradicted by the new results (severe input
novelty is a different regime from margin-based difficulty), but the two
documents together should make any manuscript text more careful to say
"epistemic separates well under severe corruption, but the same score also
tracks pure label difficulty when the input is unchanged" rather than
treating "epistemic" as a clean novelty detector anywhere.

### R2-6 — NLL/Brier, bin sensitivity, seed variability, K justification/convergence
**Status: Partially addressed, only for brand-new datasets, and only two of
the four sub-asks.** `uq_metrics_followup/`'s Analysis 2 computes ECE, ACE
(equal-mass bins), NLL, and Brier with a bin-count sensitivity sweep
(10/15/20/25 bins) — this is real, competent work directly matching what R2-6
asked for. But:
- It covers exactly **two** method/axis cells with a persisted probability
  column (`base` on the margin-boundary axis; `evidential` on the OOD-severity
  axis) — the `README.md` is explicit that "almost none" of this session's
  own new scripts saved the raw probability needed for NLL/Brier, so ~28 of
  ~30 method×axis combinations this session generated **cannot** be scored
  this way at all without new inference runs, which were explicitly not run.
- **Seed variability is not addressed anywhere in this deliverable.** The
  `README.md` states plainly: "neither ECE-eligible dataset has more than one
  seed... Multi-seed ECE... cannot be produced from any existing data in this
  project." This is the *exact* defect `FINDINGS.md` #1 already documented
  for the original manuscript (Fig 5/8/9 report seed-42-only ECE dressed up as
  a 5-seed mean) — and this session's new work reproduces the same
  single-seed limitation in its own new analysis rather than fixing it.
- **K-convergence (K=10 justification, stability of ECE/AUROC/epistemic
  component as K increases) is not touched at all** this session. Every new
  script still hardcodes K=10 (or K=16 for MC-dropout/Laplace, inherited from
  prior-session conventions, not newly justified). `RESUBMISSION_PLAN.md`
  item #9's K-sweep (K∈{10,20,50,100}) was never run.
- None of this touches the **original manuscript's** reported ECE numbers,
  which is what R2-6 was actually about — Fig 5/8/9's single-seed-as-5-seed-
  mean substitution (`FINDINGS.md` #1) is still unfixed. `RESUBMISSION_PLAN.md`'s
  own item #9 language ("K-convergence sweep... NLL/Brier/bin-sensitivity are
  already computed and just need surfacing") reads as more finished than the
  current evidence supports — "surfacing" implies the original-manuscript
  numbers already have this available; they don't, only two cells of
  brand-new data do.

### R2-7 — missing baselines/ablations (deep ensembles, epinet capacity ablation, simpler-architecture baseline)
**Status: Meaningfully advanced for two of three sub-items, on new data only;
the harder sub-item (epinet hyperparameter ablation) untouched.**
- **11a (deep ensembles)**: genuinely done, and done well — both
  `promoter_alisim` and `aleatoric_boundary` include true 5-seed and 3-seed
  DNABERT2 deep ensembles as a first-class comparison method, not an
  afterthought.
- **11c (simpler-architecture baseline)**: also genuinely done — a
  from-scratch CNN (`SmallCNN`, DeepBind/DeepSEA-lineage, no pretraining) and
  a k-mer + Random Forest baseline both appear across both new datasets,
  trained fresh, evaluated under the same harness. This directly answers
  R1/R2's "why only GLMs" concern for at least one task
  (`promoter_all`) — and interestingly, both non-pretrained baselines turn
  out to be among the *best*-calibrated-under-shift methods in the new
  results (`rf` wins the `promoter_alisim` calibration table outright at
  every non-trivial branch length; see `paper_tables.tex` Table
  `tab:alisim-calibration`).
- **11b (epinet hyperparameter ablation — `prior_scale=0`,
  `conv_prior_scale=0`, matched-capacity deterministic head)**: **not
  touched at all.** No occurrence of a zeroed prior scale or a matched-capacity
  control anywhere in this session's new READMEs or CSVs. This is still the
  single biggest unaddressed sub-item of R2-7, and it's the sub-item that
  most directly answers the reviewer's "architecture/capacity confound"
  objection (i.e., is the epinet's ECE improvement about epistemic modeling
  or just generic capacity/shrinkage) — precisely because it's the one that
  needs new training runs rather than aggregation of what already exists.

### R2-8 — Kraken2/MMseqs2 ECE comparison not meaningful
**Status: Untouched.** Nothing in this session's file set relates to
Kraken2/MMseqs2. `RESUBMISSION_PLAN.md` item #10 remains exactly as scoped
2026-09-02.

### R2-9 — reproducibility: hand-edited templates, not a complete pipeline
**Status: Newly complicated — net effect this session is worse, not better,
relative to what the reviewer asked for**, despite the genuinely valuable
underlying analysis. Concretely:
- **The batch-`z` fix was hand-copied between worktrees with `cp`/`git show`,
  not merged or released.** Per `aleatoric_boundary/README.md`: "`nn_proj/
  models/epinet/epinet.py` in this worktree was overwritten with the fixed
  version from branch `worktree-fix-epinet-batch-z`... This is the one
  exception to this task's... scope." That's a manual, undocumented-outside-
  a-README file substitution across git worktrees — exactly the kind of
  provenance gap R2-9 originally flagged, applied to the fix that is supposed
  to be this session's headline correctness improvement. As of this report,
  the fix exists on one feature branch and has been manually copy-pasted into
  at least two other worktrees' working directories; it is not merged to
  `main`, not in the version of `epinet.py` any of the four backbones'
  `train_epinet.py`/`inference.py` would import by default, and one of the
  two new datasets' follow-up pass (`aleatoric_boundary`'s 6-new-method
  extension) explicitly notes it *left the unfixed version in place* because
  "none of the 6 new methods touch epinet" — meaning the same worktree
  contains both fixed and unfixed `epinet.py` content across its history,
  by design, tracked only in prose.
- **`data_gen/label_noise/` — the data source for a large fraction of this
  session's headline numbers (`ALL_METHODS_OOD_SEVERITY.md`,
  `ALEATORIC_EPISTEMIC_CROSSCHECK.md`, and both new datasets' checkpoint
  provenance) — is entirely untracked.** It does not exist on any git branch,
  including `worktree-fix-epinet-batch-z` (confirmed: `git ls-tree` on that
  branch returns nothing under this path). It lives only as local files in
  one machine's checkout, was itself copied file-by-file into other worktrees
  during this session ("copied byte-for-byte with plain `cp`... None of this
  content was reconstructed from memory; every copied file was read from disk
  first" — `aleatoric_boundary/README.md`). Two-plus dozen one-off scripts and
  CSVs (`decomp_compare_label_noise_*.py` × 6+ variants, `all_methods_ood_*`,
  `cnn_scratch.py`, etc.) exist only this way.
- **The original manuscript's checkpoints are gone.** Both new-dataset
  READMEs state plainly: "The original manuscript's `trained_models_*/
  DNABERT2/promoter_all` checkpoint no longer exists on disk." Every new
  result this session produced substitutes `checkpoints/seed_*/DNABERT2/
  label_noise_r00/{base,epinet,evidential,...}` — a checkpoint family trained
  for an entirely different (label-noise) investigation, standing in for a
  checkpoint that can no longer be regenerated from the manuscript's own
  described pipeline (no seed/config record survives, only "same task, same
  architecture" is asserted as equivalence). If a reviewer or future
  maintainer asked "reproduce Table X's actual number," the honest answer for
  every DNABERT2/`promoter_all` result touched this session is "that exact
  checkpoint is gone; here is a same-recipe substitute."
- **`aleatoric_boundary` is explicitly not wired into `configs/experiments.yaml`
  or `configs/tasks.yaml`** ("Judgment call: adding a stub entry there... would
  misrepresent this as part of the formal grid"). `promoter_alisim` is
  partially wired (one `tests:` CSV entry), but its actual uncertainty-eval
  scripts (`run_uncertainty_eval.py`, `run_new_methods_dense.py`) are
  standalone, not invoked by `scripts/run_grid.py`, and hardcode absolute
  checkpoint paths (`/scratch/home/glh52/glm-epinet-pyt/checkpoints/...`)
  specific to one machine.
- **Environment fragility, documented but not resolved**: both new-dataset
  READMEs describe a real, nontrivial 3-way `transformers`/Python-version/
  triton conflict needed to load DNABERT2 checkpoints at all, worked around
  with a bespoke venv (`aleatoric_boundary_venv`) pinned to specific point
  releases discovered empirically during this session, not captured in any
  requirements file or repo-level environment spec.

Put together: this session added real analytical value, but essentially all
of it lives in per-session, cross-worktree, `cp`-and-`git show`-glued
scaffolding on top of a checkpoint family that already doesn't match what the
manuscript reports, run in a hand-tuned venv, using a bugfix that is itself
manually propagated file-by-file rather than merged. If a reviewer re-read
R2-9 today, "the supplied scripts are manually edited templates for
individual runs rather than a complete specification of the experimental
grid" would still land, and arguably lands *harder* now — there are simply
more one-off scripts and cross-worktree file-copies to point to than before
this session started. `RESUBMISSION_PLAN.md` item #6 marking the code-side
fix as "DONE" (referring to `run_grid.py`/`REPRODUCE.md`/prior-session fixes)
does not reflect any of this session's new work, which sits entirely outside
that "DONE" infrastructure.

### R2-10 — manuscript needs rewriting (terminology, notation, figure consolidation)
**Status: Untouched**, as expected (this was always the last-sequenced,
writing-only item). Not in scope for this session and not claimed to be.

### Data/code availability (Reviewer #2: "No")
**Status: Not improved, arguably worse to explain.** See R2-9 above — more
untracked, cross-worktree, `cp`-propagated content now exists than before.
Anyone trying to answer "is all data and code underlying the findings fully
available" today would have to additionally explain why `data_gen/label_noise/`
(a load-bearing input to several of this session's headline findings) isn't
in git at all, on any branch.

---

## Compact status summary

| Point | Status |
|---|---|
| R1-main (bio insight/practical value) | Still open |
| R1-1 (shift categories undefined) | Still open (new datasets sidestep, don't answer, the original categories) |
| R1-2 (accuracy definition for OOD/ECE) | Still open |
| R1-3 (far-OOD as hard as near-OOD; no measured axis) | Partially resolved — good method, wrong (new, not original) data |
| R1-4 / #8 (batch-shared-z bug) | Newly complicated — code fix is real and correct; the reviewer's actual flagged number (Fig 9, CARMANIA) was never recomputed; toy-model verification substituted for the plan's own required targeted confirmation |
| R2-1 (cross-task target mismatch) | Still open — two more good same-target datasets exist now, none wired into the manuscript, decision among 3 candidates undecided |
| R2-2 (categories not reproducible) | Still open |
| R2-3 (accuracy-stratified adaptivity) | Partially addressed — real analysis, but on an easy task, not the flagged 60-85%-error taxonomic cells |
| R2-4 (narrow the claims) | Reinforced by new evidence, not itself resolved (writing still pending) |
| R2-5 (decomposition validity) | Substantially reinforced — strong new evidence decomposition fails even for true ensembles; also newly complicates other epistemic-based claims and contradicts an earlier, more optimistic in-repo finding |
| R2-6 (ECE rigor: bins/NLL/Brier/seed/K) | Partially addressed — bins+NLL+Brier done for 2 of ~30 new cells; seed variability and K-convergence untouched; original manuscript's seed-42-as-5-seed-mean defect (`FINDINGS.md` #1) still unfixed |
| R2-7 (baselines/ablations) | Two of three sub-items meaningfully advanced (ensembles, simple baselines); epinet hyperparameter ablation (the one most relevant to the capacity-confound objection) untouched |
| R2-8 (Kraken2/MMseqs2) | Untouched |
| R2-9 (reproducibility) | Newly complicated — net reproducibility posture is worse: more untracked one-off scripts, cross-worktree `cp` file-shuffling, a checkpoint substitution for manuscript checkpoints that no longer exist, and a bugfix propagated by hand rather than merged |
| R2-10 (rewrite) | Untouched (as scoped) |
| Data/code availability ("No") | Not improved |

## Top 3 places `RESUBMISSION_PLAN.md` is too optimistic right now

1. **Item #8 (R1-4)** — even under its own pre-session framing this was
   flagged as "genuinely uncertain," but the plan's stage-(a) bar ("patch
   inference-time sampling on the ~8-10 flagged cells... see if AUROC moves")
   has still not been met. A reader of the plan alongside the new commit
   message could easily conclude the confirmation step happened because a
   verification narrative exists — it happened on a synthetic toy model and
   two unrelated new datasets, never on the actual flagged Fig 9 cell. The
   plan needs a line making clear that stage (a) is still outstanding on the
   manuscript's own data, and that the checkpoint needed to run it no longer
   exists.

2. **Item #9 (R2-6)** — "NLL/Brier/bin-sensitivity are already computed and
   just need surfacing" reads as if this is a formatting/exposure task on
   existing manuscript numbers. In fact, per this session's own
   `uq_metrics_followup/README.md`, essentially none of the manuscript's
   original results have a persisted probability column to compute NLL/Brier
   from, and the new analysis that *does* compute them covers 2 of ~30 new
   method×axis cells, zero of the original manuscript's reported cells, and
   zero seeds beyond 1. "Just needs surfacing" should read "needs new
   instrumented reruns," which is a materially larger lift.

3. **Item #6 (R2-9)**, marked "Code-side fix is DONE... Remaining is
   logistics: tag a versioned release, attach raw prediction CSVs." This
   framing predates essentially all of this session's work, which added a
   large volume of new, untracked, cross-worktree-glued scripts and a
   checkpoint-substitution pattern that a "tag a release" step would now also
   need to reckon with (what exactly gets tagged — the untracked
   `data_gen/label_noise/` directory too?). Treating #6 as done-but-for-
   logistics undercounts how much new, unversioned surface area this session
   added on top of it.
