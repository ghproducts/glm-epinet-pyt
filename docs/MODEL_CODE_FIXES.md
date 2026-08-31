# Model-code changes still to be made

Nothing in this document has been applied. The model, training and inference
code in `nn_proj/models/` and `nn_proj/common/` is untouched, so this branch
adds files only and cannot conflict with model work in progress.

Each item below was found while rebuilding the analysis pipeline, and each was
implemented and tested before being reverted out of this branch. The working
implementations are on the branch `backup/full-refactor-with-model-changes` if
you want to cherry-pick rather than retype.

Ordered by whether they can change a published number.

---

## Severity 1 — changes a reported result

### 1.1 `blast.py` compares mismatched label spaces

**This is the cause of Table 2's `novel_genus` precision of 0.0.**

`nn_proj/common/blast.py:27-34`

```python
def load_dataset_from_path(path: str, split: str):
    if path.startswith("InstaDeepAI/..."):
        return load_NT_tasks(task=task, split=split)
    else:
        return load_local_dataset(path=path)          # no rank, no taxa_df
```

Two problems in three lines.

First, `load_local_dataset` is called with **no `rank` and no `taxa_df`**, so the
taxonomic rank mapping that every experiment applies is skipped here. The BLAST
comparison runs in the raw species label space while the models are trained at
family, order, class or phylum.

Second, `encode_labels` defaults to `True`, so `class_encode_column` runs
**independently on the train file and the test file**. Class indices are
assigned from whatever distinct values each file happens to contain, so train
index *k* and test index *k* denote different taxa. `evaluate_best_hit`
(line 153) then compares them as strings:

```python
if pred_lab is not None and pred_lab == true_lab:
    n_correct += 1
```

**Fix:** pass `encode_labels=False` and thread `rank`/`taxa_df` through from new
CLI arguments.

**Evidence this is the cause.** The stored MMseqs2 searches carry labels inside
the sequence IDs, so they cannot suffer this bug. Recomputing best-hit precision
from them agrees with the published BLAST values everywhere the label spaces
genuinely coincide, and disagrees only where predicted:

| pair | BLAST (published) | MMseqs2 |
|---|---|---|
| promoter_all→promoter_all | 85.9 | 85.5 |
| splice_sites_acceptors→acceptors | 68.3 | 67.5 |
| splice_sites_acceptors→donors | 68.0 | 66.7 |
| gene_taxa→test | 99.6 | 99.9 |
| gene_taxa→taxa_out | 13.0 | 13.7 |
| **pbsim→id_novel_genus** | **0.0** | **20.6** |

Every row agrees to within 1.5 points except the one the bug predicts. Reproduce
with `scripts/check_tool_panels.py`.

**Also in the same function** (`blast.py:199`):

```python
split = train_ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="labels")
```

The seed is hard-coded to 42 while training splits on `data_seed`, which the
shell scripts set to the run seed (2 in the templates). The BLAST reference set
is therefore not the 90% the models trained on. Make it a `--seed` argument.

Note that dropping `encode_labels` also removes the `labels` ClassLabel feature
that `stratify_by_column` needs, so the split has to become unstratified.

**Impact:** Table 2's precision column must be regenerated. Consider dropping
the column entirely for pairs whose label spaces are disjoint by construction
(`ood_novel_family`, `ood_nonbacterial`, `gene_out`), where 0.0 is correct but
carries no information.

---

## Severity 2 — inert today, will bite on new data

### 2.1 The unmapped-taxa filter is a no-op in nine places

`nn_proj/common/datasets.py:100-106`

```python
ds = ds.map(lambda ex: {"labels": int(species_to_target.get(ex["taxid"], -1))})
...
if encode_labels and not isinstance(ds.features["labels"], ClassLabel):
    ds = ds.class_encode_column("labels")
```

Species missing from the lineage table become `-1`, and `class_encode_column`
then turns `-1` into a legitimate class the model is trained to predict.

The filter meant to prevent this is broken in two independent ways, in all nine
places it appears:

```
nn_proj/models/CARMANIA/train_base.py        nn_proj/models/CARMANIA/train_epinet.py
nn_proj/models/CARMANIA/scaling.py           nn_proj/models/DNABERT2/train_base.py
nn_proj/models/DNABERT2/scaling.py           nn_proj/models/NT_transformer/train_base.py
nn_proj/models/NT_transformer/scaling.py     nn_proj/models/hyenaDNA/train_base.py
nn_proj/models/hyenaDNA/scaling.py
```

```python
train_dataset.filter(lambda ex: ex["labels"] != -1)   # result discarded
```

1. HuggingFace datasets are immutable — `filter` returns a new dataset and the
   return value is thrown away, so nothing is filtered.
2. Even if assigned, it runs *after* `class_encode_column`, comparing encoded
   class indices against `-1`, which never matches.

**Fix:** filter inside `load_local_dataset`, before label encoding, and delete
the nine no-op calls. Add a `drop_unmapped: bool = True` parameter so the
behaviour is explicit, and print how many rows were dropped.

**Impact on published results: none.** I checked every pbsim prediction file:
zero rows carry label `-1` and zero predictions land on it, so the lineage table
covers the taxa actually used. The bug is real but never fired. It will fire
silently the first time a genome is used whose species is absent from the
lineage table — the model will quietly gain an "unknown" class and the
`nonbacterial` set will become partly answerable, which would corrupt exactly
the OOD-detection results.

### 2.2 MC-dropout rate is not what the manuscript states

The manuscript says MC-dropout uses `p = 0.1`. Only one backbone sets it:

| file | call |
|---|---|
| `NT_transformer/inference.py:95` | `enable_mc_dropout(model, p=0.1)` |
| `DNABERT2/inference.py:96` | `enable_mc_dropout(model)` |
| `hyenaDNA/inference.py:100` | `enable_mc_dropout(model)` |
| `CARMANIA/inference.py:115` | `enable_mc_dropout(model)` |

Three of four inherit whatever rate the pretrained config carries. CARMANIA's
config happens to default to `dropout_prob = 0.1`, so it agrees by luck; the
others were not verified.

There is measurable evidence this matters. Comparing base against mc_dropout
predictions on `pbsim_family/id_novel_genus`, seed 1:

| backbone | prediction agreement | mean abs. confidence change |
|---|---|---|
| NT_transformer | 0.732 | 0.120 |
| DNABERT2 | 0.797 | 0.017 |
| CARMANIA | 0.876 | 0.027 |
| **hyenaDNA** | **0.954** | **0.005** |

HyenaDNA's MC-dropout barely perturbs anything, which is a plausible mechanical
explanation for the manuscript's observation that "HyenaDNA is near-flat, with
essentially no systematic gains from decomposition" — rather than a finding
about the architecture.

**Fix:** set the rate explicitly in all four, log the rates actually enabled and
the number of dropout layers found, and warn when none are found. Then re-check
whether the HyenaDNA result survives.

Related, though harmless: `nn_proj/models/epinet/epinet.py:412` calls
`model.train()` for the mc_dropout branch, overriding the careful
eval-mode-plus-dropout setup done moments earlier in `inference.py`. No backbone
uses batch normalisation, so behaviour is currently identical, but it will
diverge for any backbone added later that does.

### 2.3 `predict()` discards the metadata needed to stratify results

`nn_proj/models/epinet/epinet.py:388-393`

```python
# metas = [
#     {col: dataset[col][i] for col in metadata_cols}
#     for i in range(len(dataset))
# ]
dataset = dataset.remove_columns(metadata_cols)
```

The `taxid` and `split` columns are dropped and the code that would have carried
them into the output is commented out. The prediction CSVs therefore cannot be
stratified by taxon, so questions like "is the epinet's calibration gain
concentrated in particular families?" cannot be answered without rerunning
inference.

**Fix:** uncomment and repair the metadata path, and write those columns into
the CSV. `nn_proj/analysis` will carry through any extra columns unchanged.

### 2.4 Inconsistent split seed in one file

`nn_proj/models/NT_transformer/train_epinet.py:54`

```python
split = train_dataset.train_test_split(test_size=0.1, seed=training_args.seed, ...)
```

Every other train/epinet/scaling script splits on `training_args.data_seed`.
Benign as the scripts are written, since they set `--seed` and `--data_seed` to
the same value, but it breaks silently the moment the two differ — and it would
break in the direction of fitting the epinet on the base model's validation
data.

---

## Severity 3 — performance and dead code

### 3.1 The convolutional prior is recomputed for every index sample

`nn_proj/models/epinet/epinet.py:154-170`

```python
def forward(self, input_ids, attention_mask, z):
    ...
    out = 0.0
    for i in range(self.Dz):            # Dz = 30 frozen CNNs
        h = self.nets[i]["net"](x)
        pooled = h.mean(dim=-1)
        pi = self.nets[i]["head"](pooled)
        out = out + pi * z[i]
    return out
```

`pi` does not depend on `z` at all — only the final `pi * z[i]` contraction
does. But `EpinetWrapper.forward` (line 319) calls this once per index sample,
so all 30 CNNs are re-evaluated K times per batch.

**Fix:** split into a `basis()` returning `[B, Dz, C]` computed once per batch,
and a `contract()` that takes `z` of shape `[Dz]` or `[S, Dz]`. Add a
`forward_multi()` on the epinet and have `EpinetWrapper` use it when present,
falling back to the loop otherwise.

**Measured:** numerically identical (max difference 9.5e-07 at K=100, i.e.
float32 rounding), and **35x faster at K=50** — 460 ms per batch down to 13 ms.

This matters beyond speed. Reviewer 2 asked for a convergence analysis over K,
noting that "given the architecture of their epinet model it seems higher K
would be computationally feasible". They are right, and this is why: with the
fix, K = 100 costs less than K = 10 does today.

### 3.2 `nn_proj/models/epinet/utils.py` is dead and broken

Nothing imports it (`grep -rn "epinet.utils\|from .utils import" nn_proj/`
returns nothing). It duplicates `compute_uncertainty` from
`nn_proj/common/utils.py` and would fail immediately if called:

- `line 41`: references an undefined `MODEL_NAME`
- `line 63`: calls `build_model_and_tokenizer(config)` with one argument; the
  function takes two
- `line 90`: calls `f1_score`, which is never imported

**Fix:** delete the file.

Also worth noting: `predict()` in that file defaults to `k_samples=16`, while
the live `predict()` in `epinet.py` defaults to 16 and the shell scripts pass
10. The manuscript reports K = 10. The live default is never used in practice
but is a trap for anyone calling the function directly.

---

## Severity 4 — design questions, not defects

### 4.1 The epistemic index is shared across the batch

`nn_proj/models/epinet/epinet.py:48-55`

```python
class GaussianIndexer(nn.Module):
    """z ~ N(0, I) with shape [Dz], shared across batch."""
    def forward(self, device=None, dtype=None):
        return torch.randn(self.index_dim, device=device, dtype=dtype)
```

One `z` is drawn per forward call and shared by every example in the batch.
(The comment at line 298 claims `[B, Dz]`; it is `[Dz]`.)

This is a defensible reading of the ENN formulation, but it has a consequence
worth stating in the manuscript: Monte Carlo estimation noise is perfectly
correlated within a batch. With ~30 batches you have ~30 effective noise draws,
not N. Since the ID and OOD sets are scored in **separate inference runs**, that
noise does not cancel between them — it becomes a systematic offset in the
AUROC comparison. The epistemic score, being a difference of two entropy
estimates, inherits this worst.

That is a plausible contributing mechanism for two reported findings: the
epistemic component performing below chance as a novelty detector, and the
sub-0.5 AUROC in Figure 9.

**Options:** draw `z` per example, or keep the shared draw and fix the seed
across paired ID/OOD runs so the offset cancels. Either way, state the choice.

### 4.2 No ablation over the epinet's own hyperparameters

`EpinetConfig` (`epinet.py:22-40`) fixes `prior_scale = 1`,
`conv_prior_scale = 1.0`, `index_dim = 30`, `hidden_sizes = (50,)`, and nothing
in the repository varies them. Every reported run uses `MLPEpinetWithConvPrior`
with all three additive terms at scale 1.

This is Reviewer 2's point 7: the comparison changes architecture and capacity,
not only uncertainty modelling. The conv prior adds a frozen random function of
the input to the logits, which would shrink confidence roughly uniformly —
their "generic confidence shrinkage" hypothesis.

The refutation is cheap because these are already parameters: run with
`prior_scale=0`, `conv_prior_scale=0`, and a matched-capacity deterministic
head. `scripts/run_grid.py` can be extended with an ablation stage.

### 4.3 Four near-identical copies of every entry point

`train_base.py`, `train_epinet.py`, `scaling.py` and `inference.py` are
duplicated across the four backbone directories with small differences:

| file | lines differing from `NT_transformer` |
|---|---|
| `scaling.py` | 4 (DNABERT2), 11 (hyenaDNA), 12 (CARMANIA) of 139 |
| `train_base.py` | 19, 23, 38 of 132 |
| `train_epinet.py` | 14, 34, 50 of 144 |
| `inference.py` | 50, 64, 78 of 138 |

What genuinely varies is small: how the config and model are constructed
(CARMANIA uses a local class), which feature function feeds the epinet
(HyenaDNA differs), where the conv prior's vocab size comes from, the tokenizer
EOS fix, and the MC-dropout rate.

This is what Reviewer 2 meant by "manually edited templates for individual runs
rather than a complete specification of the experimental grid". A backbone
registry plus shared runners takes 2521 lines to 1071 while keeping
`python -m nn_proj.models.<BACKBONE>.<entry>` working unchanged.

**This is the largest and riskiest change here, and the one most likely to
conflict with work in progress.** It is also the least urgent: it changes no
result. `tests/test_entrypoints.py` in this branch passes against both the
current code and the refactored version, so it can serve as the contract check
if you do attempt it.

---

## Suggested order

1. **1.1** `blast.py` — the only item that changes a published number.
2. **2.1** the filter no-op — cheap, and prevents a silent failure on new data.
3. **3.1** the conv prior — unblocks the K-convergence analysis a reviewer asked for.
4. **2.2** MC-dropout rates — then re-check whether the HyenaDNA result survives.
5. **3.2** delete the dead file.
6. **2.3**, **2.4** — small correctness items.
7. **4.1**, **4.2** — decide and document; these are manuscript questions as much as code.
8. **4.3** — last, when no model work is in flight.

Items 1–6 are self-contained and touch at most two files each.
