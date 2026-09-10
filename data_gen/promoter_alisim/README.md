# promoter_alisim

Epistemic-shift eval set: real `promoter_all` test sequences evolved away
from themselves with AliSim (IQ-TREE2), HKY85+Gamma4, at a grid of branch
lengths (expected substitutions/site). Same label space as `promoter_all`
throughout, so accuracy stays meaningful as a sanity check even at high
divergence.

Requires `iqtree2` on `PATH`.

## Scripts

- `make_alisim_data.py` — base grid, writes `csv_data/promoter_alisim.csv`.
- `make_alisim_data_dense.py` — denser grid (0–3.0), `..._dense.csv`.
- `make_alisim_data_extended.py` — extends the dense grid to 12.0 once
  identity-to-anchor hadn't reached saturation by 3.0.

Each writes a `..._params.json` alongside its CSV recording the exact
model/seed/grid used.

## Output columns

`name, label, branch_length, sequence, realized_identity_to_anchor`

Registered in `configs/experiments.yaml` as the `promoter_alisim` test set
under `promoter_all`.
