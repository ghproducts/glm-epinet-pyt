# promoter_alisim uncertainty evaluation: results summary

Dose-response of each UQ method's uncertainty decomposition against AliSim branch length (substitutions/site) away from real `promoter_all` test-split anchors. See `data_gen/promoter_alisim/README.md` for dataset construction and `data_gen/promoter_alisim/uncertainty_eval/README` section of the top-level README for method/checkpoint details.

## Dose-response: mean U_epistemic / U_aleatoric / accuracy by branch_length

### conv_epinet

|   branch_length |        n |   U_epistemic_mean |   U_epistemic_std |   U_aleatoric_mean |   U_aleatoric_std |   accuracy |
|----------------:|---------:|-------------------:|------------------:|-------------------:|------------------:|-----------:|
|          0.0000 | 400.0000 |             0.0660 |            0.0720 |             0.3297 |            0.2626 |     0.8725 |
|          0.0200 | 400.0000 |             0.0644 |            0.0716 |             0.3336 |            0.2689 |     0.8650 |
|          0.0500 | 400.0000 |             0.0659 |            0.0748 |             0.3305 |            0.2623 |     0.8875 |
|          0.1000 | 400.0000 |             0.0687 |            0.0731 |             0.3533 |            0.2702 |     0.8500 |
|          0.2000 | 400.0000 |             0.0783 |            0.0806 |             0.3740 |            0.2761 |     0.8225 |
|          0.4000 | 400.0000 |             0.0800 |            0.0769 |             0.4096 |            0.2944 |     0.7125 |
|          0.8000 | 400.0000 |             0.0866 |            0.0806 |             0.4225 |            0.2825 |     0.6175 |
|          1.5000 | 400.0000 |             0.0847 |            0.0769 |             0.4206 |            0.2631 |     0.5575 |

### evidential

|   branch_length |        n |   U_epistemic_mean |   U_epistemic_std |   U_aleatoric_mean |   U_aleatoric_std |   accuracy |
|----------------:|---------:|-------------------:|------------------:|-------------------:|------------------:|-----------:|
|          0.0000 | 400.0000 |             0.3534 |            0.1991 |             0.6374 |            0.1871 |     0.8850 |
|          0.0200 | 400.0000 |             0.3549 |            0.2002 |             0.6386 |            0.1884 |     0.8775 |
|          0.0500 | 400.0000 |             0.3451 |            0.1939 |             0.6297 |            0.1818 |     0.8875 |
|          0.1000 | 400.0000 |             0.3597 |            0.2019 |             0.6436 |            0.1891 |     0.8600 |
|          0.2000 | 400.0000 |             0.3781 |            0.2078 |             0.6622 |            0.1955 |     0.7950 |
|          0.4000 | 400.0000 |             0.3877 |            0.2143 |             0.6704 |            0.2018 |     0.6825 |
|          0.8000 | 400.0000 |             0.3697 |            0.2074 |             0.6534 |            0.1952 |     0.5750 |
|          1.5000 | 400.0000 |             0.3381 |            0.1790 |             0.6277 |            0.1731 |     0.5375 |

### mc_dropout

|   branch_length |        n |   U_epistemic_mean |   U_epistemic_std |   U_aleatoric_mean |   U_aleatoric_std |   accuracy |
|----------------:|---------:|-------------------:|------------------:|-------------------:|------------------:|-----------:|
|          0.0000 | 400.0000 |             0.0251 |            0.0375 |             0.3841 |            0.3037 |     0.8675 |
|          0.0200 | 400.0000 |             0.0282 |            0.0363 |             0.3873 |            0.3014 |     0.8625 |
|          0.0500 | 400.0000 |             0.0304 |            0.0394 |             0.3834 |            0.2986 |     0.8575 |
|          0.1000 | 400.0000 |             0.0345 |            0.0449 |             0.4176 |            0.3180 |     0.8225 |
|          0.2000 | 400.0000 |             0.0450 |            0.0586 |             0.4286 |            0.3232 |     0.7325 |
|          0.4000 | 400.0000 |             0.0375 |            0.0471 |             0.3714 |            0.3122 |     0.5975 |
|          0.8000 | 400.0000 |             0.0315 |            0.0379 |             0.3140 |            0.2640 |     0.5350 |
|          1.5000 | 400.0000 |             0.0268 |            0.0320 |             0.2794 |            0.2212 |     0.5075 |

## Spearman correlation: branch_length vs. score (per method)

| method      | score       |   spearman_rho |   spearman_p |    n |
|:------------|:------------|---------------:|-------------:|-----:|
| conv_epinet | U_epistemic |        0.1084  |    7.936e-10 | 3200 |
| conv_epinet | U_aleatoric |        0.1272  |    5.082e-13 | 3200 |
| evidential  | U_epistemic |        0.02623 |    0.1379    | 3200 |
| evidential  | U_aleatoric |        0.0193  |    0.275     | 3200 |
| mc_dropout  | U_epistemic |        0.0547  |    0.001965  | 3200 |
| mc_dropout  | U_aleatoric |       -0.05463 |    0.001992  | 3200 |

## Endpoint check: Mann-Whitney U, t=1.5 > t=0.0 (per method, per score)

| method      | score       | comparison    |         U |         p |   n_lo |   n_hi |   mean_lo |   mean_hi |
|:------------|:------------|:--------------|----------:|----------:|-------:|-------:|----------:|----------:|
| conv_epinet | U_epistemic | t=1.5 > t=0.0 | 9.331e+04 | 2.318e-05 |    400 |    400 |   0.06599 |   0.08466 |
| conv_epinet | U_aleatoric | t=1.5 > t=0.0 | 9.707e+04 | 8.765e-08 |    400 |    400 |   0.3297  |   0.4206  |
| evidential  | U_epistemic | t=1.5 > t=0.0 | 8.156e+04 | 0.3161    |    400 |    400 |   0.3534  |   0.3381  |
| evidential  | U_aleatoric | t=1.5 > t=0.0 | 8.114e+04 | 0.3639    |    400 |    400 |   0.6374  |   0.6277  |
| mc_dropout  | U_epistemic | t=1.5 > t=0.0 | 8.688e+04 | 0.01759   |    400 |    400 |   0.02511 |   0.02678 |
| mc_dropout  | U_aleatoric | t=1.5 > t=0.0 | 7.075e+04 | 0.9977    |    400 |    400 |   0.3841  |   0.2794  |
