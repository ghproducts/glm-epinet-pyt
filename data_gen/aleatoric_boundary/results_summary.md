# Boundary-sampling aleatoric/epistemic results

## Sanity check: accuracy by margin quartile (independent of any method's own U)

| quartile   |   n |   accuracy |   mean_margin |
|:-----------|----:|-----------:|--------------:|
| Q1         | 396 |     0.6742 |        0.4194 |
| Q2         | 396 |     0.8737 |        0.8627 |
| Q3         | 396 |     0.9773 |        0.9678 |
| Q4         | 396 |     0.9924 |        0.9887 |


## Per-method, per-quartile mean U_aleatoric / U_epistemic(-analogue), Q1 vs Q4 Mann-Whitney U

`spearman_rho_vs_margin` is the rank correlation between the continuous margin score and the uncertainty score across all 1584 examples (negative = score falls as margin rises, i.e. rises toward the boundary), supplementary to the Q1-vs-Q4 comparison.

| method         | score       |   Q1_mean |   Q2_mean |   Q3_mean |   Q4_mean |   Q1_n |   Q4_n |   mwu_U |      mwu_p | direction   |   spearman_rho_vs_margin |   spearman_p |
|:---------------|:------------|----------:|----------:|----------:|----------:|-------:|-------:|--------:|-----------:|:------------|-------------------------:|-------------:|
| cnn_ensemble   | U_aleatoric |    0.4613 |    0.3407 |    0.1404 |    0.0271 |    396 |    396 |  154512 | 1.465e-123 | Q1 > Q4     |                  -0.7747 |   3.336e-317 |
| cnn_ensemble   | U_epistemic |    0.2245 |    0.1493 |    0.0438 |    0.0051 |    396 |    396 |  153537 | 1.825e-120 | Q1 > Q4     |                  -0.7468 |   1.62e-282  |
| cnn_mc_dropout | U_aleatoric |    0.4365 |    0.3466 |    0.16   |    0.0599 |    396 |    396 |  147496 | 3.573e-102 | Q1 > Q4     |                  -0.6609 |   1.675e-199 |
| cnn_mc_dropout | U_epistemic |    0.1092 |    0.0874 |    0.0365 |    0.0106 |    396 |    396 |  146046 | 5.205e-98  | Q1 > Q4     |                  -0.6316 |   4.42e-177  |
| conv_epinet    | U_aleatoric |    0.7116 |    0.4021 |    0.1819 |    0.0774 |    396 |    396 |  156816 | 4.937e-131 | Q1 > Q4     |                  -0.9522 |   0          |
| conv_epinet    | U_epistemic |    0.1592 |    0.0875 |    0.037  |    0.0089 |    396 |    396 |  156320 | 2.093e-129 | Q1 > Q4     |                  -0.8379 |   0          |
| ensemble_k3    | U_aleatoric |    0.8045 |    0.3785 |    0.1256 |    0.0538 |    396 |    396 |  156816 | 4.937e-131 | Q1 > Q4     |                  -0.986  |   0          |
| ensemble_k3    | U_epistemic |    0.0199 |    0.0103 |    0.001  |    0      |    396 |    396 |  156532 | 4.232e-130 | Q1 > Q4     |                  -0.8577 |   0          |
| ensemble_k5    | U_aleatoric |    0.7971 |    0.3683 |    0.1212 |    0.0533 |    396 |    396 |  156816 | 4.937e-131 | Q1 > Q4     |                  -0.9861 |   0          |
| ensemble_k5    | U_epistemic |    0.0275 |    0.0119 |    0.0011 |    0.0001 |    396 |    396 |  156789 | 6.057e-131 | Q1 > Q4     |                  -0.9142 |   0          |
| evidential     | U_aleatoric |    0.8785 |    0.6666 |    0.5287 |    0.4905 |    396 |    396 |  156816 | 4.936e-131 | Q1 > Q4     |                  -0.9097 |   0          |
| evidential     | vacuity     |    0.6007 |    0.3663 |    0.2389 |    0.2113 |    396 |    396 |  156816 | 4.936e-131 | Q1 > Q4     |                  -0.9137 |   0          |
| laplace        | U_aleatoric |    0.7989 |    0.4078 |    0.1934 |    0.1024 |    396 |    396 |  156816 | 4.937e-131 | Q1 > Q4     |                  -0.9727 |   0          |
| laplace        | U_epistemic |    0.0593 |    0.052  |    0.0465 |    0.0281 |    396 |    396 |  133473 | 1.355e-65  | Q1 > Q4     |                  -0.4882 |   1.237e-95  |
| mc_dropout     | U_aleatoric |    0.726  |    0.5406 |    0.2502 |    0.0668 |    396 |    396 |  156815 | 4.974e-131 | Q1 > Q4     |                  -0.8858 |   0          |
| mc_dropout     | U_epistemic |    0.0488 |    0.0379 |    0.025  |    0.001  |    396 |    396 |  155891 | 5.25e-128  | Q1 > Q4     |                  -0.7148 |   5.057e-248 |
| rf             | U_aleatoric |    0.7339 |    0.6576 |    0.495  |    0.2261 |    396 |    396 |  154787 | 1.931e-124 | Q1 > Q4     |                  -0.8063 |   0          |
| rf             | U_epistemic |    0.1732 |    0.1669 |    0.1484 |    0.1101 |    396 |    396 |  125656 | 9.049e-49  | Q1 > Q4     |                  -0.443  |   4.198e-77  |

