# SOP4 — Hypothesis testing (Appendix 4)

- generated 2026-10-08T14:27:32 from `classification_results_record.csv` (4488 test images)
- B = 10000 stratified bootstrap resamples, seed 42
- HueView = full_face configuration (hv_full_face). The manuscript's majority vote with mean-softmax tie-break needs softmax outputs that classification_results_record.csv does not contain yet.

Models evaluated:

- `baseline_effnet.h5  sha256 2c5aea9c4028  (2026-09-06T00:37:23)`
- `hueview_forehead_final.h5  sha256 074d336db8d9  (2026-10-06T09:14:25)`
- `hueview_left_cheek_final.h5  sha256 1683291eba17  (2026-10-06T09:16:00)`
- `hueview_right_cheek_final.h5  sha256 68403fd20a8a  (2026-10-06T09:16:29)`
- `hueview_nose_bridge_final.h5  sha256 8613c7761fc9  (2026-10-06T09:16:14)`
- `hueview_jawline_final.h5  sha256 9ee9837ddeba  (2026-10-06T09:14:44)`
- `hueview_full_face_final.h5  sha256 8f27cb56aa40  (2026-10-06T09:14:34)`
- `lab_scaler_forehead_final.pkl  sha256 9aca4fc3aacb  (2026-10-06T09:16:32)`
- `lab_scaler_left_cheek_final.pkl  sha256 14bec962c3eb  (2026-10-06T09:16:35)`
- `lab_scaler_right_cheek_final.pkl  sha256 f09c6891911d  (2026-10-06T09:16:38)`
- `lab_scaler_nose_bridge_final.pkl  sha256 3566dc836c00  (2026-10-06T09:16:36)`
- `lab_scaler_jawline_final.pkl  sha256 5270614f0d2d  (2026-10-06T09:16:34)`
- `lab_scaler_full_face_final.pkl  sha256 42b4d5f341ad  (2026-10-06T09:16:33)`

## Table 31.A — H01: Baseline vs. HueView, by illumination

| Stratum | Metric | Baseline | HueView | Delta | Statistic | p-value / BCa CI | Decision | Favours |
|---|---|---|---|---|---|---|---|---|
| Low | Accuracy | 0.6648 | 0.6321 | -0.0327 | b = 223, c = 154 | p = 0.0003735 | Reject | Baseline |
| Low | Macro Precision | 0.6649 | 0.5648 | -0.1000 | mean diff = -0.1000 | 99.58% CI: [-0.1647, +0.0016] | Fail to Reject |  |
| Low | Macro Recall | 0.5472 | 0.5260 | -0.0213 | mean diff = -0.0214 | 99.58% CI: [-0.0482, +0.0057] | Fail to Reject |  |
| Low | Macro F1-Score | 0.5464 | 0.5249 | -0.0215 | mean diff = -0.0216 | 99.58% CI: [-0.0501, +0.0097] | Fail to Reject |  |
| Medium | Accuracy | 0.6325 | 0.6151 | -0.0174 | b = 194, c = 155 | p = 0.03695 | Fail to Reject |  |
| Medium | Macro Precision | 0.6724 | 0.5781 | -0.0944 | mean diff = -0.0942 | 99.58% CI: [-0.1425, -0.0426] | Reject | Baseline |
| Medium | Macro Recall | 0.5322 | 0.5472 | 0.0150 | mean diff = +0.0152 | 99.58% CI: [-0.0149, +0.0489] | Fail to Reject |  |
| Medium | Macro F1-Score | 0.5236 | 0.5248 | 0.0012 | mean diff = +0.0015 | 99.58% CI: [-0.0326, +0.0372] | Fail to Reject |  |
| High | Accuracy | 0.5299 | 0.3806 | -0.1493 | b = 28, c = 8 | p = 0.0007529 | Reject | Baseline |
| High | Macro Precision | 0.2519 | 0.3048 | 0.0529 | mean diff = +0.0410 | 99.58% CI: [-0.0722, +0.1952] | Fail to Reject |  |
| High | Macro Recall | 0.3994 | 0.3510 | -0.0485 | mean diff = -0.0483 | 99.58% CI: [-0.1125, +0.0318] | Fail to Reject |  |
| High | Macro F1-Score | 0.2963 | 0.2744 | -0.0220 | mean diff = -0.0228 | 99.58% CI: [-0.0810, +0.0730] | Fail to Reject |  |

## Table 31.D — H03: Baseline vs. HueView, full test set

| Stratum | Metric | Baseline | HueView | Delta | Statistic | p-value / BCa CI | Decision | Favours |
|---|---|---|---|---|---|---|---|---|
| Full test set | Accuracy | 0.6446 | 0.6161 | -0.0285 | b = 445, c = 317 | p = 3.411e-06 | Reject | Baseline |
| Full test set | Macro Precision | 0.6605 | 0.5599 | -0.1006 | mean diff = -0.1006 | 98.75% CI: [-0.1321, -0.0639] | Reject | Baseline |
| Full test set | Macro Recall | 0.5358 | 0.5314 | -0.0043 | mean diff = -0.0044 | 98.75% CI: [-0.0213, +0.0130] | Fail to Reject |  |
| Full test set | Macro F1-Score | 0.5379 | 0.5279 | -0.0100 | mean diff = -0.0100 | 98.75% CI: [-0.0280, +0.0092] | Fail to Reject |  |

## Table 31.B — H02 omnibus

| metric | test | statistic | p_value | Decision |
|---|---|---|---|---|
| Accuracy (omnibus) | Cochran's Q | Q = 325.797 | 0.0000 | Reject |
| Macro Precision (omnibus) | Bootstrap range across 15 pairs | significant pairs = 8 | — | Reject |
| Macro Recall (omnibus) | Bootstrap range across 15 pairs | significant pairs = 6 | — | Reject |
| Macro F1-Score (omnibus) | Bootstrap range across 15 pairs | significant pairs = 8 | — | Reject |

## Table 31.C — H02 pairwise (Delta = Region A − Region B)

| Pair | Metric | Delta | statistic | p-value / BCa CI | Decision | Favours |
|---|---|---|---|---|---|---|
| Forehead vs. Left Cheek | Accuracy | 0.0432 | b = 643, c = 837 | p = 4.463e-07 | Reject | Left Cheek |
| Forehead vs. Left Cheek | Macro Precision | 0.0380 | mean diff = +0.0381 | 99.67% CI: [+0.0133, +0.0621] | Reject | Left Cheek |
| Forehead vs. Left Cheek | Macro Recall | 0.0463 | mean diff = +0.0464 | 99.67% CI: [+0.0158, +0.0755] | Reject | Left Cheek |
| Forehead vs. Left Cheek | Macro F1-Score | 0.0451 | mean diff = +0.0451 | 99.67% CI: [+0.0191, +0.0712] | Reject | Left Cheek |
| Forehead vs. Right Cheek | Accuracy | 0.0216 | b = 692, c = 789 | p = 0.01172 | Fail to Reject |  |
| Forehead vs. Right Cheek | Macro Precision | 0.0104 | mean diff = +0.0103 | 99.67% CI: [-0.0140, +0.0355] | Fail to Reject |  |
| Forehead vs. Right Cheek | Macro Recall | -0.0006 | mean diff = -0.0006 | 99.67% CI: [-0.0311, +0.0297] | Fail to Reject |  |
| Forehead vs. Right Cheek | Macro F1-Score | 0.0047 | mean diff = +0.0046 | 99.67% CI: [-0.0207, +0.0302] | Fail to Reject |  |
| Forehead vs. Jawline | Accuracy | 0.0305 | b = 756, c = 893 | p = 0.0007396 | Reject | Jawline |
| Forehead vs. Jawline | Macro Precision | 0.0189 | mean diff = +0.0188 | 99.67% CI: [-0.0072, +0.0468] | Fail to Reject |  |
| Forehead vs. Jawline | Macro Recall | 0.0172 | mean diff = +0.0172 | 99.67% CI: [-0.0162, +0.0518] | Fail to Reject |  |
| Forehead vs. Jawline | Macro F1-Score | 0.0245 | mean diff = +0.0245 | 99.67% CI: [-0.0031, +0.0544] | Fail to Reject |  |
| Forehead vs. Nose Bridge | Accuracy | -0.0176 | b = 842, c = 763 | p = 0.04865 | Fail to Reject |  |
| Forehead vs. Nose Bridge | Macro Precision | 0.0037 | mean diff = +0.0037 | 99.67% CI: [-0.0216, +0.0303] | Fail to Reject |  |
| Forehead vs. Nose Bridge | Macro Recall | 0.0088 | mean diff = +0.0088 | 99.67% CI: [-0.0229, +0.0412] | Fail to Reject |  |
| Forehead vs. Nose Bridge | Macro F1-Score | -0.0169 | mean diff = -0.0170 | 99.67% CI: [-0.0439, +0.0106] | Fail to Reject |  |
| Forehead vs. Full Face | Accuracy | 0.1290 | b = 543, c = 1122 | p = 1.341e-46 | Reject | Full Face |
| Forehead vs. Full Face | Macro Precision | 0.0999 | mean diff = +0.1002 | 99.67% CI: [+0.0660, +0.1342] | Reject | Full Face |
| Forehead vs. Full Face | Macro Recall | 0.0385 | mean diff = +0.0387 | 99.67% CI: [+0.0078, +0.0687] | Reject | Full Face |
| Forehead vs. Full Face | Macro F1-Score | 0.0725 | mean diff = +0.0725 | 99.67% CI: [+0.0433, +0.1005] | Reject | Full Face |
| Left Cheek vs. Right Cheek | Accuracy | -0.0216 | b = 754, c = 657 | p = 0.009816 | Fail to Reject |  |
| Left Cheek vs. Right Cheek | Macro Precision | -0.0277 | mean diff = -0.0275 | 99.67% CI: [-0.0521, -0.0031] | Reject | Left Cheek |
| Left Cheek vs. Right Cheek | Macro Recall | -0.0469 | mean diff = -0.0468 | 99.67% CI: [-0.0759, -0.0170] | Reject | Left Cheek |
| Left Cheek vs. Right Cheek | Macro F1-Score | -0.0404 | mean diff = -0.0403 | 99.67% CI: [-0.0658, -0.0147] | Reject | Left Cheek |
| Left Cheek vs. Jawline | Accuracy | -0.0127 | b = 829, c = 772 | p = 0.1544 | Fail to Reject |  |
| Left Cheek vs. Jawline | Macro Precision | -0.0192 | mean diff = -0.0193 | 99.67% CI: [-0.0445, +0.0071] | Fail to Reject |  |
| Left Cheek vs. Jawline | Macro Recall | -0.0291 | mean diff = -0.0292 | 99.67% CI: [-0.0623, +0.0053] | Fail to Reject |  |
| Left Cheek vs. Jawline | Macro F1-Score | -0.0206 | mean diff = -0.0207 | 99.67% CI: [-0.0486, +0.0081] | Fail to Reject |  |
| Left Cheek vs. Nose Bridge | Accuracy | -0.0608 | b = 910, c = 637 | p = 3.488e-12 | Reject | Left Cheek |
| Left Cheek vs. Nose Bridge | Macro Precision | -0.0343 | mean diff = -0.0344 | 99.67% CI: [-0.0582, -0.0096] | Reject | Left Cheek |
| Left Cheek vs. Nose Bridge | Macro Recall | -0.0375 | mean diff = -0.0375 | 99.67% CI: [-0.0668, -0.0077] | Reject | Left Cheek |
| Left Cheek vs. Nose Bridge | Macro F1-Score | -0.0621 | mean diff = -0.0621 | 99.67% CI: [-0.0871, -0.0357] | Reject | Left Cheek |
| Left Cheek vs. Full Face | Accuracy | 0.0858 | b = 633, c = 1018 | p = 1.808e-21 | Reject | Full Face |
| Left Cheek vs. Full Face | Macro Precision | 0.0618 | mean diff = +0.0620 | 99.67% CI: [+0.0287, +0.0973] | Reject | Full Face |
| Left Cheek vs. Full Face | Macro Recall | -0.0078 | mean diff = -0.0078 | 99.67% CI: [-0.0374, +0.0221] | Fail to Reject |  |
| Left Cheek vs. Full Face | Macro F1-Score | 0.0273 | mean diff = +0.0273 | 99.67% CI: [-0.0009, +0.0559] | Fail to Reject |  |
| Right Cheek vs. Jawline | Accuracy | 0.0089 | b = 751, c = 791 | p = 0.3085 | Fail to Reject |  |
| Right Cheek vs. Jawline | Macro Precision | 0.0085 | mean diff = +0.0086 | 99.67% CI: [-0.0180, +0.0346] | Fail to Reject |  |
| Right Cheek vs. Jawline | Macro Recall | 0.0178 | mean diff = +0.0179 | 99.67% CI: [-0.0136, +0.0483] | Fail to Reject |  |
| Right Cheek vs. Jawline | Macro F1-Score | 0.0198 | mean diff = +0.0199 | 99.67% CI: [-0.0070, +0.0470] | Fail to Reject |  |
| Right Cheek vs. Nose Bridge | Accuracy | -0.0392 | b = 876, c = 700 | p = 9.149e-06 | Reject | Right Cheek |
| Right Cheek vs. Nose Bridge | Macro Precision | -0.0067 | mean diff = -0.0067 | 99.67% CI: [-0.0325, +0.0197] | Fail to Reject |  |
| Right Cheek vs. Nose Bridge | Macro Recall | 0.0094 | mean diff = +0.0094 | 99.67% CI: [-0.0213, +0.0414] | Fail to Reject |  |
| Right Cheek vs. Nose Bridge | Macro F1-Score | -0.0216 | mean diff = -0.0217 | 99.67% CI: [-0.0483, +0.0053] | Fail to Reject |  |
| Right Cheek vs. Full Face | Accuracy | 0.1074 | b = 593, c = 1075 | p = 1.464e-32 | Reject | Full Face |
| Right Cheek vs. Full Face | Macro Precision | 0.0895 | mean diff = +0.0895 | 99.67% CI: [+0.0578, +0.1247] | Reject | Full Face |
| Right Cheek vs. Full Face | Macro Recall | 0.0391 | mean diff = +0.0390 | 99.67% CI: [+0.0102, +0.0675] | Reject | Full Face |
| Right Cheek vs. Full Face | Macro F1-Score | 0.0678 | mean diff = +0.0676 | 99.67% CI: [+0.0403, +0.0962] | Reject | Full Face |
| Jawline vs. Nose Bridge | Accuracy | -0.0481 | b = 953, c = 737 | p = 1.444e-07 | Reject | Jawline |
| Jawline vs. Nose Bridge | Macro Precision | -0.0151 | mean diff = -0.0150 | 99.67% CI: [-0.0417, +0.0096] | Fail to Reject |  |
| Jawline vs. Nose Bridge | Macro Recall | -0.0084 | mean diff = -0.0082 | 99.67% CI: [-0.0409, +0.0225] | Fail to Reject |  |
| Jawline vs. Nose Bridge | Macro F1-Score | -0.0414 | mean diff = -0.0412 | 99.67% CI: [-0.0693, -0.0151] | Reject | Jawline |
| Jawline vs. Full Face | Accuracy | 0.0985 | b = 676, c = 1118 | p = 1.002e-25 | Reject | Full Face |
| Jawline vs. Full Face | Macro Precision | 0.0810 | mean diff = +0.0810 | 99.67% CI: [+0.0461, +0.1163] | Reject | Full Face |
| Jawline vs. Full Face | Macro Recall | 0.0213 | mean diff = +0.0211 | 99.67% CI: [-0.0109, +0.0547] | Fail to Reject |  |
| Jawline vs. Full Face | Macro F1-Score | 0.0480 | mean diff = +0.0478 | 99.67% CI: [+0.0172, +0.0796] | Reject | Full Face |
| Nose Bridge vs. Full Face | Accuracy | 0.1466 | b = 635, c = 1293 | p = 9.856e-52 | Reject | Full Face |
| Nose Bridge vs. Full Face | Macro Precision | 0.0962 | mean diff = +0.0962 | 99.67% CI: [+0.0637, +0.1314] | Reject | Full Face |
| Nose Bridge vs. Full Face | Macro Recall | 0.0297 | mean diff = +0.0296 | 99.67% CI: [+0.0005, +0.0593] | Reject | Full Face |
| Nose Bridge vs. Full Face | Macro F1-Score | 0.0894 | mean diff = +0.0893 | 99.67% CI: [+0.0602, +0.1198] | Reject | Full Face |

## Table 33 — Final decisions

| hypothesis | basis | bins_rejected | conclusion |
|---|---|---|---|
| H01 — Accuracy | McNemar's mid-p (per bin) vs. Bonferroni alpha = 0.0042 | Low, High | Reject H0 |
| H01 — Macro Precision | 99.58% BCa CI (per bin) excludes / includes zero | Medium | Reject H0 |
| H01 — Macro Recall | 99.58% BCa CI (per bin) excludes / includes zero | none | Fail to Reject H0 |
| H01 — Macro F1-Score | 99.58% BCa CI (per bin) excludes / includes zero | none | Fail to Reject H0 |
| H02 — Accuracy | Cochran's Q vs. alpha = 0.05; post-hoc McNemar's mid-p vs. alpha = 0.0033 | 10 of 15 pairs | Reject H0 |
| H02 — Macro Precision | Pairwise 99.67% BCa CIs exclude / include zero | 8 of 15 pairs | Reject H0 |
| H02 — Macro Recall | Pairwise 99.67% BCa CIs exclude / include zero | 6 of 15 pairs | Reject H0 |
| H02 — Macro F1-Score | Pairwise 99.67% BCa CIs exclude / include zero | 8 of 15 pairs | Reject H0 |
| H03 — Accuracy | McNemar's mid-p vs. Bonferroni alpha = 0.0125 | full test set | Reject H0 |
| H03 — Macro Precision | 98.75% BCa CI excludes / includes zero | full test set | Reject H0 |
| H03 — Macro Recall | 98.75% BCa CI excludes / includes zero | full test set | Fail to Reject H0 |
| H03 — Macro F1-Score | 98.75% BCa CI excludes / includes zero | full test set | Fail to Reject H0 |

Decision rule for H01/H02 (any bin / pair significant -> reject) is this script's reading of the manuscript; confirm with the adviser.
