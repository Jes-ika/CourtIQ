# Baseline Experiment Results: M0 vs M1

## Research Question
Does adding defensive-context features improve held-out-game prediction of basketball field-goal outcomes beyond a baseline using shot location and basic shot characteristics?

## Experimental Setup
- **Target Variable**: Field-goal success (`outcome`: True = Made / 1, False = Missed / 0)
- **Validation Strategy**: Leave-One-Group-Out (LOGO) cross-validation across independent games (10 folds).
- **Leakage Prevention**: All post-shot fields (`shotQuality`, `fouled`, `blocked`, `assisted`, etc.) strictly blacklisted.
- **Preprocessing**: Median imputation and standard scaling for continuous features; most-frequent imputation and one-hot encoding for categoricals.

### Observation Filtering and Exclusions
| Step | Count | Description |
|---|---|---|
| Total Raw Shots | 1459 | Raw shot events extracted across 10 games |
| Missing Outcome Labels | 0 | Shots missing target label |
| Duplicate Shot IDs | 0 | Duplicate shot event IDs |
| Unlocated / Missing Coordinates | 26 | Shots with missing spatial tracking coordinates |
| Fouled Misses (FIBA Non-FGA) | 111 | Fouled misses (resulted in free throws, not box-score FGA) |
| **Total Valid Shots Analyzed** | **1322** | Analyzed across both models |

---

## Model Specifications
- **M0 Baseline Model**:
  - Distance to hoop (`distance`)
  - Shot clock remaining (`shotClock`)
  - Touch-to-release time (`releaseTime`)
  - Dribbles before shot (`dribblesBefore`)
  - Shot type (`shotType`)
  - Three-point attempt flag (`three`)
  - Catch-and-shoot flag (`catchAndShoot`)
  - Created from paint flag (`createdFromPaint`)
  - Game period (`period`)
- **M1 Context Model**:
  - All M0 features plus:
  - Distance to closest defender at release (`closestDefDist`)
  - Contest level categorization (`contestLevel`: open, light, average, plus, blocked)
  - Binary contested flag (`contested`)

---

## Held-Out Evaluation Metrics
| Metric | M0 Baseline | M1 Context Model | Delta (M1 - M0) | 95% Confidence Interval (Student's t, df=9) | p-value (Paired t-test) |
|---|---|---|---|---|---|
| **Log Loss** | 0.6552 | 0.6173 | **-0.0379** | [-0.0533, -0.0225] | 0.0003 |
| **Brier Score** | 0.2315 | 0.2159 | **-0.0155** | [-0.0220, -0.0090] | 0.0004 |
| **ROC-AUC** | 0.6410 | 0.6941 | **+0.0531** | [0.0306, 0.0763] | 0.0005 |
| **ECE (5 bins)** | 0.0200 | 0.0425 | +0.0225 | - | - |

*(Note: For Log Loss and Brier Score, negative deltas indicate model improvement; for ROC-AUC, positive delta indicates improvement).*

---

## Fold-by-Fold Held-Out Game Breakdown
| Fold | Held-Out Game ID | Shots | M0 Log Loss | M1 Log Loss | Delta Log Loss | M0 Brier | M1 Brier | Delta Brier | M0 AUC | M1 AUC | Delta AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 114086 | 122 | 0.6529 | 0.6600 | +0.0070 | 0.2302 | 0.2337 | +0.0035 | 0.6653 | 0.6495 | -0.0159 |
| 2 | 114099 | 140 | 0.6280 | 0.6034 | -0.0247 | 0.2190 | 0.2085 | -0.0105 | 0.6790 | 0.7152 | +0.0362 |
| 3 | 114169 | 138 | 0.6250 | 0.5847 | -0.0403 | 0.2178 | 0.2011 | -0.0167 | 0.6888 | 0.7367 | +0.0479 |
| 4 | 114234 | 125 | 0.6895 | 0.6190 | -0.0705 | 0.2479 | 0.2178 | -0.0301 | 0.5788 | 0.6844 | +0.1055 |
| 5 | 114243 | 144 | 0.6486 | 0.5980 | -0.0505 | 0.2288 | 0.2078 | -0.0210 | 0.6392 | 0.7085 | +0.0693 |
| 6 | 178442 | 145 | 0.6562 | 0.6235 | -0.0327 | 0.2316 | 0.2178 | -0.0137 | 0.6535 | 0.7087 | +0.0552 |
| 7 | 179612 | 134 | 0.6414 | 0.6098 | -0.0316 | 0.2251 | 0.2111 | -0.0140 | 0.6562 | 0.7009 | +0.0447 |
| 8 | 184439 | 130 | 0.6695 | 0.6118 | -0.0577 | 0.2375 | 0.2153 | -0.0223 | 0.6288 | 0.7076 | +0.0788 |
| 9 | 188630 | 114 | 0.6533 | 0.6023 | -0.0510 | 0.2300 | 0.2097 | -0.0203 | 0.6657 | 0.7358 | +0.0701 |
| 10 | 191313 | 130 | 0.6931 | 0.6657 | -0.0274 | 0.2495 | 0.2395 | -0.0100 | 0.5855 | 0.6281 | +0.0426 |

---

## Methodological Findings, Assumptions, and Limitations

### Observed Facts
1. In 9 out of 10 held-out game folds, the Context Model (M1) achieved a lower Log Loss and higher ROC-AUC than the Baseline Model (M0).
2. Held-out ROC-AUC improved by an average of +5.31 percentage points (95% CI: [+0.0306, +0.0763]), demonstrating robust discrimination improvement.
3. Log loss decreased by -0.0379 (95% CI: [-0.0533, -0.0225], paired t-test p = 0.0003).
4. Exactly 111 fouled misses were excluded to adhere to official FIBA box-score FGA conventions.

### Assumptions
1. We assume that holding out entire games (Leave-One-Group-Out) isolates game-specific tactical idiosyncrasies, venue lighting, and officiating biases.
2. We assume linear combinations of distance, shot clock, and contest indicators under logistic regression capture primary additive pressure effects before non-linear modeling.

### Unresolved Questions
1. In Game 114086 (Fold 1), M1 showed slightly lower AUC (-0.0159). Was this driven by anomalous transition shot selection or defensive coverage tracking noise in that specific matchup?
2. Does the marginal predictive gain of contest level diminish once defender velocity and closing angle are modeled?
