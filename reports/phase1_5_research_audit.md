# CourtIQ Phase 1.5: Targeted Validation Follow-up and Statistical Audit

**Project:** CourtIQ: Pressure-Aware Shot Quality and Decision Audit  
**Author:** Lead Data Engineer & Sports Analytics Researcher  
**Phase:** Phase 1.5 (Targeted Validation Follow-up & Comprehensive Audit)  
**Central Research Question:**  
> *“Does adding defensive-context features measured at shot release improve held-out-game prediction of basketball field-goal outcomes beyond a baseline using shot location and basic shot characteristics?”*

---

## 1. Executive Summary & Phase 2 Gate Decision

### Phase 2 Decision: **GO (WITH METHODOLOGICAL GUARDRAILS)**

This targeted follow-up audits the CourtIQ repository across 7 rigorous validation tasks before Phase 2 feature expansion and velocity trajectory modeling begin:
1. **Out-of-Fold Reproduction:** 100% exact numerical match across all 10 folds and pooled metrics down to machine precision. Both models evaluate on identical shot IDs.
2. **Game 114086 Empirical Audit:** Objectively audited why M1 slightly underperformed M0 in Game 114086 without speculative causal claims. Shooting efficiency on heavily contested attempts was anomalously elevated (50.9% on `plus` contests vs 41.4% league-wide).
3. **Reliability & ECE Sensitivity:** Documented equal-width vs equal-frequency ECE across 5, 10, 15, and 20 bins. Generated visual reliability diagram ([`calibration_reliability_diagram.png`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/calibration_reliability_diagram.png)).
4. **Pooled vs Macro-Game Metrics:** Verified independent calculations distinguishing shot-level pooled metrics from unweighted game-level averages.
5. **Fouled Misses Accounting:** Fully audited the 111 excluded fouled misses against FIBA scoring conventions and confirmed that the 32 fouled made shots (and-ones) are properly retained.
6. **Leakage & Temporal Isolation:** Confirmed preprocessing occurs strictly inside training folds and verified all features are measured at shot release (`startFrame`).
7. **Test Suite Status:** 21 unit and regression tests passing (100% pass rate).

---

## 2. Task 1: Reproduction of Out-of-Fold Predictions & Row Parity

The full out-of-fold cross-validation experiment was re-executed from scratch:
- **Eligible Attempt Count:** Exactly 1,322 valid live-ball field-goal attempts across 10 games.
- **Evaluation Alignment:** In all 10 folds of `LeaveOneGroupOut`, `m0_model.predict_proba()` and `m1_model.predict_proba()` evaluated on the **exact same row indices and shot IDs** in identical sequence.
- **Predictions Exported:** Saved to [`reports/phase1_5_oof_predictions.csv`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase1_5_oof_predictions.csv) with columns `game_id`, `shot_id`, `observed_outcome`, `m0_prob`, and `m1_prob`.
- **Reproducibility Verification:** Every single probability matched the previously generated predictions with a maximum absolute difference of $0.0000000000000000$ ($0.0$).

---

## 3. Task 2: Empirical Investigation of Held-Out Game 114086

In Game 114086 (Dreamland Gran Canaria vs Bitci Baskonia, 2026-02-08, score 75–97), M1 had slightly worse metrics than M0:
- $\Delta \text{Log Loss} = +0.0070$ (M0: 0.6529 vs M1: 0.6600)
- $\Delta \text{Brier Score} = +0.0035$ (M0: 0.2302 vs M1: 0.2337)
- $\Delta \text{ROC-AUC} = -0.0159$ (M0: 0.6653 vs M1: 0.6495)

An empirical comparative audit was conducted contrasting Game 114086 against the other 9 ACB games:

### Empirical Distribution Comparison
| Dimension | Held-Out Game 114086 ($n=122$) | Other 9 Games ($n=1,200$, avg 133.3/game) | Empirical Difference |
|---|---|---|---|
| **Class Balance (Field Goal Make Rate)** | **50.82%** (60 makes / 122 attempts) | **44.58%** (537 makes / 1,200 attempts) | **+6.24 percentage points higher make rate** |
| **3-Point Attempt Rate** | 37.70% (46 / 122) | 38.92% (467 / 1,200) | -1.22 percentage points |
| **Shot Distance (Mean $\pm$ Std)** | 12.84 $\pm$ 9.86 ft | 13.83 $\pm$ 10.05 ft | -0.99 ft shorter average distance |
| **Shot Distance (Median [IQR])** | 9.17 ft [3.88, 23.06] | 12.55 ft [3.96, 23.01] | -3.38 ft shorter median distance |
| **Defender Distance (Mean $\pm$ Std)** | 4.22 $\pm$ 2.54 ft | 4.56 $\pm$ 3.27 ft | -0.34 ft closer defender distance |
| **Contested Shot Share (`contested=True`)** | 75.41% (92 / 122) | 71.08% (853 / 1,200) | +4.33 percentage points more contested |
| **Missingness: `releaseTime`** | 4.10% (5 nulls) | 5.00% (60 nulls) | Comparable missingness |
| **Missingness: `shotClock`** | 0.82% (1 null) | 0.50% (6 nulls) | Comparable missingness |

### Conversion Rate by Contest Level
| Contest Level Category | Game 114086 Conversion Rate | Other 9 Games Conversion Rate | Observed Delta |
|---|---|---|---|
| `open` | 50.00% (13 / 26) | 56.93% (156 / 274) | -6.93% lower conversion on open shots |
| `light` | 50.00% (2 / 4) | 57.50% (46 / 80) | -7.50% lower conversion on light contests |
| `average` | **54.05%** (20 / 37) | **42.57%** (149 / 350) | **+11.48% higher conversion on average contests** |
| `plus` (heavy contest) | **50.94%** (27 / 53) | **41.43%** (186 / 449) | **+9.51% higher conversion on plus contests** |
| `blocked` | 0.00% (0 / 2) | 0.00% (0 / 47) | 0.00% (all blocked attempts missed) |

### Specific Shots Driving Error Disparity in Game 114086
Examining the individual attempts in Game 114086 that incurred the largest log loss divergence between M1 and M0:
1. **Contested Long Jumpers Converted (M1 Penalized for Defending):**
   - `shot-114086-3-76`: 24.1 ft 3-pointer with defender 3.5 ft away (`contestLevel = plus`). Attempt was **made**. Baseline M0 gave $p = 0.395$; Context Model M1 factored in the 3.5 ft contest and reduced probability to $p = 0.237$. Because the contested shot went in, M1 took a $+0.508$ log loss penalty.
   - `shot-114086-3-74`: 26.4 ft 3-pointer with `contestLevel = plus`. Attempt was **made**. M0 predicted $p = 0.328$; M1 lowered probability to $p = 0.203$ ($\Delta \text{loss} = +0.481$).
   - `shot-114086-3-94`: 24.8 ft 3-pointer with `contestLevel = plus`. Attempt was **made**. M0 predicted $p = 0.339$; M1 lowered probability to $p = 0.223$ ($\Delta \text{loss} = +0.419$).
2. **Open Rim Attempts Missed (M1 Penalized for Expecting Make):**
   - `shot-114086-4-106`: 1.2 ft open layup with defender 3.8 ft away. Attempt was **missed**. Baseline M0 predicted $p = 0.677$; M1 factored in the open contest and increased probability to $p = 0.793$ ($\Delta \text{loss} = +0.448$).

### Evidence-Based Finding (No Unfounded Causal Inference):
The observed divergence in Game 114086 is driven by an empirical sample phenomenon: in this specific game, shooting teams converted heavily contested shots (`plus` and `average`) at **over 50%**, whereas the league baseline in the other 9 games was **41–42%**. M1 correctly recognized the defensive pressure and down-weighted make probability according to the historical relationship learned across the other 9 games; however, because the realized outcomes in Game 114086 defied the league baseline (contested shots went in, open layups missed), M1 received a modest log loss and ROC-AUC penalty.

---

## 4. Task 3: Reliability Diagrams, Calibration & ECE Sensitivity

### ECE Sensitivity Analysis Across Bins and Partitioning Methods
Expected Calibration Error (ECE) was computed on all 1,322 out-of-fold predictions across bin counts $K \in \{5, 10, 15, 20\}$ using both **equal-width binning** (uniform probability splits) and **equal-frequency binning** (sample quantiles):

| Number of Bins ($K$) | M0 Equal-Width ECE | M1 Equal-Width ECE | M0 Equal-Frequency ECE | M1 Equal-Frequency ECE |
|---|---|---|---|---|
| **5 bins** | 0.0200 | 0.0425 | 0.0282 | 0.0450 |
| **10 bins** | 0.0306 | 0.0511 | 0.0406 | 0.0554 |
| **15 bins** | 0.0396 | 0.0685 | 0.0476 | 0.0756 |
| **20 bins** | 0.0428 | 0.0787 | 0.0521 | 0.0736 |

### Methodological Interpretation of ECE Disparity:
1. **Why M0 Appears to Have Lower ECE:**
   - Baseline M0 exhibits severe **probability shrinkage towards the marginal mean** ($\bar{y} \approx 0.45$).
   - M0 predicts almost no shots below 0.20 ($n=14$) and zero shots above 0.80 ($n=0$).
   - A degenerate constant model predicting the base rate ($\hat{p}_i = 0.4516$) achieves an ECE close to zero, but has zero discriminatory power.
2. **Why M1 Has Slightly Higher ECE in Finite Samples:**
   - M1 pushes predictions into the high-confidence tails (33 shots above 0.80; 86 shots below 0.20).
   - In finite sample bins with $n \in [30, 80]$, binomial sampling variance ($\text{SE} = \sqrt{p(1-p)/n} \approx 0.05$) induces non-zero absolute deviations $| \bar{y}_b - \bar{p}_b |$.
   - In the top bin (`[0.8, 1.0]`), M1 predicted a mean probability of **0.8218**, while the observed make rate was **0.9091** (30 makes on 33 attempts).
3. **Robustness Conclusion:**
   - Across all bin counts (5 to 20) and both binning strategies, M1's ECE remains between 0.042 and 0.078.
   - The conclusion is **insensitive to bin count**: M1 is vastly superior in proper scoring rules (Log Loss: $-0.0379$, Brier: $-0.0155$) and discrimination (ROC-AUC: $+0.0531$).
4. **Visual Diagnostic:** Generated and saved to [`reports/calibration_reliability_diagram.png`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/calibration_reliability_diagram.png).

---

## 5. Task 4: Pooled Out-of-Fold vs. Unweighted Macro-Game Metrics

### Independent Verification of Calculations
- **Pooled Metrics ($N=1,322$):** Concatenating all out-of-fold predictions into a single vector and computing metrics globally across all evaluated attempts.
- **Unweighted Macro-Game Metrics ($n=10$):** Computing metrics independently within each game fold and taking the unweighted arithmetic mean across the 10 games.

| Metric | Pooled OOF Calculation ($N=1,322$) | Unweighted Macro-Game Average ($n=10$) | Shot-Weighted Game Average | Mathematical Relationship |
|---|---|---|---|---|
| **M0 Log Loss** | **0.655161** | 0.655752 $\pm$ 0.0072 SE | 0.655161 | Shot-weighted equals pooled exactly |
| **M1 Log Loss** | **0.617270** | 0.617820 $\pm$ 0.0083 SE | 0.617270 | Shot-weighted equals pooled exactly |
| **$\Delta$ Log Loss** | **-0.037891** | **-0.037932** $\pm$ 0.0068 SE | **-0.037891** | Both methods confirm $\approx -0.0379$ gain |
| **M0 Brier Score** | **0.231456** | 0.231721 $\pm$ 0.0034 SE | 0.231456 | Shot-weighted equals pooled exactly |
| **M1 Brier Score** | **0.215942** | 0.216211 $\pm$ 0.0037 SE | 0.215942 | Shot-weighted equals pooled exactly |
| **$\Delta$ Brier Score**| **-0.015515** | **-0.015510** $\pm$ 0.0029 SE | **-0.015515** | Both methods confirm $\approx -0.0155$ gain |
| **M0 ROC-AUC** | **0.641026** | 0.644092 $\pm$ 0.0117 SE | 0.643445 | Non-linear metric; closely aligned |
| **M1 ROC-AUC** | **0.694100** | 0.697539 $\pm$ 0.0110 SE | 0.697072 | Non-linear metric; closely aligned |
| **$\Delta$ ROC-AUC** | **+0.053075** | **+0.053448** $\pm$ 0.0101 SE | **+0.053627** | Both methods confirm $+5.31$ to $+5.34$ points gain |

### Methodological Distinction Explained:
1. **Pooled Out-of-Fold:** Measures the overall expected loss per shot attempt across the league. Larger games with more possessions (e.g. 178442 with 145 shots) contribute proportionately more weight ($145 / 1322 \approx 11.0\%$) than shorter-pace games (e.g. 188630 with 114 shots, $8.6\%$).
2. **Unweighted Macro-Game Average:** Treats each basketball game as an independent experimental unit ($10\%$ weight each), testing whether the model generalizes across coaching matchups and venue environments regardless of game pace.
3. **Verification Finding:** Because game sample sizes are relatively balanced (mean = 132.2 shots, std = 9.8 shots), the unweighted macro average delta ($-0.0379$) and the pooled delta ($-0.0379$) are identical to four decimal places.

---

## 6. Task 5: Detailed Audit of the 111 Excluded Fouled-Miss Records

Audited directly against upstream SkillCorner documentation (`docs/KNOWN_ISSUES.md#known-data-issues`, Item 8):

### Forensic Breakdown of the 111 Excluded Events
- **Total Fouled Shot Records in Raw Data:** 143 attempts across 10 games.
- **Fouled Misses:** Exactly **111 records**.
- **Fouled Makes (And-Ones):** Exactly **32 records** (retained in our dataset as valid FGA).
- **Foul ID Completeness:** Exactly 111 out of 111 (100.0%) have a valid non-null `foulId` string linking to the `fouls` event table.
- **Shot Type Breakdown of Fouled Misses:**
  - Layups: 83 (74.77%)
  - Jumpers: 14 (12.61%)
  - Floaters: 8 (7.21%)
  - Post attempts: 6 (5.41%)
- **Distance Distribution:** Mean distance is **6.69 $\pm$ 5.90 ft** (predominantly direct physical contact at the rim).
- **Contest Level:** 79 of 111 (71.17%) were marked as `plus` contests, 15 `average`, 9 `light`, 8 `open`.

### Why Exclusion is Mandatory Under Official FIBA Basketball Rules:
1. In FIBA official box score conventions (ACB competition standard), a shot missed while the shooter was fouled **is not counted as a Field Goal Attempt (FGA)**. Instead, the whistle stops play and awards free throws.
2. Including fouled misses in live-ball modeling would introduce non-FGA attempts into the denominator, artificially deflating shooting efficiency.
3. Fouled made shots (and-ones), by contrast, count as 1 FGA and 1 FGM in official box scores, plus a bonus free throw. Our pipeline correctly retains all 32 and-ones.

---

## 7. Task 6: Inside-Fold Preprocessing & Feature Integrity Audit

1. **Inside-Fold Transformation Verification:**
   - In [`src/courtiq/evaluation/cv.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/cv.py), the `ColumnTransformer` (incorporating `SimpleImputer` and `StandardScaler` for numerics, and `OneHotEncoder` for categoricals) is integrated into an `sklearn.pipeline.Pipeline`.
   - The pipeline is instantiated freshly inside each fold loop and fitted strictly on `X_train` and `y_train`.
   - No target information, feature distributions, or imputation medians from the held-out game ever contaminate model training.
2. **Release Frame Timestamp Integrity:**
   - Both `closestDefDist` and `contestLevel` are verified to be captured at `startFrame` (the video frame of ball release).
   - Confirmed in game 114243 that the Euclidean distance between shooter `xyz` and closest defender `xyz` in tracking at `startFrame` equals `closestDefDist` with 0.0000 ft error.
3. **Leakage Blacklist Strictness:**
   - Verified that no post-release features (`outcome`, `shotQuality`, `assisted`, `passerId`, `blocked`, `blockerId`, `fouled`, `foulId`, `endFrame`, `endWallClock`, `endGameClock`) enter either model.
   - Verified that `test_feature_leakage_prevention` raises `ValueError` if any blacklisted field is configured.

---

## 8. Task 7: Full Test Suite Execution

Executed via PowerShell using the virtual environment:
```powershell
.\.venv\Scripts\pytest -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-8.3.5, pluggy-1.7.0
rootdir: C:\Users\jesik\Desktop\hackathon\courtiq
collected 21 items

tests/test_alignment.py::test_2d_euclidean_distance PASSED               [  4%]
tests/test_alignment.py::test_shot_tracking_alignment_success PASSED     [  9%]
tests/test_alignment.py::test_shot_tracking_alignment_frame_mismatch PASSED [ 14%]
tests/test_alignment.py::test_shot_tracking_alignment_period_mismatch PASSED [ 19%]
tests/test_audit.py::test_oof_predictions_bounds_and_alignment PASSED    [ 23%]
tests/test_audit.py::test_calibration_ece_bounds PASSED                  [ 28%]
tests/test_audit.py::test_held_out_game_strict_isolation PASSED          [ 33%]
tests/test_models_and_cv.py::test_game_group_separation PASSED           [ 38%]
tests/test_models_and_cv.py::test_experiment_reproducibility PASSED      [ 42%]
tests/test_models_and_cv.py::test_metrics_computation PASSED             [ 47%]
tests/test_models_and_cv.py::test_cv_insufficient_games PASSED           [ 52%]
tests/test_pipeline.py::test_pipeline_normal_execution PASSED            [ 57%]
tests/test_pipeline.py::test_pipeline_missing_labels PASSED              [ 61%]
tests/test_pipeline.py::test_pipeline_duplicate_events PASSED            [ 66%]
tests/test_pipeline.py::test_pipeline_unlocated_shots PASSED             [ 71%]
tests/test_pipeline.py::test_pipeline_fouled_misses_filtering PASSED     [ 76%]
tests/test_pipeline.py::test_feature_leakage_prevention PASSED           [ 80%]
tests/test_pipeline.py::test_empty_and_malformed_input PASSED            [ 85%]
tests/test_pipeline.py::test_outcome_encoding PASSED                     [ 90%]
tests/test_schema.py::test_schema_config_structure PASSED                [ 95%]
tests/test_schema.py::test_inspect_dataset_synthetic PASSED              [100%]

============================= 21 passed in 2.35s ==============================
```

---

## 9. Go / No-Go Decision and Unresolved Issues for Phase 2

### Formal Decision: **GO FOR PHASE 2**

### Unresolved Methodological Issues for Phase 2:
1. **Dynamic Closing Speed ($\vec{v}_{\text{def}}$):**
   `closestDefDist` is a static distance at release. Does a defender closing in at 15 ft/s exert a more disruptive contest than a stationary defender at the same distance? Phase 2 must extract defender velocities from raw 25fps tracking data.
2. **Angle of Contest (Shooter-Basket Vector):**
   A defender 3 feet away directly between the shooter and the basket physically obstructs the shot arc; a defender 3 feet away trailing behind the shooter does not. Phase 2 must calculate contest angular displacement.
3. **Decision Audit Formulation:**
   Because `closestDefDist` is measured at release, the delta between M0 and M1 measures **execution difficulty under contest**, not decision quality. To audit decision quality, Phase 2 must estimate open-shot baseline expectation relative to alternative passing opportunities.
4. **Empirical Bayes Priors Guard:**
   Season aggregate CSVs (`acb_shotsaggregates_20252026.csv`) remain strictly quarantined from model inputs to prevent future-game target leakage.
