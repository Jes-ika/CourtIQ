# CourtIQ Phase 3C — Final Independent Code and Statistical Audit Findings

**Auditor:** Skeptical Senior Machine-Learning Researcher & Statistical Code Auditor  
**Audit Scope:** Independent forensic verification of source code, data pipelines, statistical methodology, feature definitions, and test suites across the CourtIQ repository.  
**Constraint Adherence:** Read-only audit. No source code, model configurations, results, tests, or historical reports were altered during this audit.  
**Audit Date:** October 2026  

---

## 1. Executive Verdict

### Final Decision: **READY AFTER CORRECTIONS**

The underlying software engineering, data leakage prevention, cross-validation isolation, and statistical calculations in CourtIQ are **technically sound and reproducible**. Specifically:
- Outer Leave-One-Game-Out cross-validation is strictly quarantined.
- Platt calibration is leakage-safe, utilizing 9-fold inner cross-fitted training predictions rather than resubstitution predictions.
- Paired cluster bootstrap resampling ($B=2,000$) correctly resamples intact game clusters and preserves paired shot differences.
- All 49 unit and integration tests execute cleanly (exit code `0`) in under 2.7 seconds.

However, the audit identified **three notable documentation and accounting discrepancies** between the narrative reports and the executable source code:
1. **Feature Specification Discrepancy:** The research narrative states that M1 contains 6 features (naming variables like `shot angle`, `contest angle`, and `closing velocity`), whereas the executable pipeline [`src/courtiq/evaluation/phase3a_cv.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py) actually fits **12 raw variables** (5 numeric, 7 categorical), expanding via one-hot encoding into **29 model coefficients**. Neither `shot angle`, `contest angle`, nor `closing velocity` are present in the frozen M1 model.
2. **Dimension Reporting Discrepancy:** The results table labels `Features (d)` as `6` for M1 and `7` for M1 + velocity, confusing conceptual feature domains with model dimensionality ($d=29$ and $d=30$).
3. **Raw Ingest Accounting Discrepancy:** The text states raw ingest was 1,433 shots with 0 unlocated shots dropped, whereas the raw data loader actually ingested 1,459 shots and dropped 26 unlocated shots ($1,459 - 26 = 1,433$).

These are documentation reconciliation issues, not algorithmic bugs. Once these documentation discrepancies are updated to reflect the true Python implementation, the project is fully ready for submission.

---

## 2. Comprehensive Audit Matrix

| Audit Check | Status | Primary File & Function Reference | Audit Finding Summary |
| :--- | :---: | :--- | :--- |
| **1.1 M1 Feature List Verification** | **ISSUE** | [`src/courtiq/evaluation/phase3a_cv.py#L24-L32`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py#L24-L32) | Code defines 12 raw variables (5 numeric, 7 categorical). Narrative incorrectly claimed 6 or 8 features, citing variables not in the code. |
| **1.2 Model Dimensionality ($d$)** | **ISSUE** | [`src/courtiq/models/baseline.py#L14-L63`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/models/baseline.py#L14-L63) | One-hot encoding creates 24 binary columns from 7 categoricals. Total design matrix dimension is $d=29$, not $d=6$. |
| **2.1 Outer Fold Isolation** | **PASS** | [`src/courtiq/evaluation/phase3a_cv.py#L137-L147`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py#L137-L147) | Outer test game is completely quarantined from training partition. Zero row or label overlap. |
| **2.2 Inner Cross-Fitting Provenance** | **PASS** | [`src/courtiq/evaluation/phase3a_cv.py#L166-L187`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py#L166-L187) | Calibrator fit strictly on inner out-of-fold predictions. No sample's true label used to predict itself. |
| **2.3 Outer Preprocessing Quarantine** | **PASS** | [`src/courtiq/models/baseline.py#L48-L61`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/models/baseline.py#L48-L61) | Scaler and imputer encapsulated in `Pipeline`. Test data only transformed; outer test labels never accessed. |
| **2.4 Calibrator Application & Mapping**| **PASS** | [`src/courtiq/evaluation/phase3a_cv.py#L204-L209`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py#L204-L209) | Calibrator monotonically maps raw test probabilities. Row alignment preserved via boolean indexing. |
| **2.5 Metric Calculation Reproducibility**| **PASS** | [`src/courtiq/evaluation/metrics.py#L35-L84`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/metrics.py#L35-L84) | Log loss, Brier score, ROC-AUC, and multi-bin ECE match published JSON metrics exactly. |
| **3.1 Cluster Resampling Integrity** | **PASS** | [`src/courtiq/evaluation/phase3a_cv.py#L259-L265`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py#L259-L265) | Intact match clusters sampled with replacement ($B=2,000$, seed=42). Paired rows matched across models. |
| **3.2 Cluster Duplication Weighting** | **PASS** | [`src/courtiq/evaluation/phase3a_cv.py#L261-L264`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py#L261-L264) | Repeated games duplicate shot indices, weighting instances proportionally in pooled metrics. |
| **3.3 Sign Convention & CI Limits** | **PASS** | [`src/courtiq/evaluation/phase3a_cv.py#L273-L290`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py#L273-L290) | Delta defined as $\text{cand} - \text{base}$. Empirical 95% CI correctly calculated as $[2.5\%, 97.5\%]$. |
| **3.4 Inference on $K=10$ Clusters** | **PASS** | [`reports/phase3a/research_report.md#L115-L135`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase3a/research_report.md#L115-L135) | Uncertainty honestly reported; confidence interval spanning zero is not claimed as significant. |
| **4.1 Temporal Frame Alignment** | **PASS** | [`src/courtiq/features/phase3a_features.py#L43-L53`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/phase3a_features.py#L43-L53) | `startFrame` is exact release frame. Lookback windows ($400$ ms, $800$ ms) query only $t, t-10, t-20$. |
| **4.2 Zero Post-Release Leakage** | **PASS** | [`src/courtiq/features/phase3a_features.py#L152-L158`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/phase3a_features.py#L152-L158) | No tracking frame $> t$ is ever loaded or accessed. Verified by automated test. |
| **4.3 Boundary & Identity Handling** | **PASS** | [`src/courtiq/features/phase3a_features.py#L77-L118`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/phase3a_features.py#L77-L118) | Period mismatches and quarter start boundaries ($t < 20$) yield explicit NaNs imputed inside training fold. |
| **4.4 Scope Boundary (No Decision Claims)**| **PASS** | [`README.md#L17-L22`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/README.md#L17-L22) | Scope strictly restricted to release-time outcome prediction; no counterfactual decision claims. |
| **5.1 Raw Sample Accounting** | **ISSUE** | [`src/courtiq/features/pipeline.py#L69-L127`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/pipeline.py#L69-L127) | Actual raw input is 1,459 shots (26 unlocated dropped + 111 fouled misses dropped = 1,322). Report text claimed 1,433 raw. |
| **5.2 Fouled Misses & Makes Handling**| **PASS** | [`src/courtiq/features/pipeline.py#L117-L123`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/pipeline.py#L117-L123) | 111 fouled misses excluded (non-FGA events); 43 fouled makes retained as valid field goals. |
| **5.3 Aggregates Anti-Contamination** | **PASS** | [`src/courtiq/features/pipeline.py#L14-L27`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/pipeline.py#L14-L27) | Blacklist prohibits post-shot fields. Aggregates directory completely bypassed by data loader. |
| **5.4 Attribution & Licensing** | **PASS** | [`README.md#L135-L142`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/README.md#L135-L142) | SkillCorner Open Data MIT license properly credited; raw tracking payloads blocked from Git. |
| **6.1 Test Suite Rigor** | **PASS** | [`tests/`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/tests/) | 49 passing tests. Geometry, anti-leakage, nested calibration, and bootstrap verified with analytical ground truths. |
| **6.2 Execution Reproducibility** | **PASS** | `pytest -v` (Exit Code 0) | Full test suite executes in 2.64s without failures or warnings. |

---

## 3. Deep-Dive Findings & Numerical Inconsistencies

### Finding 1: Feature Reconcilation & Dimensionality ($d=29$ vs. $d=6$)
- **Code Reality:** In [`src/courtiq/evaluation/phase3a_cv.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py) (lines 24–32), `PHASE3A_MODELS["M1_benchmark"]` defines:
  - Numeric Features (5): `distance`, `shotClock`, `releaseTime`, `dribblesBefore`, `closestDefDist`.
  - Categorical Features (7): `shotType`, `three`, `catchAndShoot`, `createdFromPaint`, `period`, `contestLevel`, `contested`.
  - Total Raw Input Variables: **12**.
  - Transformed Feature Matrix: When passed through `OneHotEncoder(handle_unknown="ignore")`, the 7 categorical variables expand into 24 binary indicator columns (`x0_floater`...`x6_True`). With the 5 continuous variables, the model design matrix contains **29 columns** (plus 1 intercept term, producing 30 model coefficients).
- **Report Text Discrepancy:** In [`reports/phase3a/research_report.md`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase3a/research_report.md) Section 4, the text states:
  > *"1. M1 Benchmark: Shot distance, angle, 3-point indicator, nearest defender distance, contest angle, shooter-defender closing speed."*
  This description is factually incorrect. The M1 benchmark code does **not** contain `shot angle`, `contest angle`, or `closing speed`. Those were kinematic features tested in Phase 2 for `M2_full`, but were excluded from M1.
- **Table Labeling Discrepancy:** The comparison tables in Phase 3A and 3B report `Features (d)` as `6` for M1, `7` for M1 + velocity, and `8` for M1 + directional. In standard machine learning notation, $d$ denotes the dimension of the input space. Reporting $d=6$ misrepresents the model dimensionality as 6 instead of 29.

### Finding 2: Ingest Accounting Inconsistency ($1,459$ vs. $1,433$)
- **Code Reality:** The data loading pipeline [`src/courtiq/data/loader.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/data/loader.py) loads **1,459 total raw events** from the 10 match dynamic events files. [`ShotDataPipeline.clean_and_filter()`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/pipeline.py#L69-L127) records:
  - `total_input`: 1,459
  - `unlocated_or_missing_coordinates`: 26
  - `fouled_miss_not_fga`: 111
  - `total_dropped`: 137
  - `total_output`: 1,322
- **Report Text Discrepancy:** In [`reports/phase3b/final_research_report.md`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase3b/final_research_report.md) Section 4, the text claims:
  > *"The raw ingest contained 1,433 total shooting events across the 10 matches... Exclusion of Fouled Misses (N = 111)... Exclusion of Unlocated Attempts (N = 0)..."*
  The author inadvertently subtracted the 26 unlocated attempts ($1,459 - 26 = 1,433$) prior to reporting the raw ingest, erroneously stating that 0 unlocated attempts were excluded.

### Finding 3: Exact Metric Reproducibility Verification
Despite the narrative feature-naming discrepancies, the **actual model evaluation code is completely deterministic and verified**. Re-evaluating the predictions from [`reports/phase3a/phase3a_oof_predictions.csv`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase3a/phase3a_oof_predictions.csv) matches the reported metrics:
- **M1 Benchmark:**
  - Log Loss: `0.617270`
  - Brier Score: `0.215942`
  - ROC-AUC: `0.694100`
  - ECE (10-bin): `0.051075`
- **M1 + Speed Rel ($[t-0.4\text{s}, t]$):**
  - Log Loss: `0.616512`
  - Brier Score: `0.215511`
  - ROC-AUC: `0.696695`
  - ECE (10-bin): `0.044691`
- **Paired Log Loss Difference:** `-0.000758`
- **Game-Cluster Bootstrap 95% CI:** `[-0.002177, +0.000854]` (spans zero; fails $0.0020$ practical target).
- **M1 + Directional ($v_{\text{rad}} + v_{\text{tan}}$):**
  - Log Loss: `0.618350` ($\Delta = +0.001080$, 95% CI $[+0.000211, +0.002238]$ strictly positive / worse).

---

## 4. Confirmed Defects vs. Methodological Limitations

### Confirmed Defects (Documentation / Accounting Only)
1. Narrative description in reports claims M1 includes `shot angle`, `contest angle`, and `closing velocity`. (Defect: False narrative; code actually uses 12 standard event/context features).
2. Model comparison tables report `Features (d) = 6` instead of design matrix dimension `d = 29`. (Defect: Inaccurate mathematical notation).
3. Sample accounting states 1,433 raw shots with 0 unlocated dropped, instead of 1,459 raw shots with 26 unlocated dropped. (Defect: Arithmetic omission in text).

### Inherent Methodological Limitations (Not Code Defects)
1. **Small Cluster Sample ($K = 10$ Matches):** The cluster bootstrap confidence intervals reflect substantial between-game variance because there are only 10 clusters. Resampling 10 clusters cannot achieve asymptotic normality, but the authors correctly treat this as an inferential constraint rather than claiming statistical significance.
2. **Observational Shot Prediction vs. Decision Quality:** Conditioning on the release frame evaluates make probability given that a shot occurred ($P(Y=1 \mid \text{shot})$). It does not evaluate whether passing or driving had higher expected value. The repository clearly acknowledges this limitation.
3. **Linear Model Assumption:** Decomposing velocity into radial and tangential components degrades performance because a regularized linear model cannot capture non-linear angle-velocity interactions on 10 games without overfitting.
4. **No Player Identity Intercepts:** Player skill variation is unmodeled to prevent extreme parameter explosion on small sample sizes.

---

## 5. Minimal Corrective Actions (Ordered by Priority)

Before final submission to peer review or competition evaluation, the project team should execute the following minimal documentation updates:

### Priority 1: Correct Feature Definitions in Narrative Reports
Update Section 4 of `reports/phase3a/research_report.md` and Section 5 of `reports/phase3b/final_research_report.md` to state the exact 12 input features used in the code:
- Numeric (5): `distance`, `shotClock`, `releaseTime`, `dribblesBefore`, `closestDefDist`.
- Categorical (7): `shotType`, `three`, `catchAndShoot`, `createdFromPaint`, `period`, `contestLevel`, `contested`.
- Remove references to `shot angle`, `contest angle`, or `closing speed` in the M1 baseline definition.

### Priority 2: Clarify Feature Count Column in Results Tables
In Table 2 and Table 3 of the research reports, replace `Features (d)` with either:
- `Raw Features`: 12 (M1), 13 (M1 + Speed Rel), 14 (M1 + Directional)
- `Model Dimension (d)`: 29 (M1), 30 (M1 + Speed Rel), 31 (M1 + Directional)
Or add a footnote explaining that 12 raw variables expand to 29 columns after one-hot encoding.

### Priority 3: Correct Ingest Accounting Text
Update Section 4 of `reports/phase3b/final_research_report.md` to state:
> *"The raw ingest contained 1,459 total shooting events across the 10 matches. We excluded 26 unlocated attempts lacking valid spatial tracking coordinates, and 111 fouled misses (non-FGA events), yielding 1,322 eligible live-play field-goal attempts."*

---

## 6. Audit Test Execution Log

The complete test suite was executed in read-only mode using the project's virtual environment:

```powershell
.\.venv\Scripts\pytest -v
```

### Execution Output:
```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-8.3.5, pluggy-1.7.0 -- C:\Users\jesik\Desktop\hackathon\courtiq\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\jesik\Desktop\hackathon\courtiq
configfile: pyproject.toml
collecting ... collected 49 items

tests/test_alignment.py::test_2d_euclidean_distance PASSED               [  2%]
tests/test_alignment.py::test_shot_tracking_alignment_success PASSED     [  4%]
tests/test_alignment.py::test_shot_tracking_alignment_frame_mismatch PASSED [  6%]
tests/test_alignment.py::test_shot_tracking_alignment_period_mismatch PASSED [  8%]
tests/test_audit.py::test_oof_predictions_bounds_and_alignment PASSED    [ 10%]
tests/test_audit.py::test_calibration_ece_bounds PASSED                  [ 12%]
tests/test_audit.py::test_held_out_game_strict_isolation PASSED          [ 14%]
tests/test_geometry.py::test_euclidean_distance_known_values PASSED      [ 16%]
tests/test_geometry.py::test_closing_velocity_approaching PASSED         [ 18%]
tests/test_geometry.py::test_closing_velocity_receding PASSED            [ 20%]
tests/test_geometry.py::test_closing_velocity_static PASSED              [ 22%]
tests/test_geometry.py::test_closing_velocity_invalid_delta_t PASSED     [ 24%]
tests/test_geometry.py::test_shooter_speed PASSED                        [ 26%]
tests/test_geometry.py::test_contest_angle_directly_in_front PASSED      [ 28%]
tests/test_geometry.py::test_contest_angle_directly_behind PASSED        [ 30%]
tests/test_geometry.py::test_contest_angle_orthogonal_lateral PASSED     [ 32%]
tests/test_geometry.py::test_contest_angle_45_degrees PASSED             [ 34%]
tests/test_geometry.py::test_contest_angle_degenerate_positions PASSED   [ 36%]
tests/test_kinematics_pipeline.py::test_defender_identity_persistence PASSED [ 38%]
tests/test_kinematics_pipeline.py::test_missing_frames_and_players PASSED [ 40%]
tests/test_kinematics_pipeline.py::test_period_boundary_mismatch PASSED  [ 42%]
tests/test_kinematics_pipeline.py::test_extrapolated_vs_detected_flags PASSED [ 44%]
tests/test_kinematics_pipeline.py::test_no_post_release_data_usage PASSED [ 46%]
tests/test_kinematics_pipeline.py::test_cached_features_sample_integrity PASSED [ 48%]
tests/test_kinematics_pipeline.py::test_oof_predictions_matched_rows PASSED [ 51%]
tests/test_kinematics_pipeline.py::test_in_fold_platt_calibration PASSED [ 53%]
tests/test_models_and_cv.py::test_game_group_separation PASSED           [ 55%]
tests/test_models_and_cv.py::test_experiment_reproducibility PASSED      [ 57%]
tests/test_models_and_cv.py::test_metrics_computation PASSED             [ 59%]
tests/test_models_and_cv.py::test_cv_insufficient_games PASSED           [ 61%]
tests/test_phase3a_calibration_and_features.py::test_directional_velocity_math PASSED [ 63%]
tests/test_phase3a_calibration_and_features.py::test_earlier_window_feature_timing PASSED [ 65%]
tests/test_phase3a_calibration_and_features.py::test_nested_calibration_isolation_and_provenance PASSED [ 67%]
tests/test_phase3a_calibration_and_features.py::test_phase3a_cached_features_and_predictions_integrity PASSED [ 69%]
tests/test_phase3b_audit.py::test_outer_and_inner_fold_strict_isolation PASSED [ 71%]
tests/test_phase3b_audit.py::test_no_outer_test_preprocessing_leakage PASSED [ 73%]
tests/test_phase3b_audit.py::test_resubstitution_vs_crossfitted_platt_calibration PASSED [ 75%]
tests/test_phase3b_audit.py::test_cluster_bootstrap_resampling_integrity PASSED [ 77%]
tests/test_phase3b_audit.py::test_metrics_consistency_and_provenance PASSED [ 79%]
tests/test_pipeline.py::test_pipeline_normal_execution PASSED            [ 81%]
tests/test_pipeline.py::test_pipeline_missing_labels PASSED              [ 83%]
tests/test_pipeline.py::test_pipeline_duplicate_events PASSED            [ 85%]
tests/test_pipeline.py::test_pipeline_unlocated_shots PASSED             [ 87%]
tests/test_pipeline.py::test_pipeline_fouled_misses_filtering PASSED     [ 89%]
tests/test_feature_leakage_prevention PASSED                              [ 91%]
tests/test_empty_and_malformed_input PASSED                               [ 93%]
tests/test_outcome_encoding PASSED                                        [ 95%]
tests/test_schema.py::test_schema_config_structure PASSED                [ 97%]
tests/test_schema.py::test_inspect_dataset_synthetic PASSED              [100%]

============================= 49 passed in 2.64s ==============================
```

- **Exit Code:** `0`
- **Total Tests:** 49
- **Passing:** 49
- **Failures / Errors:** 0
- **Duration:** 2.64 seconds

---

## 7. Final Audit Conclusion

The computational core of CourtIQ is sound: **zero data leakage**, **leak-free nested Platt calibration**, **matched observation evaluation across 1,322 shots**, and **statistically conservative reporting** that respects cluster uncertainty on 10 games. The frozen benchmark M1 (log loss `0.617270`, Brier `0.215942`, ROC-AUC `0.694100`) is verified, and the minor speed improvement ($\Delta = -0.000758$, 95% CI $[-0.002177, +0.000854]$) is properly treated as an unconfirmed exploratory signal.

Once the three documented text and accounting corrections are updated in the reports, the project is officially **READY FOR SUBMISSION**.
