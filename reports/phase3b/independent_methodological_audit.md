# CourtIQ Phase 3B — Independent Methodological & Statistical Audit Report

**Reviewer:** Independent External Statistical Reviewer & Sports Analytics Auditor  
**Repository:** CourtIQ — Basketball Shot Quality and Release-Time Kinematics  
**Audit Target:** Spanish Liga ACB 2025–26 SkillCorner Tracking Benchmark (10 Matches, 1,322 Shots)  
**Evaluation Date:** October 2026  
**Audit Scope:** Independent verification of statistical validity, anti-leakage architecture, calibration provenance, uncertainty quantification, and metric reproducibility.

---

## 1. Executive Summary & Audit Mandate

This independent methodological audit examines the implementation, validation design, and empirical claims of the CourtIQ research pipeline. The primary research question under review is:

> *"Does adding shooter movement information improve held-out-game prediction of basketball field-goal outcomes beyond a baseline using shot characteristics and defensive context measured at shot release?"*

### Overall Audit Findings:
1. **Zero Blocking Defects:** The Phase 3A/3B codebase exhibits rigorous software engineering, strict data hygiene, and zero data leakage.
2. **Leakage-Safe Nested Calibration:** The flawed resubstitution Platt scaling from Phase 2 has been completely decommissioned and replaced by a mathematically sound, 10-outer $\times$ 9-inner nested Leave-One-Game-Out cross-fitting architecture.
3. **Statistical Integrity Upheld:** The project team honestly reports that the release-lookback shooter speed improvement ($\Delta \text{LogLoss} = -0.000758$) fails to achieve statistical significance on 10 game clusters (95% bootstrap CI $[-0.002177, +0.000854]$ spans zero) and does not clear the practical target of $0.0020$.
4. **Reproducibility Verified:** Re-execution of the test suite (49 automated unit/integration tests) and evaluation scripts reproduces all metrics to within machine precision.

---

## 2. In-Depth Methodological Verification

### Item 1: Outer Leave-One-Game-Out Fold Isolation
- **Mechanism:** In [`src/courtiq/evaluation/phase3a_cv.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py) (lines 119–147), partition boundaries are generated via `sklearn.model_selection.LeaveOneGroupOut` over the `gameId` cluster variable.
- **Verification:** Inspection confirms that in each outer fold $k \in \{1, \dots, 10\}$, exactly one game cluster is quarantined as `df_test_outer` ($N \in [114, 145]$ shots), while the remaining 9 games form `df_train_outer`.
- **Automated Verification:** Verified by [`tests/test_phase3b_audit.py::test_outer_and_inner_fold_strict_isolation`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/tests/test_phase3b_audit.py). Outer test game IDs share zero intersection with outer training game IDs.

### Item 2: Inner Cross-Fitting and Platt Calibration Provenance
- **Mechanism:** Within each outer fold, an inner `LeaveOneGroupOut` cross-validation partitions `df_train_outer` across its 9 constitutive games (lines 166–183).
- **Cross-Fitting Proof:** For each inner fold $j \in \{1, \dots, 9\}$, an inner base model is trained on 8 games and predicts strictly on the held-out 9th game. The array `inner_oof_probs` contains solely cross-fitted predictions where no shot was predicted by a model trained on its match.
- **Calibration Fitting:** In line 187, `fit_platt_calibrator(inner_oof_probs, y_train_outer)` fits a univariate logistic calibrator ($\text{logit}(P_{cal}) = a \cdot \text{logit}(P_{raw}) + b$) strictly on these cross-fitted predictions.
- **Comparison to Historical Flaw:** In Phase 2, resubstitution calibration fit Platt scaling on predictions from the *training* model, yielding near-identity parameters ($a \approx 1.0$) and a $+0.000857$ test log-loss penalty. Under nested cross-fitting, calibrators learn the true out-of-sample shrinkage parameter ($a = 0.9110 \pm 0.0067, b = -0.0108 \pm 0.0013$), eliminating the penalty and reducing 10-bin ECE by 12.7% ($0.0511 \to 0.0446$).
- **Automated Verification:** Verified by `tests/test_phase3b_audit.py::test_resubstitution_vs_crossfitted_platt_calibration`.

### Item 3: Preprocessing and Test-Label Leakage Prevention
- **Mechanism:** In [`src/courtiq/models/baseline.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/models/baseline.py), feature imputation (`SimpleImputer(strategy="median")` for numerics, `SimpleImputer(strategy="most_frequent")` for categoricals) and scaling (`StandardScaler`) are encapsulated within a scikit-learn `Pipeline`.
- **Verification:** Preprocessing transformers are fit solely on `df_train_outer` during `outer_pipe.fit()`. When `df_test_outer` is passed to `outer_pipe.predict_proba()`, it is transformed using frozen training statistics.
- **Outer Test Label Quarantine:** `y_test_outer` is never accessed by `build_baseline_pipeline`, `fit_platt_calibrator`, or feature extractors. Test labels are utilized solely for computing held-out metrics after predictions are finalized.
- **Automated Verification:** Verified by `tests/test_phase3b_audit.py::test_no_outer_test_preprocessing_leakage`. Injecting extreme outliers into test sets leaves training scaler means and imputer medians unaltered.

### Item 4: Calibrated Probabilities Correspondence
- **Verification:** In `phase3a_cv.py` (lines 204–209), `p_test_raw` is generated by `outer_pipe.predict_proba(df_test_outer[all_cols])[:, 1]`, and `p_test_cal` is generated by `apply_platt_calibrator(calibrator, p_test_raw)`. Both are stored into global out-of-fold arrays using the exact boolean index mask `test_idx`.
- **Proof:** In [`reports/phase3a/phase3a_oof_predictions.csv`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase3a/phase3a_oof_predictions.csv), every row contains matched `shot_id`, `game_id`, `observed_outcome`, and paired raw/calibrated model probabilities. Probabilities strictly reside in $[0, 1]$.

### Item 5: Paired Game-Cluster Bootstrap Resampling
- **Mechanism:** In lines 247–312 of `phase3a_cv.py`, paired resampling is executed with $B = 2,000$ iterations (seed = 42).
- **Cluster Resampling Integrity:** Resampling samples intact match IDs with replacement (`sampled_games = rng.choice(unique_games, size=n_games, replace=True)`). All shots belonging to sampled games are aggregated together, preserving intra-match correlation structures.
- **Paired Differences:** On each bootstrap replicate, metrics are computed simultaneously on $y[\text{idx}]$, $P_{M1}[\text{idx}]$, and $P_{\text{cand}}[\text{idx}]$. The difference $\Delta = \text{Metric}_{\text{cand}} - \text{Metric}_{M1}$ is recorded, capturing the paired covariance.
- **Confidence Intervals:** Calculated using empirical 2.5th and 97.5th percentiles of bootstrap difference distributions.
- **Automated Verification:** Verified by `tests/test_phase3b_audit.py::test_cluster_bootstrap_resampling_integrity`.

### Item 6: Metric Definitions & Reproducibility
- **Metrics Evaluated:**
  - Log Loss: Cross-entropy using `sklearn.metrics.log_loss(y, p, labels=[0, 1])`.
  - Brier Score: Mean squared error using `sklearn.metrics.brier_score_loss(y, p)`.
  - ROC-AUC: Discrimination using `sklearn.metrics.roc_auc_score(y, p)`.
  - Expected Calibration Error (ECE): Frequency-weighted absolute deviation $\sum_{b=1}^M \frac{|B_b|}{N} |\bar{y}_b - \bar{p}_b|$ across equal-width bins ($M \in \{5, 10, 15\}$).
- **Reproducibility:** Recomputing metrics from saved out-of-fold CSV predictions reproduces the stored JSON metrics across all models:
  - M1 Benchmark: Log Loss = 0.617270, Brier = 0.215942, ROC-AUC = 0.694100.
  - M1 + Speed Rel: Log Loss = 0.616512, Brier = 0.215511, ROC-AUC = 0.696695.
  - Paired Log Loss Delta: $-0.000758$.
- **Automated Verification:** Verified by `tests/test_phase3b_audit.py::test_metrics_consistency_and_provenance`.

### Item 7: Feature Timing, Sample Identity, and Detection Subgroups
- **Prediction Timestamp:** Evaluated strictly at shot release (`startFrame`).
- **Lookback Windows:**
  - Release-lookback speed (`shooter_speed_rel`): Frames $[t-10, t]$ ($400$ ms).
  - Earlier-window speed (`shooter_speed_early`): Frames $[t-20, t-10]$ ($800$ to $400$ ms).
- **Zero Post-Release Leakage:** Frame queries never access frame indices $> t$. (Verified by `test_no_post_release_data_usage`).
- **Tracking Quality Subgroups:** Subgroups are explicitly stratified by `both_detected_pre10`:
  - Directly Detected ($N = 1,207$): Shooter tracked without optical extrapolation.
  - Extrapolated ($N = 115$): Shooter position smoothed or interpolated across occlusions.
- Subgroups are evaluated independently; no cross-cohort paired comparisons are made.

### Item 8: Statistical Claims Justification on Small Cluster Samples ($K = 10$)
- **Inferential Soundness:** In small-cluster regimes ($K = 10$), cluster bootstrap intervals carry substantial uncertainty.
- **Evaluation of Team Claims:**
  - The team did **not** claim statistical significance for shooter speed.
  - The team reported that the 95% bootstrap CI $[-0.002177, +0.000854]$ crosses zero and retained M1 as the official benchmark.
  - The team correctly identified that directional decomposition degrades performance ($\Delta \text{LogLoss} = +0.001080$, 95% CI $[+0.000211, +0.002238]$ strictly worse).
- The auditor confirms that all claims in the research report strictly adhere to sound statistical reporting standards.

---

## 3. Methodological Audit Summary Table

| Audit Criterion | Verification Method | Finding | Status |
| :--- | :--- | :--- | :---: |
| **Outer Fold Isolation** | Code inspection & `test_outer_and_inner_fold_strict_isolation` | Zero game overlap between outer train and test | **PASSED** |
| **Nested Calibration Provenance** | Provenance inspection & `test_resubstitution_vs_crossfitted_platt_calibration` | Calibrator fit solely on inner cross-fitted predictions | **PASSED** |
| **Preprocessing Anti-Leakage** | Pipeline inspection & `test_no_outer_test_preprocessing_leakage` | Scalers/imputers fit strictly inside outer training fold | **PASSED** |
| **Calibrated Predictions Mapping** | Row index check & prediction bounds test | Calibrated predictions map 1:1 to outer test shots | **PASSED** |
| **Paired Cluster Bootstrap** | Resampling code audit & `test_cluster_bootstrap_resampling_integrity` | Intact game clusters resampled with replacement ($B=2000$) | **PASSED** |
| **Metric Reproducibility** | Automated recalculation & `test_metrics_consistency_and_provenance` | Exact reproduction of all reported metrics | **PASSED** |
| **Temporal Integrity** | Timestamp inspection & `test_no_post_release_data_usage` | Release frame ($t$) strictly respected; no post-release data | **PASSED** |
| **Inference Justification** | Statistical review of bootstrap CIs and sample size | No unjustified significance claims; honest reporting | **PASSED** |

---

## 4. Auditor Recommendations for Benchmark Freezing

1. **Formal Freeze of M1 Benchmark:** Maintain M1 as the authoritative benchmark for basketball shot conversion on this dataset.
2. **Classify Shooter Velocity as Exploratory:** Document shooter speed as an unconfirmed exploratory hypothesis.
3. **Exclude Directional Velocity:** Permanently exclude directional decomposition from proposed models due to statistically confirmed performance degradation.
4. **Distinguish Validation Dimensions:** Ensure submission documentation explicitly differentiates:
   - Code Reproducibility (Verified / GO)
   - Methodological Defensibility (Verified / GO)
   - External Validation (Unverified / NO-GO; requires independent matches outside the 10-game ACB sample).
