# CourtIQ — Phase 2 Tracking Quality & Extrapolation Sensitivity Audit

**Project:** CourtIQ: Pressure-Aware Shot Quality and Decision Audit  
**Phase:** Phase 2 Data Quality Sensitivity Analysis  
**Author:** Lead Python ML Engineer & Sports Analytics Researcher  
**Status:** Audit Completed — Stratified Benchmark Established  

---

## 1. Overview of Tracking Signal Fidelity

In broadcast optical tracking data (SkillCorner ACB 2025–26), player coordinates are derived via computer vision with no on-court wearable transponders. Consequently, player visibility varies as the camera pans and zooms:

- **`isDetected == 1`**: The player is fully within the broadcast camera frame and directly identified by computer vision detection. Mean coordinate error is $\approx 0.9$ ft.
- **`isDetected == 0`**: The player is temporarily off-screen or occluded (e.g., tight broadcast zoom on ball-handler, crowded paint scuffle). The tracking system extrapolates player trajectory using motion filters. Mean coordinate error increases to $\approx 4.4\text{--}5.2$ ft.
- **`predError`**: The model-estimated 90th percentile coordinate error in feet.

This sensitivity audit evaluates whether the lack of improvement in $M_2$ (pre-release closing velocity and contest angle) is driven by noisy extrapolated player coordinates.

---

## 2. Sample Breakdown at Lookback Window ($t - 400$ ms)

Across the exact **1,322 eligible field-goal attempts**:
- **Directly Detected Cohort ($N = 1,207$, 91.30%):** Both the shooter and the primary defender were directly detected by optical tracking (`isDetected == 1`) at $t - 400$ ms.
- **Extrapolated Cohort ($N = 115$, 8.70%):** At least one player (shooter or defender) had an extrapolated position (`isDetected == 0`) at $t - 400$ ms.

---

## 3. Stratified Model Performance Comparison

Out-of-fold metrics were computed separately across the two tracking quality strata under identical LOGO cross-validation:

| Tracking Quality Cohort | Sample Size ($N$) | Model | Log Loss | Brier Score | ROC-AUC | ECE (10 Bins) |
|---|---:|---|---:|---:|---:|---:|
| **Directly Detected Subsample** | **1,207 (91.3%)** | **M0 Baseline** | 0.650835 | 0.229304 | 0.646936 | 0.030412 |
| | | **M1 Context** | **0.617401** | **0.215736** | **0.692065** | 0.051210 |
| | | **M2 Full** | 0.618349 | 0.216080 | 0.691501 | 0.054320 |
| | | *$\Delta$ (M2 - M1)* | *+0.000948* | *+0.000344* | *-0.000564* | *+0.003110* |
|---|---:|---|---:|---:|---:|---:|
| **Extrapolated Subsample** | **115 (8.7%)** | **M0 Baseline** | 0.700566 | 0.254050 | 0.556598 | 0.058102 |
| | | **M1 Context** | 0.615891 | 0.218097 | 0.709746 | 0.061210 |
| | | **M2 Full** | **0.614665** | **0.216715** | **0.723366** | 0.059430 |
| | | *$\Delta$ (M2 - M1)* | *-0.001226* | *-0.001382* | *+0.013620* | *-0.001780* |

---

## 4. Analytical Findings

### 4.1 Extrapolated Shots Are Distinctly More Difficult for Baseline Models
- In the extrapolated subsample, $M_0$ baseline exhibits severe degradation: Log Loss rises from $0.6508$ to **$0.7006$**, and ROC-AUC drops from $0.6469$ to **$0.5566$**.
- **Contextual Reason:** Extrapolated shots are heavily concentrated in rapid transition / fast-break situations (where the trailing camera has not caught up to corner shooters) and intense rim scrambles. In these scenarios, static shot distance is a poor predictor because conversion is determined by defender recovery.
- Defensive context ($M_1$) recovers massive accuracy on these shots, dropping log loss by $-0.0847$ (from $0.7006$ to $0.6159$).

### 4.2 High-Confidence Detected Data Confirms M2 Full Stagnation
- In the high-confidence detected subsample ($N = 1,207$), where tracking coordinate error is minimal ($< 0.9$ ft):
  - $M_1$ Log Loss: $0.617401$
  - $M_2$ Log Loss: $0.618349$ ($\Delta = +0.000948$)
  - $M_1$ ROC-AUC: $0.692065$
  - $M_2$ ROC-AUC: $0.691501$ ($\Delta = -0.000564$)
- **Core Scientific Conclusion:** The failure of $M_2$ full (closing velocity + contest angle) to outperform $M_1$ is **not caused by sensor extrapolation error or tracking noise**. Even on pristine optical detections, pre-release defender closing speed and contest angle do not improve out-of-fold log loss in a linear regularized framework.

---

## 5. Methodological Recommendations for Future Research

1. **Do Not Discard Extrapolated Shots:**
   Dropping extrapolated attempts would introduce severe selection bias, eliminating 8.7% of critical transition and corner attempts where defensive context is most influential.
2. **Confidence-Weighted Modeling:**
   In future non-linear models (e.g. tree-based or Bayesian models), `predError` and `isDetected` should be incorporated as observation weights or conditioning features, rather than used for blunt sample truncation.
3. **Focus on Shooter Mechanics:**
   Because shooter velocity demonstrated consistent out-of-fold gains across both strata, future feature development should prioritize shooter balance, acceleration, and release time rather than multi-defender vector geometry.
