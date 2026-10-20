# CourtIQ — Phase 2 Research Plan & Experimental Protocol

**Project:** CourtIQ: Pressure-Aware Shot Quality and Decision Audit  
**Phase:** Phase 2 Research Plan (M0 vs M1 vs M2 Protocol)  
**Author:** Lead Sports Analytics Researcher & Tracking Data Engineer  
**Status:** Approved for Implementation Following Feasibility Audit  

---

## 1. Experimental Overview & Research Hypothesis

### 1.1 Validated Research Question
> **“Does adding defensive-context and pre-release motion features measured within 400 ms of shot release improve held-out-game prediction of basketball field-goal outcomes beyond a baseline using shot location, basic shot characteristics, and static release-time defensive distance?”**

### 1.2 Target Hierarchy Alignment
This experiment strictly investigates **Target B (Pre-Release Shot Difficulty)**. It evaluates whether dynamic closing velocity and line-of-sight contest geometry predict shot conversion difficulty. In accordance with the Phase 2 Feasibility Audit findings, this experiment explicitly does **not** claim a causal decision audit (Target C) or evaluate whether the shooter "should have passed."

---

## 2. Experimental Cohort & Sample Integrity

### 2.1 Eligible Shot Cohort
The experiment evaluates exactly **1,322 eligible field-goal attempts** across all **10 Spanish Liga ACB 2025–26 matches**.

#### Matched-Row Requirement
- Every model comparison ($M_0$ vs $M_1$ vs $M_2$) must be evaluated on the **exact identical 1,322 rows**.
- If any lookback window or sensor gap drops an attempt in an exploratory variant, a strict matched-row subset must be evaluated in parallel so that performance differences reflect feature efficacy rather than sample composition shifts.
- Because the 400 ms lookback window achieves **100.00% complete tracking (1,322 / 1,322 shots)**, $M_2$ maintains 100% sample parity with $M_0$ and $M_1$.

### 2.2 Exclusion Pipeline
1. **Missing Outcome:** Exclude records missing binary label `outcome`.
2. **Duplicate IDs:** Exclude duplicate `id` keys.
3. **Missing Game ID:** Exclude records missing `gameId`.
4. **Fouled Misses (111 records):** Excluded per FIBA official boxscore rules. Under FIBA scoring, a missed shot with a shooting foul is classified as a free-throw trip and not an FGA. Including them distorts shooting percentages.
5. **Unlocated Shots (26 records):** Excluded due to missing spatial tracking at release.

---

## 3. Model Architectures & Feature Specifications

### 3.1 Model M0: Baseline Shot Characteristics
- **Numeric Features:**
  - `distance` (shooter-to-hoop Euclidean distance at release, in feet).
  - `shotClock` (seconds remaining on possession clock at release).
  - `releaseTime` (seconds from touch reception to release).
  - `dribblesBefore` (number of dribbles taken during the touch).
- **Categorical Features:**
  - `shotType` (jumper, layup, floater, post, heave).
  - `three` (binary 3PT vs 2PT).
  - `catchAndShoot` (binary indicator).
  - `createdFromPaint` (binary indicator).
  - `period` (quarter / overtime index 1–5).

### 3.2 Model M1: Static Release Defensive Context
- All features in $M_0$, plus:
  - `closestDefDist` (Euclidean distance to closest defender at release, in feet).
  - `contestLevel` (SkillCorner qualitative contest tier: open, light, average, plus, blocked).
  - `contested` (binary contest indicator).

### 3.3 Model M2: Pre-Release Motion & Contest Geometry
- All features in $M_1$, plus pre-release tracking features calculated over $[t - 400\text{ ms}, t]$ (10 frames at 25 fps):
  - `defender_closing_velocity` ($v_{\text{close}}$ in ft/s):
    $$v_{\text{close}} = \frac{D(t - 400\text{ms}) - D(t)}{0.40}$$
    Sign convention: $>0$ approaching, $<0$ receding, $0$ static.
  - `delta_defender_dist` ($\Delta D = D(t) - D(t - 400\text{ms})$ in feet).
  - `contest_angle_cos` ($\cos(\theta)$):
    $$\cos(\theta) = \hat{u}_{\text{hoop}} \cdot \hat{u}_{\text{def}}$$
    where $\hat{u}_{\text{hoop}}$ points from shooter to basket $(-40.75, 0.0)$ ft, and $\hat{u}_{\text{def}}$ points from shooter to defender.
  - `shooter_velocity_magnitude` ($v_{\text{shooter}}$ in ft/s):
    $$v_{\text{shooter}} = \frac{\|\vec{x}_s(t) - \vec{x}_s(t - 400\text{ms})\|_2}{0.40}$$

---

## 4. Cross-Validation & Leakage Safeguards

### 4.1 Grouped Leave-One-Game-Out (LOGO)
- Cross-validation partitions the dataset by `gameId` into $K = 10$ folds.
- In each fold $k \in \{1, \dots, 10\}$:
  - Held-out test fold: All shots from match $k$ ($N_{\text{test}} \approx 114\text{--}145$).
  - Training fold: All shots from the other 9 matches ($N_{\text{train}} \approx 1,177\text{--}1,208$).

### 4.2 Inside-Fold Preprocessing
- Preprocessing pipelines are fit strictly on training splits:
  - Missing numeric values: Imputed using training-fold median (`SimpleImputer(strategy="median")`).
  - Categorical encoding: Target or one-hot encoding fit strictly on training-fold categories (`handle_unknown="ignore"`).
  - Continuous scaling: Standardized using training-fold mean and variance (`StandardScaler`).
- Zero data from the held-out game is accessed during fitting, imputing, or scaling.

### 4.3 Blacklist Enforcement
The feature pipeline enforces an automated exception if any of the following fields are passed to model matrices:
`['outcome', 'shotQuality', 'fouled', 'foulId', 'assisted', 'assistOpp', 'passerId', 'blocked', 'blockerId', 'endFrame', 'endWallClock', 'endGameClock', 'acb_shotsaggregates_20252026.csv']`.

---

## 5. Statistical Evaluation Protocol

### 5.1 Primary and Secondary Metrics

| Metric | Target | Formula / Definition | Primary Role |
|---|:---:|---|---|
| **Log Loss** | Minimize | $-\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right]$ | **Primary decision metric** (strictly penalizes miscalibrated probabilities) |
| **Brier Score** | Minimize | $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$ | Secondary mean squared probability error |
| **ROC-AUC** | Maximize | Area under receiver operating characteristic curve | Secondary ranking discrimination |
| **ECE (10 bins)** | Minimize | $\sum_{b=1}^{10} \frac{\|B_b\|}{N} \|\text{acc}(B_b) - \text{conf}(B_b)\|$ | Calibration assessment |

### 5.2 Aggregation and Statistical Inference
Because the 10 training folds share 88.9% of their training data, fold-level metrics are non-independent.
1. **Pooled Out-of-Fold Metrics:** Compute loss across all 1,322 stacked predictions as the primary benchmark.
2. **Cluster-Robust Inference:** Compute cluster bootstrap standard errors (resampling games with replacement, $B = 2,000$) to establish 95% confidence intervals on $\Delta \text{Log Loss}$.
3. **Game Win Rate:** Report the number of games (out of 10) where $M_2$ achieves lower log loss than $M_1$ and $M_0$.

---

## 6. Ablation & Sensitivity Design

### 6.1 Feature Group Ablations
To isolate the marginal value of each motion component, four nested ablations will be tested under the identical LOGO harness:
- **Ablation 1 ($M_{2\text{a}}$):** $M_1$ + Defender Closing Velocity ($v_{\text{close}}$) only.
- **Ablation 2 ($M_{2\text{b}}$):** $M_1$ + Contest Angle ($\cos(\theta)$) only.
- **Ablation 3 ($M_{2\text{c}}$):** $M_1$ + Closing Velocity + Contest Angle.
- **Ablation 4 ($M_{2\text{d}}$ - Full):** $M_1$ + Closing Velocity + Contest Angle + Shooter Velocity + Distance Delta.

### 6.2 Computer Vision Detection Sensitivity
- Stratify out-of-fold performance across:
  - **Detected Subsample ($N = 1,207$, 91.3%):** Both shooter and defender tracked directly via optical detection (`isDetected == 1`).
  - **Extrapolated Subsample ($N = 115$, 8.7%):** At least one player position extrapolated by tracking filter (`isDetected == 0`).
- Test whether kinematic feature benefits are concentrated in high-confidence detected frames.

---

## 7. Explicit Feature Retention & Acceptance Criteria

A feature group or model variant will be retained and promoted to production modeling only if it satisfies all of the following empirical criteria:

1. **Primary Gain:** $\Delta \text{Log Loss} \le -0.0050$ relative to $M_1$ on pooled out-of-fold predictions.
2. **Non-Degraded Discrimination:** $\Delta \text{ROC-AUC} \ge 0.000$ (no ranking deterioration).
3. **Calibration Stability:** Pooled ECE does not deteriorate by more than $+0.0050$ relative to $M_1$.
4. **Majority Win Rate:** Outperforms $M_1$ on log loss in at least **6 out of 10** held-out games.
5. **Matched Row Verification:** Exactly identical 1,322 rows evaluated with zero data leakage.

---

## 8. Step-by-Step Implementation Protocol

```
Phase 2 Audit (Complete)
       │
       ▼
[Step 1: Feature Extraction Script]
  • Extract v_close, delta_D, cos(theta), v_shooter for 1,322 shots from raw tracking JSONL.gz
  • Store to data/processed/phase2_features.csv (read-only cached feature matrix)
       │
       ▼
[Step 2: Regression & Geometry Verification]
  • Run pytest tests/test_geometry.py (verify approaching vs receding signs and angles)
  • Verify zero NaN values across the 1,322 eligible rows
       │
       ▼
[Step 3: LOGO Modeling & Evaluation]
  • Run LOGO CV for M0, M1, and M2 ablations using strict inside-fold pipelines
  • Generate out-of-fold prediction CSVs
       │
       ▼
[Step 4: Metric & Calibration Report]
  • Compute pooled log loss, Brier score, ROC-AUC, ECE (10 bins)
  • Generate reliability diagrams
  • Check acceptance criteria for GO / NO-GO promotion
```
