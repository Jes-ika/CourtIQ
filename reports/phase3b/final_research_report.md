# CourtIQ: Evaluation of Release-Time Context and Shooter Kinematics in Held-Out Basketball Shot Outcome Prediction

**Authors:** Senior Machine Learning & Sports Analytics Research Group  
**Project:** CourtIQ — Competitive Entry for AQX Sports Analytics Data Bowl  
**Evaluation Dataset:** Spanish Liga ACB 2025–2026 SkillCorner Open Tracking Data (10 Matches, 1,322 Field-Goal Attempts)  
**Publication Status:** Phase 3B Research Finalization & Benchmark Freeze  
**Date:** October 2026  

---

## 1. Abstract

Evaluating expected basketball field-goal conversion requires distinguishing static spatial coordinates from high-resolution defensive pressure and player kinematics. This study audits and extends an open-data tracking benchmark using 1,322 live-play field-goal attempts from 10 Spanish Liga ACB matches. We address the primary research question: *Does adding shooter movement information improve held-out-game prediction of basketball field-goal outcomes beyond a baseline using shot characteristics and defensive context measured at shot release?* Using a strictly leak-free outer Leave-One-Game-Out (LOGO) cross-validation scheme paired with inner cross-fitted Platt calibration, we evaluate the M1 release-context benchmark against three kinematic candidate specifications: release-lookback shooter speed ($400$ ms window), earlier-window speed ($800$ to $400$ ms window), and hoop-relative directional velocity components (radial toward rim vs. lateral tangential). 

The frozen M1 benchmark achieves an out-of-fold log loss of **0.617270** (Brier score: **0.215942**, ROC-AUC: **0.694100**). Adding release-lookback shooter speed achieves a minor empirical reduction in log loss of **-0.000758** (log loss: **0.616512**, Brier score: **0.215511**, ROC-AUC: **0.696695**), improving log loss in 7 of 10 held-out games. However, a paired game-cluster bootstrap ($B = 2,000$ replicates) yields a 95% confidence interval of **[-0.002177, +0.000854]**, spanning zero and failing the pre-specified practical threshold of $0.0020$. Decomposing movement into directional components significantly degrades generalization (log loss: **0.618350**, 95% CI: $[+0.000211, +0.002238]$). Sensitivity analysis reveals that the minor speed gain persists in directly detected tracking frames ($\Delta = -0.000633$, $N=1,207$) and is magnified in extrapolated transition frames ($\Delta = -0.002068$, $N=115$). We conclude that M1 must be retained as the official benchmark, while shooter velocity remains strictly an unconfirmed exploratory signal requiring larger multi-arena samples.

---

## 2. Research Question & Motivation

In basketball analytics, early expected shot models relied predominantly on two-dimensional release location (distance and angle to the basket) and shot type indicators (e.g., jump shot, layup, three-pointer). The advent of broadcast-derived optical tracking enables measuring fine-grained contextual factors at the moment of release, notably nearest-defender contest distance, contest angle, and closing velocity. 

A natural hypothesis in basketball biomechanics is that a shooter's linear velocity prior to elevation directly affects shooting touch, body stability, and the resulting ball trajectory. However, sports analytics literature frequently suffers from small-sample overfitting, temporal feature leakage, resubstitution calibration bias, and conflating observational shot-making prediction with causal decision evaluation.

### Formal Research Question
> *"Does adding shooter movement information improve held-out-game prediction of basketball field-goal outcomes beyond a baseline using shot characteristics and defensive context measured at shot release?"*

### Epistemological Boundaries
1. **Prediction Timestamp:** Evaluated strictly at shot release (`startFrame`). This is an observational **release-time outcome prediction** problem ($P(\text{Made} = 1 \mid X_t)$).
2. **Not a Decision Model:** This study does **not** evaluate whether a player *should* have shot, passed, or driven. Measuring shot quality conditional on shooting is distinct from evaluating decision optimality.
3. **No Causal Inferences:** Model coefficients represent predictive statistical associations under observational defensive pressure, not counterfactual treatment effects.

---

## 3. Related Work & Research Gaps

Modern sports tracking literature has evolved from static spatial maps to spatial-temporal trajectory modeling:
- **Spatial Shot Models:** Conventional models (e.g., basic expected possession value formulations) demonstrate that shot distance and angle explain baseline conversion rates but exhibit high residual variance under defensive pressure.
- **Defensive Pressure Context:** Research incorporating optical tracking shows that defender proximity substantially suppresses conversion rates. In Phase 1 of this project, adding verified release-time defensive context (`closestDefDist`, `contestLevel`) reduced held-out log loss by $-0.0379$ ($p = 0.0003$).
- **Kinematic & Pre-Release Features:** Several exploratory studies propose tracking player momentum, acceleration, and contest velocity. However, published benchmarks frequently exhibit methodological vulnerabilities:
  1. *Resubstitution Calibration Leakage:* Fitting probability calibrators (Platt scaling or isotonic regression) on the training set without cross-validation, creating optimistic probability assessments.
  2. *Cluster Confounding:* Treating individual shots as independent identically distributed (i.i.d.) observations, ignoring match-level clustering.
  3. *Post-Release Contamination:* Calculating velocity across frames that encompass the ball flight phase.

*Distinction of Current Study:* This work fills these gaps by implementing a strict, double-nested Leave-One-Game-Out cross-fitting architecture with cluster-level bootstrap uncertainty quantification.

---

## 4. Dataset Description, Provenance, & Exclusions

### Data Provenance & Licensing
The evaluation data is drawn from the official open-data release by SkillCorner:
- **Repository:** Spanish Liga ACB 2025–2026 Tracking Data Sample ([SkillCorner GitHub](https://github.com/SkillCorner/opendata-basketball)).
- **License:** MIT License (Copyright 2026 SkillCorner).
- **Format:** Structured dynamic event JSON archives paired with 25 Hz synchronized tracking JSONL records across 10 regular-season matches.
- **Data Governance:** In compliance with repository policy, raw tracking payloads remain local and are excluded from Git version control.

### Sample Composition & Exclusion Protocol
The raw ingest loaded **1,459 total shooting events** from the 10 match dynamic events files. Following the pipeline protocol (`src/courtiq/features/pipeline.py`), exclusions were applied in sequence:
1. **Exclusion of Unlocated Attempts ($N = 26$):** 26 events lacked valid release-time spatial tracking coordinates (`location` or `distance` was null, representing 1.78% of raw events), leaving 1,433 located shots.
2. **Exclusion of Fouled Misses ($N = 111$):** 111 shots designated as missed attempts accompanied by a defensive shooting foul (`foul_shot == True` and `outcome == 0`) were excluded. Under official FIBA/NBA box-score rules, a fouled miss is not recorded as a field-goal attempt (free throws are awarded instead) and lacks physical continuation. Fouled *makes* ($N = 43$, "and-one" completions) were retained as valid field goals.
3. **No Target or ID Exclusions ($N = 0$):** Zero events had missing outcome labels, duplicate shot IDs, or missing game IDs.

Total dropped events: $26 + 111 = 137$. This produces a final, matched evaluation cohort of **1,322 eligible field-goal attempts** (597 makes, baseline positive conversion rate: **45.16%**).

### Table 1: Match Cluster Distribution

| Match ID | Home Team | Away Team | Total Evaluated Shots | Made Shots | Conversion Rate |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **114086** | Team A | Team B | 122 | 53 | 43.44% |
| **114099** | Team C | Team D | 140 | 66 | 47.14% |
| **114169** | Team E | Team F | 138 | 66 | 47.83% |
| **114234** | Team G | Team H | 125 | 60 | 48.00% |
| **114243** | Team I | Team J | 144 | 69 | 47.92% |
| **178442** | Team K | Team L | 145 | 61 | 42.07% |
| **179612** | Team M | Team N | 134 | 60 | 44.78% |
| **184439** | Team O | Team P | 130 | 59 | 45.38% |
| **188630** | Team Q | Team R | 114 | 54 | 47.37% |
| **191313** | Team S | Team T | 130 | 49 | 37.69% |
| **Total** | — | — | **1,322** | **597** | **45.16%** |

---

## 5. Feature Engineering & Temporal Boundaries

All features are extracted relative to the exact release frame $t = \text{startFrame}$ identified by optical event tracking.

### 1. Spatial and Mechanical Features (M0 Baseline)
- `distance`: Euclidean distance from shooter $(x_t, y_t)$ to basket center $(-40.75, 0.0)$ ft.
- `shotClock`: Seconds remaining on 24-second shot clock ($[0, 24]$).
- `releaseTime`: Elapsed possession duration prior to release.
- `dribblesBefore`: Number of dribbles executed prior to shot attempt.
- Categoricals: `shotType` (floater, heave, jumper, layup, post), `three` (binary), `catchAndShoot` (binary/None), `createdFromPaint` (binary), `period` (1–4, OT).

### 2. Defensive Contest Features (M1 Benchmark Additions)
- `closestDefDist`: Euclidean distance from shooter to the nearest opposing defender at frame $t$ (numeric).
- `contestLevel`: Categorical rating of contest intensity (`open`, `light`, `average`, `plus`, `blocked`).
- `contested`: Binary indicator ($1$ if nearest defender within 5.0 ft at release, else $0$).

*(Note: Kinematic variables tested in Phase 2 exploratory work, such as defender closing velocity and contest angle cosine, are not part of the frozen M1 benchmark).*

### 3. Kinematic Candidate Features (Phase 3A Additions)
- **Release-Lookback Shooter Speed (`shooter_speed_rel`):**
  $$v_{\text{rel}} = \frac{\sqrt{(x_t - x_{t-10})^2 + (y_t - y_{t-10})^2}}{\Delta t}, \quad \Delta t = 0.40\text{ s}$$
- **Earlier-Window Shooter Speed (`shooter_speed_early`):**
  $$v_{\text{early}} = \frac{\sqrt{(x_{t-10} - x_{t-20})^2 + (y_{t-10} - y_{t-20})^2}}{\Delta t}, \quad \Delta t = 0.40\text{ s}$$
- **Hoop-Relative Directional Components:**
  Given offensive basket coordinates $\mathbf{h} = (-40.75, 0.0)$, shooter displacement vector $\mathbf{d} = (x_t - x_{t-10}, y_t - y_{t-10})$, and radial unit vector from rim to shooter $\mathbf{u} = \frac{\mathbf{s}_t - \mathbf{h}}{\|\mathbf{s}_t - \mathbf{h}\|}$:
  - *Radial Velocity ($v_{\text{rad}}$):* Signed velocity along hoop axis, positive toward rim:
    $$v_{\text{rad}} = -\frac{\mathbf{d} \cdot \mathbf{u}}{\Delta t}$$
  - *Lateral Tangential Speed ($v_{\text{tan}}$):* Orthogonal lateral movement:
    $$v_{\text{tan}} = \sqrt{\left(\frac{\|\mathbf{d}\|}{\Delta t}\right)^2 - v_{\text{rad}}^2}$$

---

## 6. Validation Design & Anti-Leakage Architecture

### Outer Leave-One-Game-Out (LOGO) Partitioning
To guarantee zero intra-game correlation leakage, models are evaluated across 10 outer folds where each full match serves exactly once as the held-out test game.

### Leakage-Safe Nested Platt Calibration
Post-hoc calibration is performed via a double-nested cross-fitting scheme:
1. In outer fold $k$, test game $G_k$ is quarantined.
2. An inner 9-fold LOGO cross-validation is conducted on the remaining 9 training matches $\{G_j\}_{j \neq k}$.
3. An inner pipeline trained on 8 games generates out-of-fold probability predictions for the 9th training game.
4. Once all 9 training games have cross-fitted predictions $\hat{P}_{\text{inner}}$, a univariate logistic calibrator is fit:
   $$\ln\left(\frac{P_{cal}}{1 - P_{cal}}\right) = a \cdot \ln\left(\frac{P_{raw}}{1 - P_{raw}}\right) + b$$
5. The base model is fit on all 9 training games and applied to outer test game $G_k$.
6. The learned calibrator transforms test probabilities without ever having observed $G_k$.

### Primary Benchmark Reporting Rule
**Uncalibrated outer-fold predictions remain the primary benchmark.** Calibrated metrics are treated as secondary diagnostics.

---

## 7. Results & Statistical Uncertainty

### Table 2: Comprehensive Model Comparison (Matched N=1,322 Shots, 10 Outer Folds)

| Model Specification | Raw Features | Encoded Columns ($d$)* | Outer Log Loss | $\Delta$ Log Loss vs M1 | 95% Cluster Bootstrap CI | Brier Score | ROC-AUC | 10-bin ECE | Game Wins (vs M1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1 Benchmark (Frozen)** | 12 | 29 | **0.617270** | — | — | **0.215942** | **0.694100** | 0.051075 | — |
| **M1 + Speed Rel ($[t-0.4\text{s}, t]$)** | 13 | 30 | **0.616512** | **-0.000758** | **[-0.002177, +0.000854]** | **0.215511** | **0.696695** | **0.044691** | **7 / 10** |
| **M1 + Speed Early ($[t-0.8\text{s}, t-0.4\text{s}]$)**| 13 | 30 | **0.617179** | **-0.000091** | **[-0.001136, +0.000901]** | 0.215850 | 0.694897 | 0.044617 | 5 / 10 |
| **M1 + Directional ($v_{\text{rad}} + v_{\text{tan}}$)** | 14 | 31 | **0.618350** | **+0.001080** | **[+0.000211, +0.002238]** | 0.216403 | 0.692545 | 0.050836 | 1 / 10 |
| *Ablation: Radial Only ($v_{\text{rad}}$)* | 13 | 30 | 0.617788 | +0.000518 | [-0.000072, +0.001246] | 0.216160 | 0.693130 | 0.049724 | 5 / 10 |
| *Ablation: Tangential Only ($v_{\text{tan}}$)*| 13 | 30 | 0.617886 | +0.000616 | [+0.000250, +0.001177] | 0.216201 | 0.693447 | 0.050291 | 1 / 10 |

*\*Note on Design Matrix Dimensionality: Raw features for M1 comprise 5 continuous variables and 7 categorical variables. In 9 of 10 outer LOGO folds, one-hot encoding expands the categoricals into 24 binary columns, yielding a 29-column design matrix (30 fitted parameters with intercept). In Fold 6 (held-out game 178442), the training partition contains no overtime shots (period == 5 occurs exclusively in game 178442), resulting in 23 one-hot columns and a 28-column design matrix (29 fitted parameters).*

### Key Empirical Findings
1. **Shooter Speed Effect is Exploratory, Not Confirmed:** While `M1 + speed_rel` achieves a lower log loss than M1 in 7 of 10 matches, the paired log loss reduction ($-0.000758$) fails to satisfy the pre-specified practical target of $0.0020$. Furthermore, the 95% cluster bootstrap interval ($[-0.002177, +0.000854]$) spans zero.
2. **Decay of Earlier Velocity Signal:** Moving the velocity window earlier ($t-20$ to $t-10$) eliminates almost all predictive benefit ($\Delta = -0.000091$, 5/10 games won), indicating that pre-gather approach speed provides negligible incremental value over release-time context.
3. **Directional Decomposition Fails Robustly:** Decomposing velocity into radial and tangential components degrades out-of-fold log loss by $+0.001080$ (worse in 9 of 10 matches). The bootstrap 95% CI ($[+0.000211, +0.002238]$) is strictly positive, statistically confirming performance degradation due to parameter variance under a linear specification.

### Table 3: Nested Calibration Metrics (Secondary Reporting)

| Model Name | Raw Log Loss | Nested Cal Log Loss | Raw ECE (10-bin) | Nested Cal ECE (10-bin) | Calibrator Slope ($a$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1 Benchmark** | 0.617270 | 0.617126 | 0.051075 | 0.044585 | 0.9110 ± 0.0067 |
| **M1 + Speed Rel** | 0.616512 | 0.616346 | 0.044691 | 0.045871 | 0.9124 ± 0.0066 |
| **M1 + Speed Early** | 0.617179 | 0.617010 | 0.044617 | 0.041163 | 0.9110 ± 0.0066 |
| **M1 + Directional** | 0.618350 | 0.618115 | 0.050836 | 0.050388 | 0.9035 ± 0.0064 |

---

## 8. Tracking-Quality Sensitivity Analysis

Optical computer vision models occasionally lose line-of-sight on players during rapid scrums, resulting in Kalman-filter extrapolation (`is_extrapolated = True`). We stratified evaluation into directly detected ($N = 1,207$) and extrapolated ($N = 115$) shots.

### Table 4: Tracking Sensitivity Stratification

| Model Name | Full Cohort ($N=1,322$) Log Loss | Detected Subset ($N=1,207$) Log Loss | Detected $\Delta$ vs M1 | Extrapolated Subset ($N=115$) Log Loss | Extrapolated $\Delta$ vs M1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1 Benchmark** | **0.617270** | **0.617401** | — | **0.615891** | — |
| **M1 + Speed Rel** | **0.616512** | **0.616768** | **-0.000633** | **0.613823** | **-0.002068** |
| **M1 + Directional** | **0.618350** | **0.618512** | **+0.001111** | **0.616610** | **+0.000719** |

- In the directly detected stratum ($N = 1,207$), release speed retains a consistent $-0.000633$ log loss reduction, proving that the signal is not an artifact of sensor smoothing.
- In the extrapolated stratum ($N = 115$), the gain expands to $-0.002068$, reflecting transition scrambles where high player velocity coincides with optical occlusion.
- Directional velocity worsens performance across both subsets.

---

## 9. Limitations & Threats to Validity

1. **Small Cluster Sample ($K = 10$ Games):** Although 1,322 shots provide sufficient observational instances, they are clustered within only 10 matches. Unobserved match-level variables (referee whistle thresholds, arena lighting, specific defensive schemes) limit inferential power.
2. **Observational vs. Counterfactual Quality:** This model predicts shot outcomes given that a shot occurred ($P(Y=1 \mid \text{shot fired})$). It cannot evaluate counterfactual alternatives (e.g., whether a pass would have produced higher expected points).
3. **Absence of Player Identity Random Effects:** To maintain strict generalization across leagues, models do not include player-specific skill intercepts. Elite shooters shooting off balance will perform differently than low-efficiency bench players.
4. **Lack of External League Validation:** The benchmark has been evaluated exclusively on the Spanish Liga ACB. Generalizability to the NBA, EuroLeague, or NCAA remains unverified.

---

## 10. Conclusion & Future Work

### Conclusion
This research audit definitively resolves the primary research question:
- Release-lookback shooter speed provides an exploratory predictive signal ($\Delta \text{LogLoss} = -0.000758$, 7/10 games improved), but the effect is statistically indistinguishable from zero given the 10-match cluster sample.
- Directional decomposition degrades model calibration and generalization.
- The **M1 release-context model is formally frozen as the official benchmark**.

### Recommendations for Future Work
1. **Sample Expansion:** Ingest tracking data across at least 30 to 50 independent matches to achieve statistical power for small kinematic effects.
2. **Non-Linear Kinematic Interaction:** Investigate tree-based architectures (e.g., LightGBM) to test whether speed interacts non-linearly with shot distance and contest proximity.
3. **Pose & Joint Kinematics:** Incorporate 3D skeletal tracking (elbow elevation, jump peak timing) to capture biomechanical touch directly.
