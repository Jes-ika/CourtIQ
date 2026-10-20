# CourtIQ — Phase 2.5: Temporal Validity and Robustness Audit

**Project:** CourtIQ: Pressure-Aware Shot Quality and Decision Audit  
**Phase:** Phase 2.5 Skeptical Research & Temporal Validity Audit  
**Role:** Senior Sports Analytics Researcher, ML Validation Specialist & Tracking-Data Engineer  
**Status:** Audit Completed — Research Boundaries Formalized  

---

## 1. Executive Verdict

A rigorous, skeptical audit of the CourtIQ code repository, data pipeline, feature definitions, and Phase 2 experimental results was conducted. This audit focused specifically on the temporal validity of features, the exact prediction timestamp, the mathematical properties of the shooter-velocity feature, the cross-validation calibration methodology, and the defensibility of scientific claims.

### Summary Verdict
1. **Prediction Timestamp Alignment:**
   The entire existing pipeline operates at **$T_{\text{pred}} = t_{\text{release}}$ (`startFrame`)**. At this instant, the shot has already been physically initiated, the contest has materialized, and the basketball is leaving the shooter's hands.
2. **Research Target Reclassification:**
   - **Valid Formulation:** **Release-Time Shot Outcome Prediction (Execution Difficulty / Conditional Shot Quality)**. All features in $M_0$, $M_1$, and the backward-looking kinematics in $M_2$ are temporaly valid relative to $t_{\text{release}}$.
   - **Invalid Formulation:** **Pre-Release Decision-Quality Audit**. Claiming that the model audits pre-release decision-making or shot choice is scientifically ungrounded. At $t_{\text{release}}$, no decision alternative (passing or driving) is possible. Furthermore, release-time defensive distance (`closestDefDist`) and contest tier (`contestLevel`) represent future state relative to any pre-shot decision moment.
3. **Shooter-Velocity Implementation Finding:**
   The `shooter_velocity_magnitude` feature is strictly an **unsigned 2D scalar translation speed** ($\|\Delta \vec{x}\|_2 / \Delta t$), computed using the position at release ($t_{\text{release}}$) and 400 ms prior ($t_{\text{release}} - 0.40$ s). It contains **no directional information** (a step-back fadeaway yields the identical positive value as a forward rim drive at the same speed).
4. **Shooter-Velocity Statistical Significance:**
   While adding shooter velocity to $M_1$ improved pooled log loss by $-0.000758$ across 7 of 10 held-out matches, a cluster bootstrap across the 10 match clusters reveals a **95% confidence interval of $[-0.002176, +0.000854]$**, which **spans zero**. The observed gain is promising but statistically inconclusive on the 10-game sample.
5. **In-Fold Platt-Scaling Pitfall:**
   The current calibration routine fits Platt scaling directly on **resubstitution training predictions** (`p_train`) rather than nested out-of-fold predictions. Because the underlying model is already a logistic regression, fitting an unregularized logistic model on training logits is redundant ($\alpha \approx 1, \beta \approx 0$) and empirically deteriorated held-out test log loss and test calibration.
6. **Recommendation for Phase 3:**
   **CONDITIONAL GO** for modeling shot execution difficulty under release-time pressure, provided project claims and documentation are formally purged of "decision audit" assertions. **NO-GO** for claiming pre-release decision evaluation.

---

## 2. Prediction-Timestamp and Leakage Audit

### 2.1 Complete Data Flow & Source Semantics
The data pipeline traces through three layers:
1. **Raw Tracking Data (`{gameId}_tracking_data.jsonl.gz`):** Broadcast optical tracking at 25 fps.
2. **Dynamic Events (`{gameId}_dynamic_events.json`):** Play-by-play events derived by SkillCorner Game Intelligence models.
3. **Supervised Feature Matrix (`df_clean` + `phase2_features.csv`):** 1,322 eligible field-goal attempts across 10 matches.

### 2.2 Semantic Meaning of `startFrame`
According to SkillCorner data documentation ([`shots.md`](file:///C:/Users/jesik/.gemini/antigravity-ide/brain/6e657e94-5888-4246-919f-b9658277d018/scratch/opendata-basketball/docs/data_dictionary/shots.md) line 15):
- `startFrame`: *"Frame (at 25fps) since the start of video of the release of the shot."*
- `startWallClock`: *"Time in ms between the release of the shot and the start of the video."*
- `startGameClock`: *"Game clock in seconds at the release of the shot."*
- `endFrame`: *"Frame (at 25fps) since the start of the video of the end of the shot."*

`startFrame` is **not** shot initiation, ball pickup, or the moment of tactical decision. It is the exact frame where the ball departs the shooter's fingers ($t_{\text{release}}$).

### 2.3 Temporal Validity Relative to Prediction Timestamps

```
Timeline of a Basketball Shot Event:
──────────────────────────────────────────────────────────────────────────────────
   t_catch             t_decision           t_release (startFrame)       t_end (endFrame)
      │                    │                         │                         │
      │ ◄─── Touch ──────► │ ◄── Shooting Motion ──► │ ◄── Ball Flight ──────► │
      │                    │                         │                         │
      ▼                    ▼                         ▼                         ▼
   [Touch Start]       [Decision]             [Prediction T_pred]         [Outcome]
```

#### Evaluation Under $T_{\text{pred}} = t_{\text{release}}$ (Release-Time Prediction)
- **Status: VALID & LEAK-FREE.**
- Every feature utilized in $M_0$, $M_1$, and $M_2$ occurs at or prior to $t_{\text{release}}$:
  - `distance`, `closestDefDist`, `contestLevel`: measured at $t_{\text{release}}$.
  - `dribblesBefore`, `releaseTime`: describe touch history leading up to $t_{\text{release}}$.
  - Kinematics ($v_{\text{close}}$, $\cos(\theta)$, $v_{\text{shooter}}$): backward difference over $[t_{\text{release}} - 0.40\text{s}, t_{\text{release}}]$.
- No frames subsequent to `startFrame` are accessed.
- Post-shot fields (`endFrame`, `endWallClock`, `endGameClock`, `blocked`, `blockerId`, `fouled`, `foulId`, `shotQuality`) remain strictly blacklisted in [`src/courtiq/features/pipeline.py:14-27`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/pipeline.py#L14-L27).

#### Evaluation Under $T_{\text{pred}} = t_{\text{decision}} < t_{\text{release}}$ (Strictly Pre-Release Prediction)
- **Status: SEVERE FUTURE LEAKAGE.**
- If the model is framed as evaluating the player's choice *before* shooting:
  - `distance` at $t_{\text{release}}$ is in the future.
  - `closestDefDist` and `contestLevel` at $t_{\text{release}}$ are in the future.
  - `releaseTime` and `dribblesBefore` require knowing the termination of the touch.
  - `shooter_velocity_magnitude` uses `s_rel_pos` (position at $t_{\text{release}}$), which is in the future.
  - **Defender Selection Leakage:** In [`src/courtiq/features/kinematics.py:122,126`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/kinematics.py#L122-L126), `def_id = shot['closestDefId']`. The identity of the primary defender is determined by proximity at $t_{\text{release}}$. Looking up that same defender at $t - 0.40$ s retroactively conditions on who ended up closest at release, leaking future defensive rotations.

---

## 3. Feature-Level Temporal-Validity Table

The table below audits every feature across $M_0$, $M_1$, and $M_2$:

| Feature Name | Formula / Definition | Timestamps Used | Source File & Field | Missingness Handling | Valid at $T_{\text{pred}} = t_{\text{release}}$? | Valid at $T_{\text{pred}} < t_{\text{release}}$? | Temporal Validity & Implementation Concern |
|---|---|:---:|---|---|:---:|:---:|---|
| `distance` | $\|\vec{x}_s(t) - \vec{x}_{\text{hoop}}\|_2$ | $t_{\text{release}}$ | `shots.distance` | 0.0% in eligible set (26 unlocated filtered) | **Yes** | **No** | Measures release location. Future relative to decision. |
| `shotClock` | Seconds remaining on shot clock | $t_{\text{release}}$ | `shots.shotClock` | 0.48% missing; inside-fold median imputed | **Yes** | **No** | Differs from decision clock by elapsed shooting motion. |
| `releaseTime` | Elapsed seconds from touch catch to release | $[t_{\text{catch}}, t_{\text{release}}]$ | `shots.releaseTime` | 0.0% in eligible set | **Yes** | **No** | Requires observing release frame to compute touch duration. |
| `dribblesBefore` | Dribbles taken during touch | $[t_{\text{catch}}, t_{\text{release}}]$ | `shots.dribblesBefore` | 0.0% in eligible set | **Yes** | **No** | Requires observing entire touch up to release. |
| `shotType` | Categorical shot classification | Release instant | `shots.shotType` | 0.0% in eligible set | **Yes** | **No** | Classified based on release mechanics (layup, jumper, etc.). |
| `three` | Binary indicator ($>22.15$ ft FIBA arc) | $t_{\text{release}}$ | `shots.three` | 0.0% | **Yes** | **No** | Dependent on foot position at release. |
| `catchAndShoot` | Binary flag (releaseTime $\le 1.5$s or complexShotType) | $[t_{\text{catch}}, t_{\text{release}}]$ | `shots.catchAndShoot` | 0.0% in eligible set | **Yes** | **No** | Directly defined by total touch-to-release duration. |
| `createdFromPaint` | Touch origin indicator | Prior touch | `shots.createdFromPaint` | 0.0% | **Yes** | **Yes** | Backward-looking historical context from earlier in possession. |
| `period` | Quarter / Overtime index (1–5) | $t_{\text{release}}$ | `shots.period` | 0.0% | **Yes** | **Yes** | Game state constant across possession. |
| `closestDefDist` | Distance to nearest defender at release | $t_{\text{release}}$ | `shots.closestDefDist` | 0.0% in eligible set | **Yes** | **No** | Core $M_1$ feature. Measured strictly at release instant. |
| `contestLevel` | Contest tier: open, light, average, plus, blocked | $t_{\text{release}}$ | `shots.contestLevel` | 0.0% in eligible set | **Yes** | **No** | Core $M_1$ feature. Blocked shots are contestLevel='blocked' (inherent miss). |
| `contested` | Binary indicator (average, plus, blocked) | $t_{\text{release}}$ | `shots.contested` | 0.0% in eligible set | **Yes** | **No** | Derived directly from release `contestLevel`. |
| `defender_closing_velocity` | $(D(t - 0.4) - D(t)) / 0.4$ | $[t - 0.4\text{s}, t]$ | Tracking frames `sf-10`, `sf` | 0.0% in eligible set | **Yes** | **No** | Uses release distance $D(t)$ and defender identified at release. |
| `delta_defender_dist` | $D(t) - D(t - 0.4)$ | $[t - 0.4\text{s}, t]$ | Tracking frames `sf-10`, `sf` | 0.0% in eligible set | **Yes** | **No** | Linear scalar transformation of $v_{\text{close}}$ (redundant). |
| `contest_angle_cos` | $\hat{u}_{\text{hoop}} \cdot \hat{u}_{\text{def}}$ | $t_{\text{release}}$ | Tracking frame `sf` | 0.0% in eligible set | **Yes** | **No** | Geometry computed strictly at release instant. |
| `shooter_velocity_magnitude` | $\|\vec{x}_s(t) - \vec{x}_s(t - 0.4)\|_2 / 0.4$ | $[t - 0.4\text{s}, t]$ | Tracking frames `sf-10`, `sf` | 0.0% in eligible set | **Yes** | **No** | Backward difference ending at release frame `startFrame`. |

---

## 4. Separation of Research Questions

To ensure scientific integrity, the project must separate two distinct analytical targets:

### Formulation A: Release-Time Outcome Prediction (RECOMMENDED)
- **Research Question:** *"Given the shooter's location, movement speed, and defensive contest characteristics at the moment the shot is released, what is the probability that the shot is made?"*
- **Prediction Timestamp:** $T_{\text{pred}} = t_{\text{release}}$ (`startFrame`).
- **Valid Feature Set:** All features currently implemented in $M_0$, $M_1$, and backward kinematics ending at $t_{\text{release}}$.
- **Scientific Interpretation:** Measures **shot execution difficulty** and conditional conversion probability. Useful for quantifying shot quality, defensive contest suppression, and shooter execution under pressure.
- **Mandatory Rebranding:** Must drop all claims of evaluating "shot decision making" or "shot selection quality." Once released, the shooter has no decision alternatives.

### Formulation B: Strictly Pre-Release Outcome Prediction (DISCOURAGED FOR CURRENT DATA)
- **Research Question:** *"At the moment a player receives the ball or initiates an action, what is the expected value of choosing to shoot versus alternative options?"*
- **Prediction Timestamp:** $T_{\text{pred}} = t_{\text{catch}}$ or $t_{\text{decision}} \le t_{\text{release}} - 400$ ms.
- **Invalidity with Current Pipeline:** All primary features (`closestDefDist`, `contestLevel`, `distance`, `shooter_velocity_magnitude`) would have to be stripped because they use release-frame data.
- **Data Gap:** The dataset lacks counterfactual outcomes (what would have occurred had the player passed or driven). Observational tracking alone cannot estimate counterfactual value without severe confounding.

**Recommendation:** Proceed exclusively with **Formulation A**. Update project documentation to reflect **"CourtIQ: Release-Time Shot Quality and Defensive Pressure Audit"**.

---

## 5. Shooter-Velocity Implementation Audit

### 5.1 Verification of Code & Formulas
Inspected [`src/courtiq/features/geometry.py:67-80`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/geometry.py#L67-L80) and [`src/courtiq/features/kinematics.py:181-183`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/kinematics.py#L181-L183):

```python
def compute_shooter_speed(
    pos_pre: Sequence[float],
    pos_release: Sequence[float],
    delta_t: float = 0.4,
) -> float:
  dist = compute_euclidean_distance(pos_pre, pos_release)
  return dist / delta_t
```

#### Verified Properties:
1. **Units & Sampling:** Units are feet per second ($\text{ft/s}$). Frame rate is 25 fps. Lookback window is 10 frames ($0.40$ s). Calculation is numerically exact.
2. **Displacement Magnitude vs Vector Velocity:**
   The function calculates $v = \frac{\sqrt{(x_t - x_{t-0.4})^2 + (y_t - y_{t-0.4})^2}}{0.40}$.
   This is an **unsigned scalar translation speed** ($v \ge 0$). It does **not** preserve directional sign:
   - A shooter fading backwards away from the basket at $6$ ft/s receives $v = +6.0$ ft/s.
   - A cutter driving directly toward the basket at $6$ ft/s receives $v = +6.0$ ft/s.
   - A shooter sliding laterally along the three-point arc at $6$ ft/s receives $v = +6.0$ ft/s.
   The feature measures kinetic movement intensity, not directional momentum relative to the hoop.
3. **Release Frame Usage:**
   `pos_release` is extracted from `frame_release` at `startFrame`. It directly incorporates the shooter's coordinates at the moment of release.

### 5.2 Verification of Phase 2 Ablation Results
The ablation results reported in Phase 2 were independently audited:
- **Sample Size:** Exactly 1,322 shots across 10 matches (verified in [`reports/phase2_oof_predictions.csv`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase2_oof_predictions.csv)).
- **Preprocessing:** `StandardScaler` and `SimpleImputer` fit strictly inside each training fold using `Pipeline` (verified in [`src/courtiq/evaluation/phase2_cv.py:192-199`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase2_cv.py#L192-L199)).
- **Metric Comparison:**
  - $M_1$ Context Log Loss: **0.617270** | Brier: **0.215942** | ROC-AUC: **0.694100**
  - $M_1$ + Shooter Velocity Log Loss: **0.616512** | Brier: **0.215511** | ROC-AUC: **0.696695**
  - $\Delta \text{Log Loss} = -0.000758$ (-0.12% relative gain).

#### Per-Game Consistency and Cluster Bootstrap Audit
- **Game-Level Win Rate:** Shooter velocity improved log loss in **7 of 10 matches**:
  - Improved: 114086 (-0.0009), 114243 (-0.0024), 178442 (-0.0008), 179612 (-0.0032), 184439 (-0.0002), 188630 (-0.0054), 191313 (-0.0022).
  - Degraded: 114099 (+0.0012), 114169 (+0.0038), 114234 (+0.0019).
- **Cluster Bootstrap Uncertainty ($B = 2,000$ resamples of game clusters):**
  - Mean Delta: $-0.000724$
  - Standard Error: $0.000766$
  - **95% Confidence Interval:** **$[-0.002176, +0.000854]$**

**Critical Finding:** Because the 95% bootstrap confidence interval includes zero ($p > 0.15$), the $-0.000758$ log-loss improvement **cannot be claimed as statistically significant or definitively validated**. It is a weak exploratory signal that warrants cautious observation, not a confirmed breakthrough.

---

## 6. Reproducibility & Calibration Methodology Audit

### 6.1 Script Execution & Metric Reproducibility
- The entire evaluation pipeline was verified by running `scripts/run_phase2.py`.
- Generated metrics in [`reports/phase2_model_metrics.json`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase2_model_metrics.json) and [`reports/phase2_ablation_results.csv`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase2_ablation_results.csv) match reported values down to the 6th decimal place.
- All 40 unit and integration tests pass cleanly (`pytest -v`).

### 6.2 Critique of In-Fold Platt Scaling Implementation
Inspected [`src/courtiq/evaluation/phase2_cv.py:84-102, 201-209`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase2_cv.py#L84-L102):

```python
# Raw predictions
p_train = pipe.predict_proba(df_train)[:, 1]
p_test = pipe.predict_proba(df_test)[:, 1]

# Platt calibration
p_test_cal = fit_and_predict_platt_calibration(p_train, y_train, p_test)
```

#### Methodological Flaws Identified:
1. **Resubstitution Logit Fitting:**
   `p_train` is obtained by evaluating `pipe` on the **exact same training data** used to fit `pipe`. In machine learning, fitting a calibration model on resubstitution training probabilities is invalid because training predictions are inherently overconfident.
2. **Redundant Parameterization:**
   The base model `pipe` is an L2-regularized logistic regression. A secondary Platt scaling fits:
   $$P(Y=1) = \sigma(\alpha \cdot \text{logit}(p_{\text{train}}) + \beta)$$
   Because `pipe` was already trained by maximum likelihood to optimize logistic loss on `df_train`, on `df_train` the optimal slope is $\alpha \approx 1.0$ and intercept $\beta \approx 0.0$.
3. **Empirical Consequence:**
   Platt scaling fitted on resubstitution training logits fails to learn out-of-sample miscalibration. In fact, on held-out test games, Platt scaling **degraded** log loss:
   - $M_1$ uncalibrated Log Loss: **0.617270** $\to$ Platt calibrated Log Loss: **0.618127** (worse by $+0.000857$).
   - $M_2$ uncalibrated Log Loss: **0.618029** $\to$ Platt calibrated Log Loss: **0.619022** (worse by $+0.000993$).
4. **Correction Required for Future Phases:**
   Calibration models must be fit on **nested out-of-fold cross-validation predictions** (e.g. 5-fold cross-validation internal to each training split), or an internal validation hold-out set, never on training resubstitution logits.

---

## 7. Minimal Proposed Robustness Experiment

To rigorously validate whether shooter motion adds genuine signal without temporal contamination, the following minimal experiment is designed (to be executed only after Phase 2.5 approval):

### 7.1 Experimental Design Specifications

| Comparison Model | Definition | Key Research Hypothesis |
|---|---|---|
| **$M_1$ Benchmark** | Existing release-time context model (12 features) | Baseline benchmark |
| **$M_{1\text{, v\_rel}}$** | $M_1$ + existing release-lookback velocity ($\Delta t = [t_{\text{rel}} - 0.4\text{s}, t_{\text{rel}}]$) | Replicates Phase 2 exploratory result |
| **$M_{1\text{, v\_pre}}$** | $M_1$ + strictly pre-release velocity ($\Delta t = [t_{\text{rel}} - 0.8\text{s}, t_{\text{rel}} - 0.4\text{s}]$) | Tests if motion prior to the final shooting motion carries independent predictive value |
| **$M_{1\text{, v\_dir}}$** | $M_1$ + directional velocity decomposition (radial toward hoop vs tangential lateral) | Tests whether directional momentum resolves the ambiguity of scalar speed |
| **$M_{1\text{, detected\_only}}$** | Evaluated strictly on shots with `both_detected_pre == True` ($N = 1,207$) | Eliminates any sensor extrapolation noise |

### 7.2 Experimental Constraints
- **Matched Cohort:** Exactly identical 1,322 shots evaluated across primary comparisons.
- **Evaluation Metrics:** Out-of-fold Log Loss (primary), Brier Score, ROC-AUC, ECE at 5, 10, and 15 bins with reliability curves.
- **Inference:** Cluster bootstrap 95% confidence intervals across games ($B = 2,000$).
- **Success Criterion:** A feature variant is accepted only if:
  $$\Delta \text{Log Loss} \le -0.0020 \quad \text{AND} \quad \text{95% Bootstrap CI strictly below 0} \quad \text{AND} \quad \ge 7/10 \text{ games improved}$$

---

## 8. Specific Code Files & Line References Requiring Attention

| File Path | Line Range | Description of Issue & Required Modification |
|---|:---:|---|
| [`src/courtiq/features/kinematics.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/kinematics.py) | Lines 121–126 | `def_id = shot['closestDefId']`: Uses defender identity determined at release frame. If pre-release prediction is ever pursued, nearest defender must be computed dynamically at lookback frame. |
| [`src/courtiq/features/kinematics.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/kinematics.py) | Lines 181–183 | `compute_shooter_speed(s_pre_pos, s_rel_pos)`: Uses release-frame position `s_rel_pos`. Valid for $T_{\text{pred}} = t_{\text{release}}$, but must be documented explicitly as backward-looking from release. |
| [`src/courtiq/features/geometry.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/geometry.py) | Lines 67–80 | `compute_shooter_speed`: Calculates unsigned 2D Euclidean displacement speed without directional projection relative to hoop. Directional decomposition should be explored in future work. |
| [`src/courtiq/evaluation/phase2_cv.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase2_cv.py) | Lines 84–102 | `fit_and_predict_platt_calibration`: Fits Platt scaling on resubstitution training logits `p_train`. Must be replaced with nested out-of-fold cross-validation calibration. |
| [`src/courtiq/features/pipeline.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/pipeline.py) | Lines 29–33 | Feature lists lack explicit prediction timestamp metadata tags ($T_{\text{pred}} = t_{\text{release}}$). |

---

## 9. Evidentiary Classification & Phase 3 Recommendation

### Evidentiary Classification Table

| Finding / Claim | Classification | Empirical Basis |
|---|:---:|---|
| `startFrame` represents the release frame ($t_{\text{release}}$) | **Verified** | SkillCorner data dictionary (`shots.md` line 15), verified against tracking timestamps |
| All existing features are leak-free relative to $T_{\text{pred}} = t_{\text{release}}$ | **Verified** | Inspected blacklist, frame indexing ($\le \text{startFrame}$), and feature extraction code |
| Existing features leak future information relative to pre-release decision moment | **Verified** | `closestDefDist`, `contestLevel`, and release coords occur after decision |
| Shooter velocity is an unsigned scalar translation speed | **Verified** | Euclidean norm formula in `geometry.py:79` |
| Shooter velocity log-loss gain ($\Delta = -0.000758$) is statistically significant | **Not Supported by Current Evidence** | Cluster bootstrap 95% CI is $[-0.002176, +0.000854]$ (spans zero) |
| In-fold Platt scaling on training logits improves held-out calibration | **Not Supported by Current Evidence** | Degrades test log loss ($+0.000857$) and increases test ECE |
| Counterfactual shot decisions can be audited with current tracking data | **Not Supported by Current Evidence** | Causal identification deficit; unobserved counterfactual passes/drives |

---

### Final Go / No-Go Recommendation for Phase 3

#### Recommendation: **CONDITIONAL GO**
Phase 3 modeling may proceed **only under the following mandatory constraints**:

1. **Title & Research Scope Rebranding:**
   The project must formally drop "Decision Audit" from its claims. The valid scientific question is:
   > *"CourtIQ: Pressure-Aware Shot Quality and Outcome Prediction at Release."*
2. **Shooter Velocity Treatment:**
   `shooter_velocity_magnitude` must be treated as an **exploratory offensive movement feature**, with full transparency that its 95% bootstrap CI spans zero on the 10-game sample.
3. **Platt-Scaling Deprecation:**
   Resubstitution in-fold Platt scaling must be replaced by nested cross-validated calibration before reporting calibrated probabilities.
4. **No Causal Claims:**
   The model must not label shots as "bad decisions" or claim to optimize shot selection. Predictive shot difficulty conditional on release context is the sole defended output.
