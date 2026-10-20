# CourtIQ — Phase 2 Feature Audit: Pre-Release Kinematics & Contest Geometry

**Project:** CourtIQ: Pressure-Aware Shot Quality and Decision Audit  
**Phase:** Phase 2 Feature Engineering and Implementation Audit  
**Author:** Lead Python ML Engineer & Sports Analytics Researcher  
**Status:** Implementation Verified — Complete Sample Parity Established  

---

## 1. Executive Summary & Audit Overview

Following the approved Phase 2 Research Plan, pre-release kinematic and spatial geometry features were implemented in [`src/courtiq/features/kinematics.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/kinematics.py) and evaluated across the Spanish Liga ACB 2025–26 dataset.

### Summary of Extraction Results
- **Evaluated Cohort:** Exactly **1,322 eligible field-goal attempts** across all **10 matches**.
- **Lookback Window:** **400 ms** at 25 FPS ($\Delta t = 0.40$ s, 10 tracking frames prior to `startFrame`).
- **Feature Extraction Success Rate:** **1,322 / 1,322 (100.00%)**.
- **Missingness:** **0 missing values (0.00%)** across all kinematic features.
- **Matched-Row Guarantee:** Exactly identical row keys (`id`) and game groupings (`gameId`) maintained across all models ($M_0$, $M_1$, $M_2$, and ablations).

---

## 2. Mathematical Definitions & Coordinate Conventions

All kinematic features are computed strictly from frames $t - \Delta t$ and $t$, where $t = \text{startFrame}$ (shot release) and $\Delta t = 0.40$ s (10 frames at 25 fps).

### 2.1 Court Geometry & Hoop Location
- **Court Coordinate System:** Origin $(0.0, 0.0)$ ft is at center court. Length is oriented along the x-axis, width along the y-axis.
- **Offensive Hoop Position:** In SkillCorner's normalized ACB dynamic event coordinates, the offensive basket for every field goal attempt is located at:
  $$\vec{x}_{\text{hoop}} = (-40.75, 0.0) \quad \text{ft}$$
  Empirical verification: Across all 1,433 located shot events, the Euclidean distance from $\text{shots.location}$ to $(-40.75, 0.0)$ exactly equals $\text{shots.distance}$ with maximum absolute error $= 0.0000$ ft.

### 2.2 Mathematical Formulations

#### 1. Defender Closing Velocity ($v_{\text{close}}$)
Let $\vec{x}_s(t) = (x_s(t), y_s(t))$ be the 2D position of the shooter and $\vec{x}_d(t) = (x_d(t), y_d(t))$ be the 2D position of the primary defender (`closestDefId`) at release.  
Let $D(t) = \|\vec{x}_s(t) - \vec{x}_d(t)\|_2$ be the Euclidean distance.

$$v_{\text{close}} = \frac{D(t - 0.4) - D(t)}{0.4} \quad (\text{ft/s})$$

**Sign Convention and Physical Meaning:**
- $v_{\text{close}} > 0$: **Approaching Defender.** Distance shrank during shooting motion ($D(t) < D(t - 0.4)$). Defender closed in.
- $v_{\text{close}} < 0$: **Receding Defender.** Distance expanded during shooting motion ($D(t) > D(t - 0.4)$), e.g., fell back, bit on pump fake, or rotated away.
- $v_{\text{close}} = 0$: **Static Contest.** Separation distance remained unchanged.

#### 2. Distance Delta ($\Delta D$)
$$\Delta D = D(t) - D(t - 0.4) \quad (\text{ft})$$
- Note on mathematical redundancy: $\Delta D = -0.4 \times v_{\text{close}}$. Standardized scaling inside logistic regression maps any linear scalar transformation to the exact same feature. Consequently, including both in the same model creates perfect collinearity. It is evaluated solely as an alternative representation ablation.

#### 3. Contest Angle Cosine ($\cos(\theta)$)
Let $\hat{u}_{\text{hoop}} = \frac{\vec{x}_{\text{hoop}} - \vec{x}_s(t)}{\|\vec{x}_{\text{hoop}} - \vec{x}_s(t)\|_2}$ and $\hat{u}_{\text{def}} = \frac{\vec{x}_d(t) - \vec{x}_s(t)}{\|\vec{x}_d(t) - \vec{x}_s(t)\|_2}$.

$$\cos(\theta) = \hat{u}_{\text{hoop}} \cdot \hat{u}_{\text{def}}$$

**Physical Interpretation:**
- $\cos(\theta) = 1.0$ ($\theta = 0^\circ$): Defender directly on the line of sight between shooter and rim (maximal contest obstruction).
- $\cos(\theta) = 0.0$ ($\theta = 90^\circ$): Defender positioned laterally (side contest).
- $\cos(\theta) = -1.0$ ($\theta = 180^\circ$): Defender trailing directly behind the shooter.

#### 4. Shooter Velocity Magnitude ($v_{\text{shooter}}$)
$$v_{\text{shooter}} = \frac{\|\vec{x}_s(t) - \vec{x}_s(t - 0.4)\|_2}{0.4} \quad (\text{ft/s})$$

---

## 3. Preflight Alignment & Entity Integrity Checks

Before model fitting, five structural checks were verified on the live tracking stream:

1. **Defender Identity Persistence:**
   At release $t$, the primary defender is identified by `closestDefId`. At lookback $t - 10$, the extraction pipeline strictly retrieves the **exact same player ID**, ensuring that distance changes reflect the motion of that specific defender rather than substituting a different opponent who happened to be closer 400 ms earlier.
2. **Period Continuity:**
   In 100.0% of the 1,322 eligible attempts, both $t$ and $t - 10$ belong to the same period. Zero shots crossed inter-quarter or overtime boundaries.
3. **Live Play vs Stoppages:**
   1,301 shots (98.4%) had live game clock running at $t - 10$. The remaining 21 shots occurred immediately after inbound whistles or end-of-period buzzer beaters where clock started at release. In all 21 cases, player tracking coordinates were continuously tracked on court.
4. **No Post-Release Frames:**
   The extraction pipeline strictly accesses frames where $\text{frameIdx} \le \text{startFrame}$. Zero post-release trajectory data enters feature matrices.
5. **No Duplicate Shot Rows:**
   After joining tracking and dynamic events, `id` uniqueness was preserved: exactly 1,322 distinct rows with zero dropped observations.

---

## 4. Empirical Feature Distributions

Extracted values across all 1,322 eligible attempts saved in [`data/processed/phase2_features.csv`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/data/processed/phase2_features.csv):

| Feature Name | Symbol | Units | Mean | Std | Min | 25% | Median | 75% | Max | Missingness |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Defender Closing Velocity** | $v_{\text{close}}$ | ft/s | +3.625 | 4.586 | -10.572 | +0.655 | +2.852 | +5.938 | +20.155 | 0.0% |
| **Distance Delta** | $\Delta D$ | ft | -1.450 | 1.834 | -8.062 | -2.375 | -1.141 | -0.262 | +4.229 | 0.0% |
| **Contest Angle Cosine** | $\cos(\theta)$ | $[-1, 1]$ | +0.279 | 0.691 | -1.000 | -0.407 | +0.449 | +0.942 | +1.000 | 0.0% |
| **Contest Angle Degrees** | $\theta$ | deg | 73.81 | 48.06 | 0.00 | 19.53 | 63.31 | 114.01 | 180.00 | 0.0% |
| **Shooter Velocity Magnitude** | $v_{\text{shooter}}$ | ft/s | 5.923 | 3.931 | 0.064 | 2.825 | 5.176 | 8.448 | 19.280 | 0.0% |

### Key Distribution Insights
- **Closing Velocity Bias:** The mean closing velocity is $+3.625$ ft/s (positive), reflecting that in the 400 ms preceding release, defenders are overwhelmingly closing out or lunging toward the shooter. 79.2% of shots exhibit positive closing velocity.
- **Contest Angle Bimodality:** $\cos(\theta)$ shows a bimodal distribution with peaks near $+1.0$ (direct frontal contests, $4.1^\circ$ to $20^\circ$) and negative values (recoveries from behind or trailing guards on fast breaks and paint drives).
- **Shooter Speed:** Averages $5.92$ ft/s, spanning from stationary spot-up shooters ($< 1.0$ ft/s) to high-speed off-dribble pull-ups and cutting layups ($> 15.0$ ft/s).
