# CourtIQ — Phase 2 Data-Feasibility Audit & Research Design

**Project:** CourtIQ: Pressure-Aware Shot Quality and Decision Audit  
**Author:** Lead Sports Analytics Researcher & Tracking Data Engineer  
**Date:** October 2026  
**Status:** Audit Completed — Research Design Established  

---

## 1. Executive Summary & Workstream Decisions

Before implementing new feature engineering or expanding predictive models to include pre-release kinematics, a thorough, read-only data-feasibility audit was conducted across the Spanish Liga ACB 2025–26 tracking dataset provided by SkillCorner.

This audit inspected all 10 available matches, 1,472,932 tracking frames, 1,459 raw shot events, and associated dynamic event tables (`possessions`, `passes`, `drives`, `touches`, `chances`). All claims have been empirically verified against the raw files, schemas, and coordinate transformations.

### Executive Decisions by Workstream

| Workstream | Scope | Verdict | Key Rationale & Constraints |
|---|---|:---:|---|
| **Workstream A** | Repository & Raw Tracking Audit | **GO** | All 10 games have complete tracking (`.jsonl.gz`) and dynamic event JSONs. Coordinate conventions, hoop location ($x=-40.75, y=0.0$), and player IDs are strictly verified. |
| **Workstream B** | Pre-Release Temporal Feasibility & Kinematics | **GO** | A 400 ms lookback window (10 frames at 25 fps) yields **100.00% frame and entity coverage** across all 1,322 eligible shots with zero sample attrition. Vector math is validated via unit tests. |
| **Workstream C** | Decision-Audit Feasibility (Alternative Options) | **NO-GO** *(for Target C)*<br>**GO** *(for Target B)* | **NO-GO** for counterfactual decision evaluation (passing/driving expected value) due to unobserved counterfactuals and planar $z=0$ tracking limits. **GO** for Target B (pre-release shot execution difficulty). |
| **Workstream D** | Leakage & Validation Audit | **GO** | Inside-fold preprocessing and game-isolated Leave-One-Game-Out (LOGO) cross-validation are verified. Season aggregates are strictly blacklisted. Fold-dependence inference guidelines are established. |
| **Workstream E** | Feature Feasibility Matrix | **CONDITIONAL GO** | Approved for closing velocity, distance delta, and contest angle; conditional for nearby defender count and shot clock; rejected for passing lane risk and counterfactual value. |
| **Workstream F** | Research Design & Experimental Plan (M0/M1/M2) | **GO** | Matched-row protocol designed comparing M0 vs M1 vs M2 on the identical 1,322 shots with log loss, Brier score, ROC-AUC, ECE calibration, and ablations. |

---

## 2. Workstream A — Repository and Data Audit

### 2.1 File Inventory and Usable Tracking Sample

The repository was audited across raw data paths and documentation. The available dataset consists of:
- **10 Complete Matches** with paired optical tracking (`{gameId}_tracking_data.jsonl.gz`), match metadata (`{gameId}_game_data.json`), and dynamic events (`{gameId}_dynamic_events.json`).
- **Season Aggregates** (`acb_shotsaggregates_20252026.csv`, `acb_drivesaggregates_20252026.csv`, `acb_picksaggregates_20252026.csv`) spanning 293 games. **Finding:** Season aggregates are strictly excluded from model feature sets to avoid post-hoc season-wide target leakage.

#### Verified Match Inventory

| Match ID | Date | Home Team | Away Team | Total Frames | Live Frames | Dead Frames | Evaluated Shots |
|---|---|---|---|---:|---:|---:|---:|
| **114086** | 2025-10-04 | Basquet Girona | Unicaja | 124,796 | 92,188 (73.9%) | 32,608 | 122 |
| **114099** | 2025-10-05 | Baxi Manresa | Valencia Basket | 147,892 | 75,655 (51.2%) | 72,237 | 140 |
| **114169** | 2025-10-11 | Real Madrid | Joventut Badalona | 172,257 | 106,176 (61.6%) | 66,081 | 138 |
| **114234** | 2025-10-18 | Lenovo Tenerife | Coviran Granada | 158,923 | 72,132 (45.4%) | 86,791 | 125 |
| **114243** | 2025-10-19 | Dreamland Gran Canaria | Surne Bilbao | 146,870 | 77,023 (52.4%) | 69,847 | 144 |
| **178442** | 2025-10-25 | MoraBanc Andorra | Leyma Coruña | 162,479 | 89,642 (55.2%) | 72,837 | 145 |
| **179612** | 2025-11-01 | Casademont Zaragoza | FC Barcelona | 140,856 | 83,437 (59.2%) | 57,419 | 134 |
| **184439** | 2025-11-08 | UCAM Murcia | Rio Breogan | 154,021 | 77,428 (50.3%) | 76,593 | 130 |
| **188630** | 2025-11-15 | Hiopos Lleida | Baskonia | 121,793 | 73,138 (60.1%) | 48,655 | 114 |
| **191313** | 2025-11-22 | La Laguna Tenerife | Baxi Manresa | 143,045 | 82,885 (57.9%) | 60,160 | 130 |
| **Total** | — | — | — | **1,472,932** | **829,704 (56.3%)** | **643,228** | **1,322** |

### 2.2 Frame Characteristics and Coordinate System

- **Sampling Frequency:** Fixed at **25 frames per second** (1 frame = 40.0 ms).
- **Court Dimensions & Coordinate Conventions:**
  - Origin $(0, 0)$ is at the exact physical center of the court.
  - Coordinate units are in **feet**.
  - All event and tracking tables share the same coordinate convention: length along the x-axis, width along the y-axis.
  - For every attacking action in `shots` and `tracking_data`, the offensive hoop is located on the **negative x-axis**.
  - **Empirically Fitted Hoop Location:** $\vec{x}_{\text{hoop}} = (-40.75, 0.0)$ ft. Least-squares regression against reported shot locations and release distances yielded a residual sum of squares of $1.69 \times 10^{-11}$, confirming the exact center of the basket.
- **Dead Time Behavior:**
  - 43.7% of all recorded frames represent dead time (stoppages, free throws, timeouts, inter-quarter breaks).
  - On dead frames, `homePlayers`, `awayPlayers`, and `ball` are delivered as empty arrays `{}`.
  - Timestamps (`frameIdx`, `wallClock`, `gameClock`, `period`) remain strictly continuous across dead time.
- **Player & Team Linking:**
  - Offense and defense are linked via `offTeamId` and `defTeamId`.
  - Shooter identity is linked via `shots.shooterId` to `tracking_data.homePlayers[].playerId` or `awayPlayers[].playerId`.
  - Primary defender is identified by `shots.closestDefId`.
  - On-court participation is recorded per chance in `chance_players`.

### 2.3 Evidence Classification Table

| Finding / Claim | Classification | Verification Source |
|---|:---:|---|
| 10 games with usable 25fps optical tracking | **Directly Observed** | Decompressed `.jsonl.gz` inspection across all 10 match folders |
| Origin at center court, units in feet | **Confirmed by Documentation** | `docs/PRIMER.md` lines 13–19, `docs/data_dictionary/README.md` |
| Offensive hoop located at $x = -40.75$ ft, $y = 0.0$ ft | **Directly Observed & Fitted** | Least-squares optimization on 1,322 shot coordinates vs `shots.distance` |
| 1,459 raw shots, 111 fouled misses, 26 untracked, 1,322 eligible | **Directly Observed** | Full dataset pipeline execution (`courtiq/features/pipeline.py`) |
| 400ms lookback window yields 100% coverage on eligible shots | **Directly Observed** | Temporal frame audit script (`audit_tracking_coverage.py`) |
| Player height $z$ is kept at $0.0$ ft (no jumping/verticality in player tracking) | **Confirmed by Documentation** | `docs/data_dictionary/tracking_data.md` line 29 |
| Counterfactual pass outcomes cannot be observed from tracking | **Directly Observed** | Dynamic events only record executed passes (`shots`, `passes`, `drives`) |

---

## 3. Workstream B — Pre-Release Temporal Feasibility & Motion Kinematics

### 3.1 Empirical Lookback Window Coverage

The existing release-time defensive features (`closestDefDist`, `contestLevel`) are measured at `startFrame`. To establish whether motion prior to release can be estimated, four candidate lookback windows were evaluated across all 1,322 eligible field goal attempts:

| Window | Lookback Frames | Duration ($\Delta t$) | Available Frames | Shooter & Defender Tracked | Both Detected by CV | All 10 Players Tracked |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Release** | 0 frames | 0 ms | 1,322 (100.0%) | 1,322 (100.0%) | 1,172 (88.7%) | 1,322 (100.0%) |
| **200 ms** | 5 frames | 200 ms | 1,322 (100.0%) | 1,322 (100.0%) | 1,191 (90.1%) | 1,322 (100.0%) |
| **400 ms** | 10 frames | 400 ms | **1,322 (100.0%)** | **1,322 (100.0%)** | **1,207 (91.3%)** | **1,322 (100.0%)** |
| **800 ms** | 20 frames | 800 ms | 1,322 (100.0%) | 1,318 (99.7%) | 1,222 (92.4%) | 1,319 (99.8%) |

#### Key Temporal Findings
1. **Zero Sample Attrition at 400 ms:** Exactly 1,322 of 1,322 shots (100.00%) have complete tracking for both the shooter and the primary defender across the entire $[t - 400\text{ ms}, t]$ interval.
2. **Detection Quality:** Over 91% of shooter and defender positions at $t - 400$ ms are directly detected by computer vision (`isDetected == 1`), with the remainder smoothly extrapolated (`predError` mean $< 4.5$ ft).
3. **800 ms Boundary Drop:** At 800 ms (20 frames), 4 shots cross a dead-ball whistle or period start boundary, reducing coverage to 1,318 / 1,322 (99.70%).
4. **Feasibility Verdict:** **400 ms ($\Delta t = 0.40$ s, 10 frames)** is selected as the optimal primary lookback window. It guarantees that the lookback occurs strictly during live play while avoiding sample attrition.

---

### 3.2 Formal Mathematical Formulations

All motion features are derived strictly from frames $t - \Delta t$ and $t$, where $t = \text{startFrame}$ (shot release).

#### 1. Defender Closing Velocity ($v_{\text{close}}$)
Let $\vec{x}_s(t) = (x_s(t), y_s(t))$ be the 2D position of the shooter and $\vec{x}_d(t) = (x_d(t), y_d(t))$ be the 2D position of the closest defender (`closestDefId`) at release.  
Let $D(t) = \|\vec{x}_s(t) - \vec{x}_d(t)\|_2$ be the Euclidean distance.

The closing velocity is defined as:
$$v_{\text{close}} = \frac{D(t - \Delta t) - D(t)}{\Delta t} \quad (\text{ft/s})$$

**Sign Convention and Physical Meaning:**
- $v_{\text{close}} > 0$: **Approaching Defender.** The defender was farther away at $t - \Delta t$ than at release ($D(t) < D(t - \Delta t)$). The defender is actively closing out or lunging into the contest.
- $v_{\text{close}} < 0$: **Receding Defender.** The defender was closer at $t - \Delta t$ and moved farther away at release ($D(t) > D(t - \Delta t)$), e.g., recovering from a pump-fake, giving space, or rotating away.
- $v_{\text{close}} = 0$: **Static Contest.** The defender maintained a constant separation distance relative to the shooter.

#### 2. Distance Change ($\Delta D$)
$$\Delta D = D(t) - D(t - \Delta t) \quad (\text{ft})$$
Negative values indicate distance closed ($\Delta D = -v_{\text{close}} \cdot \Delta t$).

#### 3. Contest Angle Geometry ($\cos(\theta)$)
Let $\vec{x}_{\text{hoop}} = (-40.75, 0.0)$ ft. Define the two unit directional vectors originating from the shooter at release:
- Direction to hoop:
  $$\hat{u}_{\text{hoop}} = \frac{\vec{x}_{\text{hoop}} - \vec{x}_s(t)}{\|\vec{x}_{\text{hoop}} - \vec{x}_s(t)\|_2}$$
- Direction to defender:
  $$\hat{u}_{\text{def}} = \frac{\vec{x}_d(t) - \vec{x}_s(t)}{\|\vec{x}_d(t) - \vec{x}_s(t)\|_2}$$

The contest alignment metric is the scalar projection (cosine):
$$\cos(\theta) = \hat{u}_{\text{hoop}} \cdot \hat{u}_{\text{def}}$$

**Physical Interpretation:**
- $\cos(\theta) = 1.0$ ($\theta = 0^\circ$): Defender is directly between the shooter and the hoop (maximal visual and physical obstruction).
- $\cos(\theta) = 0.0$ ($\theta = 90^\circ$): Defender is positioned purely laterally (side contest).
- $\cos(\theta) = -1.0$ ($\theta = 180^\circ$): Defender is directly behind the shooter (trailing contest, minimal disruption to rim sightline).

#### 4. Shooter Velocity Magnitude ($v_{\text{shooter}}$)
$$v_{\text{shooter}} = \frac{\|\vec{x}_s(t) - \vec{x}_s(t - \Delta t)\|_2}{\Delta t} \quad (\text{ft/s})$$

#### 5. Handling of Missing / Degenerate Observations
- If a defender cannot be tracked at $t - \Delta t$, the observation is imputed using the inside-fold median of $v_{\text{close}}$. (In the 400ms window across the 10 ACB games, missingness is 0.0%).
- If $D(t) < 10^{-4}$ ft (degenerate coordinate collision), $\cos(\theta)$ is assigned $0.0$ ($\theta = 90^\circ$) to avoid division by zero.

---

## 4. Workstream C — Decision-Audit Feasibility Assessment

A central motivation in modern sports analytics is establishing whether an athlete made the "right decision" on a possession. This feasibility audit examined whether the ACB tracking dataset supports a rigorous shot-decision audit.

### 4.1 Data Element Availability for Decision Auditing

| Decision Element | Status | Source File & Field | Limitations / Deficits |
|---|:---:|---|---|
| Teammate positions & IDs | **Present** | `dynamic_events.json [possessions.offPlayerIds]`, `tracking_data [home/awayPlayers]` | Coordinates are available at $z=0$. |
| Opponent defensive assignments | **Partially Available** | `dynamic_events.json [matchups]` | Intervals only record assigned matchup pairs, not defensive rotations or help recovery vectors. |
| Ball possession continuity | **Present** | `dynamic_events.json [possessions, chances, touches]` | Accurately segments continuous team control. |
| Passing lane openness & interception risk | **Absent** | *None* | Planar tracking ($z=0$) does not record defender arm reach, hand elevation, head orientation, or passing corridor interception probability. |
| Counterfactual pass outcomes | **Absent** | *None* | We observe the outcome of passes that were executed, but **never observe what would have occurred had a shooter passed instead of shot**. |
| Counterfactual drive outcomes | **Absent** | *None* | Drive events (`drives`) only exist when a player drove; counterfactual drive success cannot be observed. |
| Shot clock & game clock | **Present** | `shots.shotClock`, `shots.startGameClock` | Accurately recorded with $<0.5\%$ missingness. |

### 4.2 Hierarchy of Research Targets

To maintain scientific integrity, research objectives must be separated into three distinct targets:

```
[Target A: Shot Quality Prediction]  ==> GO (Current M0 / M1)
       |
       v
[Target B: Pre-Release Shot Difficulty] ==> GO (Proposed M2)
       |
       v
[Target C: Decision-Quality Audit]   ==> NO-GO (Fundamentally Unidentified)
```

1. **Target A — Shot-Quality Prediction (GO):**
   - *Question:* What is the expected probability of making the shot that was taken, given release-time characteristics?
   - *Status:* Validated by M0 (location) and M1 (release defensive distance/contest).
2. **Target B — Pre-Release Shot Difficulty (GO):**
   - *Question:* Does incorporating pre-release motion (defender closing velocity, shooter balance, contest angle) improve out-of-sample shot outcome prediction?
   - *Status:* Feasible. All required temporal and kinematic signals exist with 100% coverage at 400 ms.
3. **Target C — Decision-Quality Audit (NO-GO):**
   - *Question:* Did the shooter make the optimal decision compared to passing to an open teammate or driving to the rim?
   - *Status:* **NO-GO.**

### 4.3 Why Target C Cannot Be Supported by the Current Data

Evaluating decision optimality requires estimating the counterfactual value function:
$$\Delta \text{EV} = \text{EV}(\text{Shot Taken}) - \max\left(\text{EV}(\text{Pass to Teammate } i), \text{EV}(\text{Drive})\right)$$

In observational tracking data, this comparison suffers from four fatal methodological flaws:
1. **The Fundamental Problem of Causal Inference:** When a player shoots, the potential outcomes of passing or driving are unobserved. Replacing counterfactual passes with unconditional teammate shooting averages induces severe selection bias, because shooters choose to shoot precisely when they perceive teammates to be covered or out of rhythm.
2. **No Defensive Biometrics or Arm Envelopes:** Planar $(x, y)$ tracking cannot determine if a passing lane is open. A defender positioned 5 feet away may have active hands directly deflecting the passing lane.
3. **Severe Confounding by Play Design:** Tactical instructions (e.g., designed isolation plays, end-of-shot-clock orders) dictate decisions regardless of spatial openness.
4. **Conclusion:** Labeling an attempt "suboptimal" or a "poor decision" merely because the shot had low expected quality or missed is scientifically ungrounded. Target C must not be claimed in this research phase.

---

## 5. Workstream D — Leakage and Validation Audit

### 5.1 Verification of Leave-One-Game-Out (LOGO) Isolation

The validation harness was audited to ensure strict independence between training folds and test folds:
1. **Group Variable Separation:** The grouping variable `gameId` cleanly partitions observations. In each fold, all shots from 1 held-out game are evaluated, while all shots from the remaining 9 games form the training set.
2. **Inside-Fold Preprocessing:** Preprocessing components (`SimpleImputer`, `StandardScaler`, and categorical encoders) are strictly instantiated and fitted inside each cross-validation fold using `Pipeline` or `make_pipeline`. No statistics (medians, means, standard deviations, vocabularies) are computed on the full dataset or held-out folds.
3. **Post-Shot Field Blacklist:** All fields occurring after `startFrame` are strictly excluded from feature extraction:
   `['outcome', 'shotQuality', 'fouled', 'foulId', 'assisted', 'assistOpp', 'passerId', 'blocked', 'blockerId', 'endFrame', 'endWallClock', 'endGameClock']`.
4. **Season Aggregate Blacklist:** `acb_shotsaggregates_20252026.csv` contains season-long totals (293 games) for players. It is strictly blacklisted to prevent future-label leakage.

### 5.2 Filter Auditing & Sample Integrity

The dataset cleaning pipeline was audited:
- **Raw Shots:** 1,459 attempts.
- **Fouled Misses Excluded:** 111 records. Under FIBA official rules, a field goal attempt missed during a foul is not recorded as an FGA (free throws are awarded instead). Including them would distort conversion probabilities and mismatch box scores.
- **Unlocated Shots Excluded:** 26 records (missing release-time spatial coordinates).
- **Final Evaluated Sample:** Exactly **1,322 eligible field goal attempts** across all 10 games.
- **Matched-Row Integrity:** M0, M1, and the proposed M2 are evaluated on the exact identical 1,322 shot attempts.

### 5.3 Limitations of 10-Game Cluster Cross-Validation

A critical methodological caution applies to the 10-game sample:
- In Leave-One-Game-Out CV with $K = 10$, each training fold shares 8 of 9 games (88.9%) with every other training fold.
- Consequently, test fold errors are statistically dependent.
- **Warning:** Standard paired $t$-tests or $p$-values across the 10 folds violate independence assumptions and will produce artificially narrow confidence intervals.
- **Recommendation:** Model superiority must be assessed through **pooled out-of-fold log loss**, cluster-robust bootstrap standard errors across games, and sign-consistency tests (e.g., number of games where M1/M2 outperforms baseline), rather than naive fold-level $p$-values.

---

## 6. Workstream E — Feature Feasibility Matrix

Fourteen candidate features were audited across scientific rationale, data availability, leakage risk, and implementation complexity. Full details are exported in [`reports/phase2_feature_feasibility.csv`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase2_feature_feasibility.csv).

| Feature Name | Category | Availability | Missingness | Leakage Risk | Feasibility Rating |
|---|---|:---:|:---:|:---:|:---:|
| `distance` | Spatial Baseline | 100.0% | 0.0% | None | **GO** |
| `shot_type` | Shot Characteristics | 100.0% | 0.0% | None | **GO** |
| `three_pt` | Shot Value | 100.0% | 0.0% | None | **GO** |
| `closest_def_dist` | Release Context | 100.0% | 0.0% | None | **GO** |
| `contest_level` | Release Context | 100.0% | 0.0% | Low | **GO** |
| `defender_closing_velocity` | Pre-Release Kinematics | 100.0% | 0.0% | None | **GO** |
| `delta_defender_dist` | Pre-Release Kinematics | 100.0% | 0.0% | None | **GO** |
| `contest_angle_cos` | Contest Geometry | 100.0% | 0.0% | None | **GO** |
| `shooter_velocity_magnitude` | Pre-Release Kinematics | 100.0% | 0.0% | None | **GO** |
| `nearby_defenders_count_6ft` | Crowding / Context | 100.0% | 0.0% | None | **CONDITIONAL GO** |
| `shot_clock_seconds` | Game State | 99.5% | 0.48% | None | **CONDITIONAL GO** |
| `teammate_spacing_dispersion` | Spatial Spacing | 100.0% | 0.0% | None | **CONDITIONAL GO** |
| `passing_lane_interception_risk` | Decision Audit | 0.0% | High | High | **NO-GO** |
| `counterfactual_decision_ev` | Decision Audit | 0.0% | High | High | **NO-GO** |

---

## 7. Workstream F — Experimental Plan (M0 vs M1 vs M2)

The next phase of research will compare three strictly nested model architectures:
1. **M0 (Baseline):** Shot location (`distance`) + basic characteristics (`shotType`, `three`, `catchAndShoot`, `createdFromPaint`, `period`).
2. **M1 (Release Context):** M0 + release-time defensive context (`closestDefDist`, `contestLevel`, `contested`).
3. **M2 (Pre-Release Motion & Geometry):** M1 + pre-release kinematics (`defender_closing_velocity`, `contest_angle_cos`, `shooter_velocity_magnitude`, `delta_defender_dist`).

### Evaluation Protocol Summary
- **Sample:** Exactly identical 1,322 eligible field-goal attempts across all 10 matches.
- **Cross-Validation:** Leave-One-Game-Out (LOGO) across the 10 games.
- **Primary Metric:** Out-of-fold Log Loss.
- **Secondary Metrics:** Brier Score, ROC-AUC.
- **Calibration Assessment:** Expected Calibration Error (ECE) across 10 uniform bins, backed by reliability curves and Hosmer-Lemeshow goodness-of-fit.
- **Ablation Structure:**
  - M2a: M1 + Defender Closing Velocity only.
  - M2b: M1 + Contest Angle only.
  - M2c: Full M2 (Closing Velocity + Contest Angle + Shooter Velocity).
- **Feature Retention Acceptance Criterion:**
  $$\Delta \text{Log Loss} \le -0.0050 \quad \text{AND} \quad \Delta \text{ROC-AUC} \ge 0.0 \quad \text{AND} \quad \text{ECE non-degraded}$$

---

## 8. Limitations and Unresolved Questions

1. **Small Match Sample ($N = 10$):** While 1,322 shots provide reasonable sample size for basic parameter fitting, 10 matches limit between-game variance representation. Future validation on external matches or an expanded ACB release will be essential.
2. **Planar Coordinates ($z = 0$):** Player tracking coordinates do not include vertical elevation (jump height, arm contest reach). Consequently, jump-shot release height and high-point contest blocks cannot be directly calculated from tracking alone.
3. **Extrapolated Detection Uncertainty:** While 91.3% of player positions at $t - 400$ ms are directly computer-vision detected, 8.7% are extrapolated. A sensitivity analysis comparing detected vs extrapolated rows must accompany M2 evaluation.
4. **Decision Optimality vs Shot Difficulty:** As established, predictive accuracy on shot outcomes does not validate coaching claims of "optimal shot selection." Any future public dissemination must strictly adhere to Target B framing (shot execution difficulty under dynamic pressure).

---

## 9. Deliverables Inventory

The following Phase 2 audit artifacts have been created and verified:
1. `reports/phase2_feasibility_audit.md` (this report)
2. `reports/phase2_data_inventory.csv` (field-level data inventory across all match, event, and aggregate tables)
3. `reports/phase2_feature_feasibility.csv` (candidate feature matrix with scientific rationales and feasibility ratings)
4. `reports/phase2_coverage_summary.json` (reproducible counts of frames, games, shots, and lookback coverage)
5. `reports/phase2_research_plan.md` (formal step-by-step experimental protocol for M0 vs M1 vs M2)
6. `src/courtiq/features/geometry.py` (validated vector kinematics and contest angle functions)
7. `tests/test_geometry.py` (32 passing unit tests covering vector mathematics, closing velocity signs, and contest angles)
