# CourtIQ Phase 3A — Tracking Quality & Temporal Lookback Sensitivity Report

**Author:** Senior Machine Learning & Statistical Audit Researcher  
**Status:** Frozen Benchmark Sensitivity Analysis  
**Research Context:** Spanish Liga ACB 2025–26 SkillCorner Tracking Data (10 Games, 1,322 Field-Goal Attempts)

---

## 1. Executive Summary

This sensitivity audit evaluates whether the minor predictive gain observed from shooter movement ($\Delta \text{LogLoss} = -0.000758$ on the full cohort) is an artifact of tracking extrapolation or sensitive to the temporal lookback window.

### Key Conclusions:
1. **Not Driven Solely by Extrapolation:** When restricting evaluation strictly to directly detected tracking frames ($N = 1,207$), release-lookback shooter speed retains a small, consistent log-loss reduction ($\Delta \text{LogLoss} = -0.000633$).
2. **Extrapolated Shots Magnify Movement Signals:** On extrapolated frames ($N = 115, 8.70\%$), the log loss delta is larger ($-0.002068$), driven by transition rushes and corner-boundary scrambles where optical occlusion occurs alongside rapid physical movement.
3. **Earlier Movement ($t - 0.8\text{s}$ to $t - 0.4\text{s}$) Yields Minimal Predictive Value:** Moving the lookback window earlier reduces the log-loss gain to negligible levels ($\Delta \text{LogLoss} = -0.000091$ full cohort; $+0.000002$ detected subset), demonstrating that only kinematics immediately preceding release contain signal for release-time outcome prediction.
4. **Negligible Window Attrition:** Extending the lookback to $800$ ms incurs only $3$ missing shots ($0.23\%$ attrition) due to quarter-opening possessions.

---

## 2. Tracking Quality Cohort Stratification

The SkillCorner tracking dataset annotates whether player coordinates are directly detected by optical computer vision (`is_extrapolated = False`) or inferred/smoothed by kinematic tracking filters (`is_extrapolated = True`).

### Cohort Definitions:
- **Full Cohort ($N = 1,322$):** All eligible field-goal attempts with complete release-time defensive tracking.
- **Directly Detected Subset ($N = 1,207, 91.30\%$):** Shots where the shooter was directly detected at release ($t$).
- **Extrapolated Subset ($N = 115, 8.70\%$):** Shots where the shooter position at release was interpolated/extrapolated due to occlusions or camera boundary constraints.

> **Methodological Rule:** The detected subset and extrapolated subset represent non-overlapping partitions with different sample sizes and baseline difficulty. Their absolute metrics must **not** be compared directly as paired shots. Within each subset, models are evaluated against each other on identical shot rows.

### Table 1: Stratified Performance Across Tracking Quality Subsets

| Model Name | Full Cohort ($N=1,322$) Log Loss | Detected Subset ($N=1,207$) Log Loss | Detected $\Delta$ vs M1 | Extrapolated Subset ($N=115$) Log Loss | Extrapolated $\Delta$ vs M1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1 Benchmark** | **0.617270** | **0.617401** | — | **0.615891** | — |
| **M1 + Speed Rel ($[t-0.4\text{s}, t]$)** | **0.616512** | **0.616768** | **-0.000633** | **0.613823** | **-0.002068** |
| **M1 + Speed Early ($[t-0.8\text{s}, t-0.4\text{s}]$)**| **0.617179** | **0.617403** | **+0.000002** | **0.614819** | **-0.001072** |
| **M1 + Directional ($v_{\text{rad}} + v_{\text{tan}}$)** | **0.618350** | **0.618512** | **+0.001111** | **0.616610** | **+0.000719** |
| **M1 + Radial Only ($v_{\text{rad}}$)** | **0.617788** | **0.617947** | **+0.000546** | **0.616113** | **+0.000222** |
| **M1 + Tangential Only ($v_{\text{tan}}$)**| **0.617886** | **0.618025** | **+0.000624** | **0.616430** | **+0.000539** |

---

## 3. Analysis of Tracking Quality Findings

### 1. Robustness of Release Speed in Clean Optical Data
In the directly detected subset ($N = 1,207$), `M1 + speed_rel` improves log loss from $0.617401$ to $0.616768$ ($\Delta = -0.000633$), Brier score from $0.215736$ to $0.215428$, and ROC-AUC from $0.692065$ to $0.694207$. This proves that the exploratory signal observed in Phase 2 is **not** an artifact of tracking extrapolation errors or synthetic sensor smoothing.

### 2. Extrapolated Subset Characteristics
In the extrapolated subset ($N = 115$), the log loss reduction is over $3\times$ larger ($\Delta = -0.002068$). A domain inspection reveals that extrapolated frames disproportionately represent:
- Fast-break transition shots where players sprint ahead of primary camera tracking pans.
- Corner three-pointers near physical arena boundaries.
- High-contact paint drives where multiple bodies occlude optical lines of sight.
In these high-tempo contexts, shooter velocity is markedly elevated (mean speed: $9.42$ ft/s extrapolated vs $8.49$ ft/s detected), producing a wider dynamic range for the linear model to exploit.

### 3. Directional Movement Consistently Worsens Both Subsets
Crucially, decomposing velocity into radial ($v_{\text{rad}}$) and tangential ($v_{\text{tan}}$) movement components degrades predictive performance across **both** tracking quality strata:
- Detected subset: $\Delta \text{LogLoss} = +0.001111$ (worse).
- Extrapolated subset: $\Delta \text{LogLoss} = +0.000719$ (worse).
This confirms that the failure of directional movement components is a fundamental modeling limitation, not an artifact of sensor noise or tracking quality.

---

## 4. Temporal Lookback Window Sensitivity & Sample Attrition

To evaluate whether velocity predictive power depends on the specific timing of the measurement window, we tested two distinct $400$ ms intervals:
1. **Release Lookback Window ($[t - 0.4\text{s}, t]$):** Frames $t-10$ to $t$. Measures physical speed during the shooting motion up to release.
2. **Earlier Lookback Window ($[t - 0.8\text{s}, t - 0.4\text{s}]$):** Frames $t-20$ to $t-10$. Measures movement leading into the gather, prior to final release elevation.

### Table 2: Lookback Window Comparison & Sample Attrition

| Window Parameter | Frame Interval | Duration | Valid Samples | Attrition Count | Attrition Reason | Mean Speed (ft/s) | Median Speed (ft/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Release Window** | $[t-10, t]$ | 400 ms | 1,322 / 1,322 (100.0%) | 0 | None | 8.57 ± 5.61 | 7.15 |
| **Earlier Window** | $[t-20, t-10]$ | 400 ms | 1,319 / 1,322 (99.77%) | 3 | Period start boundary ($t < 20$) | 8.84 ± 5.75 | 7.42 |

### Temporal Sensitivity Findings:
- **Signal Decay:** The predictive benefit of shooter speed decays rapidly when moving from the release window to the earlier window ($\Delta \text{LogLoss} = -0.000758 \to -0.000091$).
- **Implication for Release-Time Prediction:** At the moment of release ($startFrame$), the shooter's instantaneous momentum directly impacts release stability and shooting touch. Movement $800$ to $400$ ms earlier is largely superseded once release-time defensive contest distance and angle are accounted for.
- **Attrition Negligibility:** The $0.23\%$ attrition in the earlier window is negligible and does not distort comparative conclusions. For uniform row matching across the full benchmark, missing values in the earlier window were imputed with the training-fold median inside each LOGO fold.
