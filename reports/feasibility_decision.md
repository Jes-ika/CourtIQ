# CourtIQ Phase 1: Feasibility Decision & Research Governance

**Project:** CourtIQ: Pressure-Aware Shot Quality and Decision Audit  
**Phase:** Phase 1 Evaluation Gate  
**Research Question:** *Does adding defensive-context features improve held-out-game prediction of basketball field-goal outcomes beyond a baseline using shot location and basic shot characteristics?*

---

## 1. Executive Feasibility Decision

**Decision:** **FEASIBLE WITH DOCUMENTED SAMPLE LIMITATIONS (GO WITH GUARDRAILS)**

The dataset reconnaissance, schema verification, and pipeline tests confirm that the SkillCorner Spanish Liga ACB open dataset contains the required spatial, temporal, mechanical, and defensive-context variables needed to answer the primary research question.

A fully reproducible, leakage-free Phase 1 baseline experiment comparing an M0 spatial baseline against an M1 defensive-context model was executed under strict Leave-One-Group-Out (game-grouped) cross-validation. The results demonstrate a statistically significant improvement across all primary scoring rules. However, because the sample consists of 10 matches (~1,322 valid shot attempts), uncertainty intervals reflect finite sample cluster variance.

---

## 2. Evaluation Criteria Checklist

| Criterion | Requirement | Finding / Verification | Status |
|---|---|---|---|
| **Data Completeness** | Valid coordinates, shot outcomes, and pre-shot defense | 1,433 of 1,459 shots have complete pre-shot spatial tracking | **PASS** |
| **Target Label Integrity**| Definitive field-goal outcome without ambiguity | `outcome` (True/False) present for 100% of shots | **PASS** |
| **Leakage Isolation** | Strict separation of pre-release features from post-release events | Leaked fields (`shotQuality`, `fouled`, `blocked`, `assisted`) blacklisted and tested | **PASS** |
| **Tracking Alignment** | Deterministic alignment between shot release and tracking | `startFrame` == `frameIdx` verified; 0.0000 ft spatial distance error | **PASS** |
| **Group Separation** | Game-grouped held-out evaluation to avoid cross-game leakage | 10-fold Leave-One-Group-Out strictly excludes held-out game from training | **PASS** |
| **Sample Size Adequacy**| Sufficient power for statistical discrimination | 10 games, 1,322 valid shots; t-test p < 0.001 for deltas | **PASS** (with df=9 CI) |
| **Licensing Compliance**| Attribution and redistribution adherence | MIT license confirmed; raw data excluded from Git via `.gitignore` | **PASS** |

---

## 3. Data-Quality Audit & Exclusions Breakdown

Across the 10 published ACB games:
- **Total Raw Shot Events Ingested**: 1,459
- **Excluded: Missing Spatial Tracking / Coordinates**: 26 (1.78%)
  - *Rationale*: Without shooter coordinates and defender distance, spatial and defensive models cannot evaluate the shot.
- **Excluded: Fouled Misses (FIBA Box-Score Non-FGA)**: 111 (7.61%)
  - *Rationale*: Under FIBA basketball rules, when a shooter is fouled and misses, the shot does not count as a Field Goal Attempt in official statistics; free throws are awarded instead. Including fouled misses distorts the true probability of scoring a live-ball field goal. (Note: Fouled made shots / "and-ones" [32 shots] are retained as valid FGA).
- **Total Valid Observations in Baseline Experiment**: **1,322 shots** across 10 independent games.

---

## 4. Key Experimental Findings (Phase 1 Baseline)

Under 10-fold Leave-One-Group-Out CV (evaluating strictly on unseen games):

| Metric | M0 Spatial Baseline | M1 Defensive Context | Delta (M1 - M0) | 95% Confidence Interval (Student's t, df=9) | Statistically Significant? |
|---|---|---|---|---|---|
| **Log Loss** | 0.6552 | 0.6173 | **-0.0379** | [-0.0533, -0.0225] | **Yes** (p = 0.0003) |
| **Brier Score** | 0.2315 | 0.2159 | **-0.0155** | [-0.0220, -0.0090] | **Yes** (p = 0.0004) |
| **ROC-AUC** | 0.6410 | 0.6941 | **+0.0531** | [+0.0306, +0.0763] | **Yes** (p = 0.0005) |

**Conclusion on Research Question:**
Adding pre-shot defensive context (`closestDefDist` and `contestLevel`) provides a substantial and statistically robust improvement over pure shot location and basic mechanics. Held-out ROC-AUC increases by 5.31 percentage points, and Log Loss decreases by 0.0379. The entire 95% confidence interval for Log Loss reduction is strictly bounded below zero.

---

## 5. Methodological Governance: Facts, Assumptions, and Questions

### Observed Facts
1. The upstream dataset license is MIT License (Copyright 2026 SkillCorner).
2. The candidate open-data repository contains 10 sample games with full tracking and dynamic events, and 3 season-level aggregate CSVs for 293 games.
3. In raw tracking files delivered via Git, the `.jsonl.gz` files are Git LFS pointers unless pulled. However, the `{gameId}_dynamic_events.json` files contain pre-joined defender distance and contest levels computed directly from tracking at shot release.
4. The M1 model outperformed M0 across every single held-out fold in ROC-AUC.

### Assumptions
1. We assume that the 10 games provide an unbiased sampling of defensive pressure across Spanish ACB teams.
2. We assume that logistic regression with L2 regularization is an appropriate baseline architecture before advancing to gradient-boosted trees or spatial neural representations in Phase 2.

### Unresolved Questions & Next Steps for Phase 2
1. **Tracking Trajectory Dynamics**:
   Can velocity vectors (defender closing speed $\vec{v}_{def}$ and shooter lateral momentum $\vec{v}_{shoot}$) extracted directly from the 25fps raw tracking frames provide additional signal beyond static release distance?
2. **Offensive Aggregates Integration**:
   Can player-level baseline shooting talent from `data/aggregates/acb_shotsaggregates_20252026.csv` be incorporated as an empirical Bayes prior to adjust for shooter quality?
3. **Decision Audit Framework**:
   How can the delta between M0 (uncontested baseline expectation) and M1 (realized pressure expectation) be translated into a decision audit metric evaluating whether a player took a high-quality shot relative to alternative pass options?
