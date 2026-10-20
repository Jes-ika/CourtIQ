# CourtIQ Phase 3D — Documentation Correction & Research Freeze Log

**Author:** Lead Data Engineer & Statistical Code Auditor  
**Date:** October 20, 2026  
**Status:** Complete & Verified  
**Mandate:** Apply corrections identified in `reports/phase3c/final_audit_findings.md` while strictly preserving the frozen M1 benchmark, empirical metrics, and underlying model code.

---

## 1. Executive Summary

During the Phase 3C independent code and statistical audit (`reports/phase3c/final_audit_findings.md`), three documentation discrepancies were identified between the executable codebase and certain narrative reports:
1. **M1 Feature Definition Misattribution:** Previous narrative text colloquially attributed geometric and velocity features (`shot angle`, `contest angle`, `closing velocity`) to the M1 benchmark, whereas the executable code in `src/courtiq/evaluation/phase3a_cv.py` defines M1 strictly as 12 raw variables (5 numeric + 7 categorical).
2. **Ambiguous Feature-Dimension Header:** Benchmark tables reported an ambiguous `Features (d)` column that conflated raw input variables with one-hot encoded design-matrix dimensions ($d$). Furthermore, the fold-dependent variation in encoded columns ($d=29$ in 9 folds; $d=28$ in Fold 6 due to overtime period absence) was unstated.
3. **Sample Ingestion Accounting Clarification:** Event pipeline counts were inconsistently described across early iterations, obscuring the precise sequential filtering chain.

In Phase 3D, all three discrepancies have been remediated across `README.md`, `reports/phase3a/research_report.md`, and `reports/phase3b/final_research_report.md`. Zero model source code, test definitions, hyperparameters, or stored prediction/metric files were modified. All 49 automated tests pass with exit code 0.

---

## 2. Itemized Corrections

### Correction 1: M1 Feature Definition Reconciled to Executable Code
- **Finding:** The executable benchmark implementation in [`src/courtiq/evaluation/phase3a_cv.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/evaluation/phase3a_cv.py) (lines 24–39) and [`src/courtiq/features/pipeline.py`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/src/courtiq/features/pipeline.py) (lines 29–33) configures M1 with exactly 12 raw features:
  - **Numeric (5):** `distance`, `shotClock`, `releaseTime`, `dribblesBefore`, `closestDefDist`
  - **Categorical (7):** `shotType`, `three`, `catchAndShoot`, `createdFromPaint`, `period`, `contestLevel`, `contested`
- **Correction Applied:**
  - In `reports/phase3a/research_report.md` (Sections 4 and 8), removed erroneous narrative references attributing `shot angle`, `contest angle`, and `closing velocity` to M1.
  - In `reports/phase3b/final_research_report.md` (Section 5), explicitly itemized the 5 numeric and 7 categorical features comprising M1, clarifying that `defender_closing_velocity` and `contest_angle` were tested in exploratory Phase 2 ablations and are strictly excluded from M1.
  - In `README.md` (Section 1 and Section 4), added the comprehensive, verified 12-feature list for M1.

### Correction 2: Disambiguation of Feature Counts and Design Matrix Dimensions ($d$)
- **Finding:** Benchmark tables previously displayed an ambiguous `Features (d)` column showing scalar values (e.g., 6, 7, 8) that matched neither the raw variable count nor the encoded matrix dimensionality.
- **Empirical Pipeline Verification:**
  - Raw continuous inputs pass through `StandardScaler` (1 column each).
  - Categorical variables pass through `OneHotEncoder(handle_unknown="ignore")`.
  - Cardinality analysis of categorical features across the 1,322 eligible shots:
    - `shotType`: 5 categories (`floater`, `heave`, `jumper`, `layup`, `post`)
    - `three`: 2 categories (`False`, `True`)
    - `catchAndShoot`: 3 categories (`False`, `True`, `None`/missing)
    - `createdFromPaint`: 2 categories (`False`, `True`)
    - `period`: 5 categories across the dataset (`1`, `2`, `3`, `4`, `5`). However, `period == 5` (overtime) occurs strictly in Game `178442` (4 shots).
    - `contestLevel`: 5 categories (`open`, `light`, `average`, `plus`, `blocked`)
    - `contested`: 2 categories (`False`, `True`)
  - **One-Hot Encoding Expansion:**
    - Folds 1–5, 7–10 (9 training folds): Training set contains shots from periods 1–5 $\to 5 + 2 + 3 + 2 + 5 + 5 + 2 = 24$ dummy columns.
      - **M1 Design Matrix:** $5 \text{ numeric} + 24 \text{ OHE} = 29$ design columns ($d = 29$). With intercept: 30 parameters.
      - **M1 + Speed Rel:** $6 \text{ numeric} + 24 \text{ OHE} = 30$ design columns ($d = 30$). With intercept: 31 parameters.
      - **M1 + Directional:** $7 \text{ numeric} + 24 \text{ OHE} = 31$ design columns ($d = 31$). With intercept: 32 parameters.
    - Fold 6 (held-out Game `178442`): Training set contains only regulation periods 1–4 $\to 5 + 2 + 3 + 2 + 4 + 5 + 2 = 23$ dummy columns.
      - **M1 Design Matrix:** $5 \text{ numeric} + 23 \text{ OHE} = 28$ design columns ($d = 28$). With intercept: 29 parameters.
      - At test time on Game `178442`, the 4 overtime shots have their `period` one-hot vector encoded as all zeros by `OneHotEncoder(handle_unknown="ignore")`, preserving leak-free validity.
- **Correction Applied:**
  - Replaced ambiguous `Features (d)` columns across all benchmark tables with two distinct columns: `Raw Features` and `Design Columns (d)*`.
  - Appended explicit methodological footnotes detailing the categorical expansion and explaining the Fold 6 overtime dimension shift ($d=28$ vs $d=29$).

### Correction 3: Harmonization of Dataset Sample Accounting
- **Finding:** Earliest prototype notes casually referenced raw event counts or unlocated shots without documenting the exact sequential filter order.
- **Verified Sequential Pipeline Accounting:**
  $$\begin{aligned}
  \text{Total Raw Events Ingested} &= \mathbf{1,459} \\
  \text{Less: Unlocated / Missing Coordinates Events} &= -26 \quad (1.78\%) \\
  \text{Less: Fouled Misses (Non-Field-Goal Attempts)} &= -111 \quad (7.61\%) \\
  \hline
  \text{Retained Eligible Live-Play Field-Goal Attempts} &= \mathbf{1,322} \quad (100.0\%)
  \end{aligned}$$
  - Class balance: 597 makes ($45.16\%$), 725 misses ($54.84\%$).
- **Correction Applied:**
  - Synchronized ASCII architecture diagrams, data methodology sections, and summary tables in `README.md`, `reports/phase3a/research_report.md`, and `reports/phase3b/final_research_report.md` to display this verified four-stage progression.

---

## 4. Preserved Scientific Boundaries & Benchmark Governance

All updated documentation consistently reinforces the following scientific principles:
1. **M1 is the Official Benchmark:** M1 (Log Loss = `0.617270`, Brier = `0.215942`, ROC-AUC = `0.694100`) remains the frozen primary baseline.
2. **Shooter Speed is Exploratory:** Release-lookback shooter speed (`M1 + speed_rel`) demonstrates an exploratory improvement ($\Delta \text{LogLoss} = -0.000758$, 7/10 outer game wins) that is not statistically confirmed. The paired game-cluster bootstrap 95% confidence interval is strictly $[-0.002177, +0.000854]$, which spans zero and fails to reach the practical significance threshold ($0.0020$).
3. **Scope of Inference:** The benchmark evaluates 10 Spanish Liga ACB games (1,322 shots). Results do not claim generalization to NBA, EuroLeague, or NCAA contexts.
4. **No Counterfactual Claims:** The model predicts shot outcome at release frame ($startFrame$). It does not evaluate shot selection or perform counterfactual audits of whether a player should have passed or driven.
5. **Precise Calibration Accounting:** Nested cross-fitted Platt calibration consistently shrinks probabilities toward the prior ($a = 0.9110 \pm 0.0067$), improving M1 log loss ($-0.000144$) and reducing 10-bin ECE ($0.0511 \to 0.0446$), while uncalibrated predictions remain the primary benchmark.

---

## 5. Verification and Audit Artifacts

### Files Changed (Documentation Only)
- [`README.md`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/README.md): Corrected M1 feature description, updated sequential data flow diagram, updated benchmark table with `Raw Features` and `Design Columns (d)*`, and added fold variation footnote.
- [`reports/phase3a/research_report.md`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase3a/research_report.md): Corrected M1 feature lists in Sections 4 and 8; updated Table 3 header and footnote.
- [`reports/phase3b/final_research_report.md`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase3b/final_research_report.md): Corrected sample accounting in Section 4; updated M1 feature breakdown in Section 5; updated Table 2 header and footnote.
- [`reports/phase3d/documentation_correction_log.md`](file:///c:/Users/jesik/Desktop/hackathon/courtiq/reports/phase3d/documentation_correction_log.md): Created this comprehensive log.

### Files Preserved (Zero Modifications)
- **Source Code:** Zero modifications to any file in `src/` (`src/courtiq/evaluation/phase3a_cv.py`, `src/courtiq/models/baseline.py`, `src/courtiq/features/pipeline.py`, etc.).
- **Tests:** Zero modifications to any test in `tests/`.
- **Stored Predictions & Metrics:** All `.parquet`, `.csv`, and `.json` artifacts in `data/processed/`, `reports/phase3a/`, and `reports/phase3b/` remain frozen.

---

## 6. Automated Test Suite Execution

The existing automated test suite was executed without modification:

```powershell
.\.venv\Scripts\pytest -v
```

**Execution Results:**
- **Exit Code:** `0`
- **Test Summary:** `49 passed in 1.29s`
- **Test Suite Breakdown:**
  - `tests/test_alignment.py`: 4 passed
  - `tests/test_audit.py`: 3 passed
  - `tests/test_geometry.py`: 11 passed
  - `tests/test_kinematics_pipeline.py`: 8 passed
  - `tests/test_models_and_cv.py`: 4 passed
  - `tests/test_phase3a_calibration_and_features.py`: 4 passed
  - `tests/test_phase3b_audit.py`: 5 passed
  - `tests/test_pipeline.py`: 8 passed
  - `tests/test_schema.py`: 2 passed

---

## 7. Remaining Inconsistencies or Unverified Items

- **None.** Every numerical claim, sample tally, feature list, and dimensional property reported in the final documentation has been verified against the executable source code and reproducible test fixtures.

---

## 8. Final Verdict

# **READY FOR FINAL REVIEW**

*(The CourtIQ research documentation is fully synchronized with the verified codebase. The M1 benchmark is frozen, and scientific boundaries are rigorously preserved.)*
