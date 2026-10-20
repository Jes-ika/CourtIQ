# CourtIQ Phase 3B — Submission Readiness Checklist & Final Recommendation

**Auditor / Lead Evaluator:** Independent ML Audit & Verification Specialist  
**Evaluation Target:** CourtIQ Basketball Shot Quality Benchmark  
**Date:** October 2026  
**Decision Governance:** AQX Sports Analytics Data Bowl 4.0 / Academic Research Submission  

---

## 1. Submission Readiness Checklist

### Part A: Software Engineering & Data Hygiene
- [x] **Repository Cleanliness:** No raw tracking JSONL/JSON data files checked into Git. `.gitignore` strictly blocks all match payloads.
- [x] **Zero Code Duplication / Leakage:** Shared logic properly compartmentalized under `src/courtiq/`.
- [x] **Dependency Specification:** Pinned `requirements.txt` and standard `pyproject.toml` support Python 3.10 to 3.12.
- [x] **Automated Test Coverage:** 49 test cases passing in under 2 seconds across 10 distinct test modules.

### Part B: Methodological & Statistical Defensibility
- [x] **Outer Fold Quarantining:** Leave-One-Game-Out cross-validation across all 10 independent matches. Zero test-game instances in outer training splits.
- [x] **Leakage-Safe Nested Calibration:** Platt scaling fit strictly on inner cross-fitted training predictions (9-fold inner LOGO). No resubstitution calibration leakage.
- [x] **Preprocessing Anti-Leakage:** Imputation and scaling encapsulated in scikit-learn `Pipeline`; outer test data only transformed, never fit.
- [x] **Cluster-Aware Bootstrap Resampling:** Resampling intact game clusters ($B = 2,000$) rather than treating shots as independent rows.
- [x] **Exact Metric Reproducibility:** Out-of-fold predictions reproduce reported log loss, Brier score, ROC-AUC, and ECE to 6 decimal places.
- [x] **Honest Statistical Reporting:** Zero unsupported significance claims. Release speed improvement acknowledged as exploratory; directional decomposition confirmed as detrimental.

### Part C: Research Scope & Scientific Claims
- [x] **Release-Time Boundary:** Evaluation timestamp strictly locked to shot release (`startFrame`).
- [x] **No Decision Claims:** Explicitly documented that model predicts shot conversion under pressure, not counterfactual shot selection optimality.
- [x] **No Causal Inferences:** Documented as observational conditional likelihood estimation.
- [x] **Tracking Quality Stratified:** Evaluated on directly detected ($N = 1,207$) vs. extrapolated ($N = 115$) tracking frames.

---

## 2. Three Distinct Validation Standards (Explicit Breakdown)

In academic and competitive ML evaluation, scientific credibility requires separating three fundamentally different standards:

### Standard 1: Code Reproducibility
- **Status:** **VERIFIED / GO**
- **Basis:** Any independent researcher can clone this repository, run `pip install -e .`, execute `python scripts/run_phase3a.py`, and obtain the exact logged metrics, plots, and prediction tables to machine precision. All 49 unit and integration tests execute cleanly.

### Standard 2: Evaluation Methodology Defensibility
- **Status:** **DEFENSIBLE / GO**
- **Basis:** The double-nested Leave-One-Game-Out cross-validation, in-fold inner cross-fitting Platt scaling, cluster-level bootstrap uncertainty quantification, and pre-specified evaluation thresholds withstand the highest standard of peer review. Data leakage, resubstitution bias, and pseudo-replication have been completely audited and prevented.

### Standard 3: External Findings Validation
- **Status:** **NOT EXTERNALLY VALIDATED / NO-GO FOR GENERALIZABILITY CLAIMS**
- **Basis:** The empirical findings are based exclusively on a 10-game sample (1,322 shots) from the Spanish Liga ACB. No genuinely independent matches or external basketball leagues (e.g., NBA, EuroLeague, NCAA) have been evaluated. While internal held-out validity across the 10 games is sound, external cross-league validity remains unverified and must not be claimed.

---

## 3. Final Submission Recommendation

### Executive Verdict: **CONDITIONAL GO (READY FOR SUBMISSION)**

**Conditions Met for Approval:**
1. **Benchmark Frozen at M1:** M1 is retained as the official benchmark.
2. **Kinematics Retained as Exploratory:** Shooter velocity is explicitly presented as an unconfirmed hypothesis.
3. **Directional Components Excluded:** Directional velocity is formally excluded from production specifications.
4. **Transparent Limitation Declaration:** The 10-match sample constraint and lack of external league validation are clearly highlighted in the abstract, limitations, and portfolio documentation.
