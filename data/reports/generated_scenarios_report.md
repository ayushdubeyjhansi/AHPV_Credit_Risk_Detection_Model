# Generated Test Scenarios Report: Credit Risk Stress-Testing & Robustness

## 1. Executive Summary & Purpose
This report documents the creation, characteristics, validation, and governance of synthetically generated stress-test scenarios for **Project SC09 (AHPV Credit Risk / Loan Approval Detection)**. 

Standard benchmark holdout sets assess model performance under observed historical distributions. However, regulatory standards in banking and algorithmic lending require stress-testing models against extreme edge cases, non-linear feature interactions, and potential demographic disparities before operational consideration. 

These 256 synthetic scenarios are designed exclusively to audit model boundary resilience and decision stability without contaminating the empirical evaluation splits.

---

## 2. Dataset Provenance & Generation Specifications

* **Baseline Empirical Dataset**: `banking_loan_approval_dataset_100k.csv` (100,000 empirical applicant records)
* **Target Feature**: `LOAN_APPROVAL` (`0` = Rejected, `1` = Approved)
* **Generated Scenarios Volume**: 256 synthetic applicant edge cases
* **Target Balance**: Exactly 128 instances per class (50% Class 0 / 50% Class 1)
* **Deterministic Random Seed**: `42`
* **Perturbation Technique**: 1% Interquartile Range (IQR) Gaussian jitter perturbation applied to continuous numerical attributes, combined with rule-based synthetic boundary edge cases.
* **Output File Location**: `data/processed/generated_loan_approval_test_scenarios.csv`
* **Artifact Dimensions**: 256 rows × 19 columns (16 applicant features + `LOAN_APPROVAL` + `Scenario_ID` + `Stress_Type`)

---

## 3. Dedicated Credit Risk Stress-Test Archetypes

The 256 scenarios incorporate four critical financial edge-case archetypes to evaluate classifier safety margins:

### Archetype 1: High Liquidity vs. Past Delinquency (`STRESS-LOAN-001`)
* **Profile Setup**: Applicant with high deposit reserves (`balance > €80,000`), stable employment (`job = management`), but an active credit delinquency flag (`default = 1`).
* **Evaluation Objective**: Audits whether the risk classifier treats an active default as a non-negotiable disqualifier or mistakenly overrides insolvency risk due to liquid deposit balances.

### Archetype 2: Multi-Leveraged Young Borrower (`STRESS-LOAN-002`)
* **Profile Setup**: Young applicant (`age = 21`, `job = services`) carrying concurrent debt lines (`housing_loan = 1`, `personal_loan = 1`) on thin financial margins (`balance < €200`).
* **Evaluation Objective**: Stresses debt-to-exposure thresholds at early career stages to ensure the classifier does not produce false approvals driven by aggregate demographic acceptance rates.

### Archetype 3: Outreach Saturation & Campaign Fatigue (`STRESS-LOAN-003`)
* **Profile Setup**: Extreme contact volume in current campaign (`campaign >= 25`, `duration < 45 seconds`) with zero prior touchpoints (`pdays = -1`, `previous = 0`).
* **Evaluation Objective**: Confirms the model penalizes severe outreach fatigue and disinterest proxies rather than treating frequent contact as positive customer engagement.

### Archetype 4: Fixed-Income Asset Preservation (`STRESS-LOAN-004`)
* **Profile Setup**: Senior applicant (`age >= 64`, `job = retired`) with no debt (`housing_loan = 0`, `personal_loan = 0`), moderate savings (`balance = €15,000`), and confirmed past repayment success (`poutcome = success`).
* **Evaluation Objective**: Audits fairness guardrails to verify that the absence of active earned employment wages does not result in discriminatory rejections for creditworthy senior applicants.

---

## 4. Class Balance & Schema Specification

The scenario generator enforces exact parity across lending outcomes to equally stress false-positive and false-negative decision margins:

| Decision Class | Target Value | Scenario Count | Proportion (%) | Primary Audit Focus |
| :--- | :--- | :--- | :--- | :--- |
| **Loan Rejected** | `0` | 128 | 50.0% | Risk under-estimation / Default leakage |
| **Loan Approved** | `1` | 128 | 50.0% | Unwarranted denial / Fair lending equity |
| **Total** | — | **256** | **100.0%** | Full Decision Boundary Audit |

### Schema Verification
* **Scenario_ID**: Deterministic formatted keys (`GEN-LOAN-001` through `GEN-LOAN-256`).
* **Stress_Type**: Explicit categorization tags (`High_Liquidity_Past_Default`, `Multi_Leveraged_Youth`, `Campaign_Fatigue`, `Asset_Rich_Senior`, `Boundary_Jitter`).
* **Data_Source**: Tagged explicitly as `Synthetically_Generated` to prevent downstream confusion with historical records.

---

## 5. Automated Validation Rules & Integrity Criteria

Before the scenario artifact is exported to `data/processed/`, it must satisfy four automated validation assertions:

1. **Shape Integrity**: Output dataframe must measure exactly 256 rows and 19 columns with all baseline feature names preserved.
2. **Missingness Invariance**: Exactly zero `NaN`, `null`, or infinite values across numerical and categorical features.
3. **Class Parity**: Exact 128:128 binary distribution of the `LOAN_APPROVAL` target.
4. **Physical & Statutory Bounds**:
   * Age bounded strictly within legal borrowing limits: $18 \le \text{age} \le 120$.
   * Non-negative constraints strictly enforced: $\text{duration} \ge 0$, $\text{campaign} \ge 1$, $\text{previous} \ge 0$.
   * Categorical features restricted to the validated domain vocabulary.

---

## 6. Strict Governance & Usage Boundaries

> ### **MANDATORY USAGE BOUNDARY**
> The 256 generated credit scenarios are reserved **strictly for downstream product stress-testing, boundary audit suites, adversarial evaluation, and robustness demonstrations**.
>
> Under no circumstances may generated scenarios be merged into or permitted to enter:
> - The primary training partition (`data/processed/train.csv`)
> - The validation tuning partition (`data/processed/validation.csv`)
> - The empirical holdout test partition (`data/processed/test.csv`)
>
> All primary model evaluation metrics (ROC-AUC, Precision, Recall, F1-Score) and official grading benchmarks must be computed exclusively on empirical data.