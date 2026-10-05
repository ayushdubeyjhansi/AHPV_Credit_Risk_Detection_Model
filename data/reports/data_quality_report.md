# Data Quality & Preparation Report: SC09 Banking Loan Approval

## 1. Project Purpose & Scope
The objective of project **SC09 (AHPV Credit Risk / Loan Approval Detection)** is to build an automated machine learning pipeline that evaluates applicant profiles and predicts loan approval decisions (`LOAN_APPROVAL`).

---

## 2. Dataset Provenance & Metadata
* **Project Reference**: SC09 - AHPV Credit Risk Detection
* **Dataset Identifier**: `banking_loan_approval_dataset_100k.csv`
* **Source Repository / URL**: [ Kaggle ](https://www.kaggle.com/datasets/parthangare/banking-loan-approval-dataset)
* **Licence**: all data belong to its owners
* **Original Record Dimensions**: 100,000 rows ×  17 columns
* **Target Feature**: `LOAN_APPROVAL`

---

## 3. Raw Inspection & Missing Values (Before Cleaning)

*Fill in the missing counts from your Cell 3 output (`missing_before`):*

| Input Field | Description | Units / Format |
| :--- | :--- | :--- |
| `age` | Age of the loan applicant | Years (Integer) |
| `job` | Occupation or job type of the applicant | Categorical |
| `marital` | Marital status | Categorical |
| `education` | Highest level of education | Categorical |
| `default` | Indicates if the applicant has credit in default | Binary (0 = No, 1 = Yes) |
| `balance` | Average yearly account balance | Currency (Integer) |
| `housing_loan` | Indicates if the applicant has an existing housing loan | Binary (0 = No, 1 = Yes) |
| `personal_loan` | Indicates if the applicant has an existing personal loan | Binary (0 = No, 1 = Yes) |
| `contact` | Type of communication used to contact the applicant | Categorical |
| `day` | Last contact day of the month | Day (Integer) |
| `month` | Last contact month of the year | Month (Categorical) |
| `duration` | Duration of the last contact | Seconds (Integer) |
| `campaign` | Number of contacts performed during this campaign | Count (Integer) |
| `pdays` | Days passed since the client was last contacted from a previous campaign | Days (Integer) |
| `previous` | Number of contacts performed before this campaign | Count (Integer) |
| `poutcome` | Outcome of the previous marketing campaign | Categorical |

---

## 4. Target Class Distribution

| Class Label | Meaning | Count | Proportion (%) |
| :--- | :--- | :--- | :--- |
| `0` | Rejected | 27,058 | 27.06% |
| `1` | Approved | 72,942 | 72.94% |
| **Total** | | **100,000** | **100.00%** |
---

## 5. Cleaning Summary

*Fill in from your Cell 5 output (`DATA CLEANING SUMMARY`):*

* **Rows before cleaning**: 100,000
* **Duplicate rows detected & removed**: 0
* **Rows after cleaning**: 100,000
* **Remaining missing values**: 0
* **Imputation rule**: Numeric feature medians (target excluded from imputation)
---

## 6. Dataset Splits Summary

Splits were generated using a fixed random seed of `42` with class-wise stratification:

| Split Partition | Expected Ratio | Row Count | Class 0 (Rejected) | Class 1 (Approved) |
| :--- | :--- | :--- | :--- | :--- |
| **Full Cleaned** | 100% | 100,000 | 27,058 (27.06%) | 72,942 (72.94%) |
| **Training** | 70% | 69,999 | 18,940 (27.06%) | 51,059 (72.94%) |
| **Validation** | 15% | 14,999 | 4,058 (27.05%) | 10,941 (72.95%) |
| **Test** | 15% | 15,002 | 4,060 (27.06%) | 10,942 (72.94%) |

*All files saved to `data/processed/` (`cleaned_data.csv`, `train.csv`, `validation.csv`, `test.csv`).*
---
---

## 7. Data Validation & Integrity Rules

Prior to data partitioning, the prepared dataset was audited against programmatic business rules. All checks executed with zero errors, confirming the data is safe for model training:

* **Schema Completeness**: All 16 explanatory features and the target column (`LOAN_APPROVAL`) are present.
* **Missing Value Check**: Confirmed 0 null/NaN occurrences across all fields post-imputation.
* **Domain Range Constraints**: Applicant ages are verified within valid adult lending bounds ($18 \le \text{age} \le 120$).
* **Sign Constraint Checks**: Variables requiring non-negative quantities (`age`, `duration`, `campaign`, `previous`) contain 0 negative entries.
* **Target Label Validity**: Target strictly contains valid binary decision labels (`0` and `1`).
* **Deduplication Check**: Confirmed 0 duplicate observations.
* **Validation Outcome**: `VALIDATION PASSED` (Cleaned data approved for splitting).

---

## 8. Partitioning Strategy, Reproducibility & Downstream Usage

A class-wise stratified sampling routine was executed with a single fixed seed to prevent class skew and eliminate data leakage.

* **Random Seed**: `42`
* **Partition Ratios**: 70% Train / 15% Validation / 15% Test

| Split Name | Output File Path | Row Count | Target Distribution (`0` / `1`) | Downstream Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Cleaned Full** | `data/processed/cleaned_data.csv` | 100,000 | 27,058 (27.06%) / 72,942 (72.94%) | Complete preprocessed benchmark dataset for baseline reference and overall distribution audits. |
| **Training** | `data/processed/train.csv` | 69,999 | 18,940 (27.06%) / 51,059 (72.94%) | Primary subset for fitting machine learning models and optimizing classification parameters. |
| **Validation** | `data/processed/validation.csv` | 14,999 | 4,058 (27.05%) / 10,941 (72.95%) | Model evaluation during hyperparameter tuning, feature selection, and early-stopping thresholding. |
| **Test** | `data/processed/test.csv` | 15,002 | 4,060 (27.06%) / 10,942 (72.94%) | Unseen holdout partition reserved strictly for final generalization scoring and reporting test metrics. |
---

## 9. Data Limitations & Honest Scope Statement

### Empirical Provenance vs. Synthetic Scenarios
* **Baseline Empirical Observation Boundaries**: The dataset contains 100,000 tabular records representing historical applicant profiles and interaction outcomes. These records reflect fixed operational conditions captured during the data collection window.
* **Separation of Synthetic Stress-Test Scenarios**: Any subsequent counterfactual cases, adversarial inputs, edge cases, or stress-test scenarios introduced during later modeling stages must be explicitly tagged and stored as **generated/synthetic scenarios**. They must never be conflated with or represented as true historical banking observations.
* **Non-Generalizability to Shifting Macroeconomic Conditions**: The patterns encoded in these observations do not capture unmeasured external economic shocks (e.g., rapid interest rate hikes, regulatory statutory changes, or sudden credit contraction events).

---

## 10. Project Safety & Fair Lending Note

* **Advisory Decision Support Only**: This model pipeline is developed strictly as a decision-support and risk-scoring tool. It must never function as a fully autonomous rejection engine without human-in-the-loop review by a qualified credit risk officer.
* **Fair Lending & Demographic Guardrails**: Features such as `age` and `marital` status must be audited for adverse impact and disparate misclassification rates to ensure lending fairness and compliance with statutory non-discrimination standards before any model artifact is considered for deployment.
