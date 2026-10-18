# Step 2: Data Ingestion, Provenance, and Preparation

## 1. Data Provenance & Lineage
- **Source**: Banking Loan Approval Dataset (100,000 tabular records).
- **Collection Context**: Historical consumer loan applications recording demographic attributes, financial metrics, and approval/rejection outcomes.
- **Raw File Location**: `data/raw/` (or configured dataset ingestion path).
- **Target Variable**: `LOAN_APPROVAL` (Binary: `1` for Approved, `0` for Rejected).

## 2. Ingestion & Validation
- **Integrity Checks**: Verified 0 missing/null entries across all features and target columns prior to partitioning.
- **Schema Validation**: Identified continuous numeric variables and discrete categorical columns.

## 3. Train / Validation / Test Splitting Strategy
To strictly avoid data leakage, data was split deterministically using a fixed random state (`random_state=42`):
- **Train Set (`data/processed/train.csv`)**: Used exclusively to fit transformations and train model parameters.
- **Validation Set (`data/processed/validation.csv`)**: Used for threshold tuning, iteration tracking, and hyperparameter selection.
- **Test Set (`data/processed/test.csv`)**: Held out as a final unseen evaluation split.

## 4. Feature Preparation & Fuzzification
- Continuous variables (Age, Account Balance, Contact Duration) are prepared for zero-mean scaling (`StandardScaler`).
- Categorical features (Job, Marital Status, Education, Housing) are tracked for ordinal mapping (`OrdinalEncoder`).
- Membership representations capture non-linear boundary transitions without altering raw split independence.

## 5. Artifacts & Deliverables
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`