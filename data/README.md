# Dataset Documentation

## Overview
This repository contains the dataset and initial planning documentation for the Banking Loan Approval dataset.

* **Source URL:** https://www.kaggle.com/datasets/parthangare/banking-loan-approval-dataset
* **License:** Refer to the dataset page (typically Data files © Original Authors or CC0)
* **Original Record Count:** 100,000 records

## Data Dictionary

### Target Meaning
**`LOAN_APPROVAL`**: A binary outcome indicating whether a bank loan was approved (1) or not approved (0).

### Input Fields & Units
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

## Limitations
* **Missing Temporal Context:** The dataset includes the day and month of contact, but lacks the specific year, making it difficult to account for broader macroeconomic trends over time.
* **Class Imbalance:** The target variable is somewhat imbalanced, with roughly 73% of loans approved and 27% not approved, which may affect the baseline accuracy and evaluation metrics of predictive models.
* **Ambiguous Currency:** The `balance` field lacks a specified currency unit (e.g., USD, EUR, INR).

## Next Steps

### Step 2 Plan
1. **Exploratory Data Analysis (EDA):** Visualize the distributions of numerical features (like `balance` and `duration`) and analyze how categorical variables (like `job` and `education`) correlate with `LOAN_APPROVAL`.
2. **Data Preprocessing:** Apply One-Hot Encoding to the categorical features and scale the numerical features using Standard or Min-Max scaling to prepare the data for modeling.
3. **Baseline Modeling:** Train a baseline classification model (such as Logistic Regression or a Random Forest) and evaluate it using metrics like Precision, Recall, and the F1-score to account for the slight class imbalance.
