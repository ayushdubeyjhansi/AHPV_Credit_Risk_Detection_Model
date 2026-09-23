
# Step 1 Project Contract

| Member              | Role                                       | Primary Responsibility                                                                                                                                                                           |
| ------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Ayush Dubey**     | **Group Leader & System Integration Lead** | Coordinate the team, design overall system architecture, integrate ANN + Fuzzy Logic + baseline, manage GitHub/repository, track milestones, and coordinate final testing/demo.                  |
| **Himanshu**        | **ANN / Machine Learning Lead**            | Develop the ANN risk classifier, perform preprocessing and normalization, implement training/testing, evaluate prediction performance, and generate ANN performance/convergence visualizations.  |
| **Pradhuman Dixit** | **Fuzzy Logic & Explainability Lead**      | Design fuzzy linguistic variables and membership functions, implement fuzzification/rules/aggregation/defuzzification, handle borderline cases, and create fuzzy-rule/membership visualizations. |
| **Vinit Sharama**   | **Data, Evaluation & UI/Deployment Lead**  | Prepare and validate the dataset/scenarios, implement baseline score-threshold method, perform comparison/evaluation, develop the Streamlit UI, and handle deployment and result visualization.  |

## One-sentence problem
Given anonymous credit-related features such as income, debt, and repayment history, estimate the applicant’s credit risk, confidence level, and recommended action (review, approve, or reject) using an explainable decision-support system.

## User of the product
An educational risk reviewer who uses an explainable decision-support prototype to assess credit risk, understand model reasoning, and review recommended actions.

## Inputs and units
### Dataset Overview

| Column | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `age` | Integer | Age of the customer | `18 to 74` |
| `job` | Categorical | Type of employment | `retired, technician, admin, services, management` |
| `marital` | Categorical | Marital status | `single`, `married`, `divorced` |
| `education` | Categorical | Highest education level reached | `primary`, `secondary`, `tertiary`|
| `default` | Binary / Boolean | Credit default history | `1`, `0` |
| `balance` | Numeric | Average yearly balance in account | `-2000 t0 150k` |
| `housing_loan` | Binary / Boolean | Has an active housing loan | `1`, `0` |
| `personal_loan` | Binary / Boolean | Has an active personal loan | `1`, `0` |
| `contact` | Categorical | Communication channel used | `cellular`, `telephone` |
| `day` | Integer | Last contact day of the month | `1` to `31` |
| `month` | Categorical | Last contact month of the year | `jan`, `may`, `nov` |
| `duration` | Integer | Last contact duration (seconds) | `120`, `340` |
| `campaign` | Integer | Number of contacts during this campaign | `1`, `3`, `6` |
| `pdays` | Integer | Days passed after previous campaign contact (`-1` = not contacted) | `-1`, `99`, `180` |
| `previous` | Integer | Number of contacts performed before this campaign | `0`, `2`, `5` |
| `poutcome` | Categorical | Outcome of previous marketing campaign | `success`, `failure`, `other`, `unknown` |
## Outputs and units
###database overview

| Column | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| `LOAN_APPROVAL` | Binary  | Target label indicating final loan approval status | `1`, `0` |
## Baseline method
Write the numbered baseline rules from this guide.
## Soft Computing method for M1
An ANN risk classifier, compared with a visible score-threshold baseline.
## Advanced method for M2
Use fuzzy logic for borderline cases, check calibration and fairness, and deploy the app.
## Dataset/scenario sources
Add real URLs, licences, and assumptions.
## Five mandatory test cases
Copy and complete the five cases in this guide.
## Product V1 screen sketch
Insert a photo or exported image of the sketch.
## Risks and assumptions
List what may be wrong, missing, simulated, or temporary.
## Step 1 completion evidence
Add links to files, commits, and validation output.
