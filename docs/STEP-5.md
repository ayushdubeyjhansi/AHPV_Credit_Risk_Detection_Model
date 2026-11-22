# Step 5 - Product V1

## Team and project
- **Team name**: AHPV
- **Project code and title**: AHPV Credit Risk Detection Model (Loan Approval Prediction)
- **Branch and commit**: step-5-product-v1, 

## User task
Explain who will use the interface, which inputs they provide and which decision or output it supports:
- **Intended Users**: Bank loan officers, credit risk analysts, and underwriters reviewing consumer loan applications.
- **Provided Inputs**: Demographic indicators, financial metrics, and customer history attributes (e.g., applicant age, account balance, contact duration, campaign contacts, job sector, marital status, education level, and housing loan status).
- **Supported Decision / Output**: 
  - Produces an automated loan decision (`APPROVED` or `REJECTED / UNDERWRITER REVIEW`) based on a calibrated risk threshold ($\ge 0.75$).
  - Displays real-time risk meter gauges showing the exact Approval Probability (%) alongside the Default Exposure Risk (%) to determine whether the applicant qualifies for instant straight-through loan disbursement or requires manual underwriter escalation.

## Interface and method
- **Interface type**: Interactive script / UI decision dashboard (Streamlit / Python CLI runner)
- **Entry file**: `src/train_ann.py` (model training) / `src/app.py`
- **Saved model/controller/optimiser**: `models/ann_pipeline_model.joblib` (ColumnTransformer + MLPClassifier)
- **Input schema and units**:
  - `age`: Applicant age (Years, continuous integer)
  - `balance`: Average yearly bank balance (Currency units, continuous numeric)
  - `duration`: Last contact duration (Seconds, continuous numeric)
  - `campaign`: Number of contacts during this campaign (Count, discrete numeric)
  - `pdays`: Days passed since last contact from a previous campaign (Days, numeric)
  - `previous`: Number of contacts performed before this campaign (Count, numeric)
  - `job`: Employment sector (Categorical string)
  - `marital`: Marital status (Categorical string)
  - `education`: Education level attained (Categorical string)
  - `housing`: Has housing loan? (Binary / Categorical: yes/no)
- **Output labels, units and meaning**:
  - `LOAN_APPROVAL = 1`: Approved (Low credit risk, meets threshold policy)
  - `LOAN_APPROVAL = 0`: Rejected / Underwriter Review (High credit risk exposure)
  - `Approval Probability`: Percentage score ($0.0\% - 100.0\%$) from model sigmoid output
  - `Default Risk`: Complement percentage ($100.0\% - \text{Approval Probability}$)
- **Data flow**: input -> validation -> saved method -> explained output.
  - Input applicant features -> Validate types, non-null values, and ranges -> Pre-loaded pipeline forward pass (`predict_proba`) -> Render Loan Decision Card and Risk Meters (without re-running training).

## Run instructions
- **Python version**: Python 3.10+ / 3.13
- **Dependency installation command**:
  ```powershell
  pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Step 5 - Interactive Inference Architecture & Flow Plan

## 1. System Overview & Core Principle
The application serves real-time loan underwriting predictions using the pre-trained ANN pipeline (`models/ann_pipeline_model.joblib`)[cite: 1]. 

> **Architectural Constraint**: Training routines (`fit`, backpropagation, optimizer updates) are strictly excluded from the user interface. The prediction button only executes forward inference using cached artifacts in memory.

---
## 2. End-to-End System Flow
The interaction follows a strictly decoupled 4-stage pipeline where model weights are frozen and no training logic runs inside the UI loop:

1. **User Input Form**: Collects applicant attributes (e.g., age, balance, duration, job, housing).
2. **Validation Layer**: Checks for missing fields, invalid data types, and boundary anomalies before forwarding data.
3. **Inference Controller**: Feeds validated inputs through the pre-loaded `ann_pipeline_model.joblib` artifact and applies the business decision threshold ($\ge 0.75$).
4. **Readable Output**: Displays the final verdict (`APPROVED` vs. `REJECTED / UNDERWRITER REVIEW`) alongside dual risk meter gauges (Approval Probability vs. Default Risk Exposure).

---

## 3. Detailed Component Plan

### Stage 1: Input Form (User Interface)
Collects applicant demographic and financial variables required by the feature schema:
- **Numerical Inputs**: Age, Balance, Duration, Campaign, Previous Contact metrics.
- **Categorical Dropdowns**: Job Type, Marital Status, Education, Housing Loan status.
- **Action Elements**: "Run Risk Assessment" button and "Reset" button.

### Stage 2: Validation & Sanitization Layer
Validates data before reaching the model:
- **Null / Missing Value Checks**: Rejects submission if any required field is empty.
- **Boundary Verification**: Ensures numeric ranges make real-world sense (e.g., $Age \ge 18$, $Duration \ge 0$).
- **Categorical Integrity**: Restricts selections to valid categories known to the training set schema to prevent downstream pipeline crashes.

### Stage 3: Inference Controller & Pre-Loaded Pipeline
- **Lifecycle Management**: The controller loads `ann_pipeline_model.joblib` once at startup (using `@st.cache_resource` in Streamlit or global application memory in Flask/FastAPI)[cite: 1].
- **No Training at Runtime**: The button handler executes only `predict_proba()` and does not invoke `.fit()`.
- **Threshold Policy Enforcement**: Applies the business decision boundary ($\ge 0.75$ for instant automated approval) rather than arbitrary uncalibrated scores.

### Stage 4: Readable Output (Decision Card & Gauges)
Translates the raw float output into clear operational feedback conforming to `product-v1-sketch.png`:
- **Decision Status**: High-visibility badge (`APPROVED` in green or `REJECTED / UNDERWRITER REVIEW` in red/amber).
- **Approval Confidence Gauge**: Bar/meter depicting the approval probability percentage.
- **Default Risk Exposure**: Inverted risk gauge quantifying potential credit loss.
- **Recommended Next Step**: "Disburse Loan" vs. "Route Application to Manual Underwriting Audit".

---

## 4. Directory & File Placement Guide

| Component | Standard Path | Status / Fallback if Not Yet Built |
| :--- | :--- | :--- |
| **Model Artifact** | `models/ann_pipeline_model.joblib` | Generated by running `src/train_ann.py`[cite: 1]. Must exist before launching the UI. |
| **Inference App / UI** | `src/app.py` or `src/inference_ui.py` | Create here. Implement with Streamlit, Gradio, or lightweight HTML/Flask. |
| **Input Validation Helpers** | `src/validators.py` | Create here, or keep as helper functions inside the UI script if building a single-file prototype. |
| **UI Reference Mockup** | `docs/product-v1-sketch.png` | Saved in `docs/` from the previous step. |
| **Evaluation Evidence** | `results/step3_ann_metrics.csv` | Exported metrics validating baseline threshold behavior[cite: 1]. |

---

## 5. Fallback Implementation (If UI Framework Is Not Yet Installed)

If you have not set up a dedicated frontend framework yet, place this lightweight CLI/interactive test runner in `src/predict_sample.py` to satisfy the exact same flow:

```python
import joblib
import pandas as pd
from pathlib import Path

# 1. Pre-load pipeline outside the prediction loop
MODEL_PATH = Path("models/ann_pipeline_model.joblib")
if not MODEL_PATH.is_file():
    raise FileNotFoundError("Model artifact missing. Run src/train_ann.py first.")

pipeline = joblib.load(MODEL_PATH)
THRESHOLD = 0.75

def predict_applicant(input_dict: dict):
    # 2. Validation
    required_keys = ['age', 'balance', 'duration', 'job', 'marital', 'education', 'housing']
    for key in required_keys:
        if key not in input_dict or input_dict[key] is None:
            raise ValueError(f"Validation failed: missing field '{key}'")
            
    # 3. Predict (strictly forward pass)
    df_sample = pd.DataFrame([input_dict])
    prob_approved = pipeline.predict_proba(df_sample)[0, 1]
    
    # 4. Readable Output
    status = "APPROVED" if prob_approved >= THRESHOLD else "REJECTED"
    return {
        "Status": status,
        "Approval Probability": f"{prob_approved * 100:.2f}%",
        "Default Risk": f"{(1.0 - prob_approved) * 100:.2f}%",
        "Action": "Disburse Loan" if status == "APPROVED" else "Refer to Underwriter"
    }