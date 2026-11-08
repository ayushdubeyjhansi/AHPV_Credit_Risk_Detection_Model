# Step 5 – Product V1 Report

## 1. Application Entry Point

**Entry file:** `app.py`

**Application type:** Streamlit web application

**Launch command:**
```text
python -m streamlit run app.py
```

## 2. Product Description

The Product V1 is a Credit Risk Loan Approval Prediction application. It accepts applicant financial and demographic attributes and uses the saved Step 3 ANN model to generate a loan approval prediction and decision-support information.

## 3. Input Features

The application uses the following 16 input features:

- age
- job
- marital
- education
- default
- balance
- housing_loan
- personal_loan
- contact
- day
- month
- duration
- campaign
- pdays
- previous
- poutcome

## 4. Output

The application provides:

- Status
- Predicted class
- Loan label
- Approval probability
- Default risk
- Policy threshold
- Recommended action

## 5. Model and Inference Path

**Inference module:**
`src/product_predict.py`

**Saved model:**
`models/ann_pipeline_model.joblib`

The application sends the entered applicant data through the validation and inference pipeline and displays the resulting prediction.

## 6. Automated Testing

Five automated tests were implemented using Python `unittest`.

Tests cover:

1. Valid input
2. Missing required numeric feature
3. Invalid age
4. Negative duration
5. Prediction output

**Result:** 5 tests passed successfully (`OK`).

## 7. Manual Validation

The following manual validation cases were tested:

| Test | Condition | Expected Result |
|---|---|---|
| 1 | Age = 18 | Accepted |
| 2 | Age = 120 | Accepted |
| 3 | Age = 17 | Rejected |
| 4 | Duration = -1 | Rejected |
| 5 | Age missing | Rejected |

Screenshots for these tests are stored in:

`reports/step5_evidence/`

## 8. Working Product Evidence

The working application successfully accepts applicant information and produces a loan decision, approval probability, default risk and recommended action.

Example successful prediction:

- Model Class: 1
- Decision: Loan Approved
- Approval Confidence: 96.25%
- Default Risk: 3.75%
- Policy Threshold: >= 75.0%
- Recommended Action: Disburse Loan

## 9. CSV Verification

A reference prediction CSV was created at:

`reports/step5_ui_reference_prediction.csv`

The reference prediction records the applicant input and corresponding model output from the working Product V1 interface.

## 10. Evidence Files

The following screenshots were collected:

- `01_working_product_prediction.png`
- `02_age_18.png`
- `03_age_120.png`
- `04_age_17_error.png`
- `05_negative_duration_error.png`
- `06_missing_age_error.png`

## 11. Limitations

- The system is a decision-support demonstration and does not replace formal credit underwriting.
- Predictions depend on the quality and distribution of the training data.
- Model probabilities should not be interpreted as guaranteed loan approval.
- Formal regulatory and business checks are still required before a real lending decision.
- The current validation layer checks required fields and selected numeric boundaries.

## 12. Dependencies

**Python version:** Python 3.14.8

**Installed dependencies:**

```text
altair==6.3.0
anyio==4.16.0
asttokens==3.0.2
attrs==26.1.0
certifi==2026.10.23
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
colorama==0.4.6
comm==0.2.3
contourpy==1.4.0
cycler==0.12.1
debugpy==1.8.22
executing==2.3.0
fonttools==4.67.0
h11==0.16.0
idna==3.20
ipykernel==7.4.0
ipython==9.17.1
ipython_pygments_lexers==1.1.1
itsdangerous==2.2.0
jedi==0.20.1
Jinja2==3.1.6
joblib==1.6.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter_client==8.10.0
jupyter_core==5.9.1
kiwisolver==1.5.1
MarkupSafe==3.0.4
matplotlib==3.11.2
matplotlib-inline==0.2.2
narwhals==2.27.1
nest-asyncio2==1.7.4
numpy==2.5.4
packaging==26.3
pandas==3.0.6
parso==0.8.7
pillow==13.0.0
platformdirs==4.13.3
prompt_toolkit==3.0.53
protobuf==7.36.2
psutil==7.2.2
pure_eval==0.2.4
pyarrow==26.0.0
pydeck==0.9.3
Pygments==2.21.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
python-multipart==0.0.33
pyzmq==27.2.0
referencing==0.37.0
requests==2.34.2
rpds-py==2026.9.1
scikit-learn==1.9.1
scipy==1.18.1
seaborn==0.13.2
six==1.17.0
stack-data==0.6.3
starlette==1.10.0
streamlit==1.66.1
threadpoolctl==3.7.0
tornado==6.5.10
traitlets==5.16.1
typing_extensions==4.16.0
tzdata==2026.5
urllib3==2.8.0
uvicorn==0.54.0
watchdog==6.0.0
wcwidth==0.9.2
websockets==17.4
'''

## 13. Team Contributions

Document the actual contribution of each team member here.

Example:

- Member 1: Model development and training
- Member 2: Data preparation and preprocessing
- Member 3: Product V1 interface and integration
- Member 4: Testing and documentation
![alt text](<Screenshot 2026-11-07 233621.png>)