# Step 6 - Testing and validation

## Project and environment
- **Project**: SC09 AHPV Credit Risk Detection Model (Loan Approval Prediction)
- **Branch**: `step-6-testing`[cite: 19]
- **Commit tested**: `[paste output of: git rev-parse --short HEAD]`[cite: 23]
- **Python version**: `[paste output of: python --version]` (project `.venv`)[cite: 23]
- **Launch from the repository root**: `python -m streamlit run app.py`[cite: 23]
- **Reference input**: first data row in `data/processed/validation.csv`[cite: 23]

## Planned tests and results
Keep the other 15 fields fixed when testing a single input. `PASS` means the application or script behaved as expected (including rejecting invalid inputs with clear validation errors and withholding predictions)[cite: 45].

| ID | Input or action | Expected behaviour | Actual result | Status |
| :--- | :--- | :--- | :--- | :--- |
| T01 | Load existing sample and predict | Display a predicted class, readable label, approval probability, default risk, policy action, and credit-risk disclaimer notice. Allow CSV download. | Displayed predicted class, decision label, approval probability, default risk, recommended action, and compliance notice. Exported result CSV (`data/reports/step6_ui_prediction.csv`). | PASS |
| T02 | Reload sample and set `age` to `18` | Accept the lower age boundary (`18`) and display a prediction. | Accepted `age = 18` and returned a valid loan decision and probability output. | PASS |
| T03 | Reload sample and set `age` to `120` | Accept the upper age boundary (`120`) and display a prediction. | Accepted `age = 120` and returned a valid loan decision and probability output. | PASS |
| T04 | Reload sample and set `age` to `15` | Show the allowed age range message (`18 to 120`) and withhold the result. | Displayed `Invalid age '15.0'. Age must be between 18 and 120.` and withheld the prediction. | PASS |
| T05 | Reload sample and enter `abc` in `balance` | Ask for a valid numeric value and withhold the result. | Displayed `Feature 'balance' must be a valid number, got: abc` and withheld the prediction. | PASS |
| T06 | Reload sample and clear `job` (or `balance`) | Name the missing or empty field and withhold the result. | Displayed missing/empty field validation error (`Feature 'job' cannot be empty.`) and withheld the prediction. | PASS |
| T07 | Start a disposable copy on port `8502` with `models/ann_pipeline_model.joblib` renamed | Explain that the saved model file is missing and stop before rendering the input form. | Browser on `http://localhost:8502` displayed the missing-model error message and stopped before showing the prediction form. | PASS |
| T08 | Restore the copied model filename and relaunch on port `8502` | The form returns and the normal validation sample predicts again. | Restored form rendered on `http://localhost:8502` and returned the expected prediction for the validation sample. | PASS |
| T09 | Run `python src/compare_step6_predictions.py` to compare browser and Python CSVs | All 16 SC09 features, approval probabilities, predicted class, and label match (`PASS`). | Script printed `PASS: All 16 SC09 features, predicted class, and label match.` | PASS |
| T10 | Run `python -m unittest -v test_product_predict` | All 6 input validation unit tests run and finish with `OK`. | All 6 validation unit tests (`test_normal`, `test_age_boundaries`, `test_out_of_range_age`, `test_invalid_numeric_text`, `test_negative_duration`, `test_missing_feature`) passed (`Ran 6 tests ... OK`). | PASS |

## Evidence
| Test IDs | File path or exact observed output |
| :--- | :--- |
| T01 | Browser download saved as `data/reports/step6_ui_prediction.csv`. |
| T02–T03 | Browser UI accepted exact boundary values `age = 18` and `age = 120` while keeping the remaining 15 features fixed, returning valid predictions in both runs. |
| T04–T06 | Browser UI displayed specific `ValueError` validation alerts (`Invalid age '15.0'...`, `Feature 'balance' must be a valid number...`, `Feature 'job' cannot be empty.`) and withheld model output. |
| T07–T08 | Isolated temporary copy on `http://localhost:8502` displayed the missing-model error when `ann_pipeline_model.joblib` was renamed, and recovered normal inference once restored. |
| T09 | `src/compare_step6_predictions.py` compared `data/reports/step6_ui_prediction.csv` against `data/reports/step5_ui_reference_prediction.csv` (generated via `src/check_product_prediction.py`) and reported `PASS`. |
| T10 | `test_product_predict.py`: 6 named validation tests executed via `python -m unittest -v test_product_predict` showed `ok` and finished with `OK`. |

## Failures, fixes and retests
- **Initial CSV comparison mismatch (T09)**: The first comparison run failed (`FAIL: Differences in age, balance...`) because the UI form was submitted with default fallback values (`age = 35`, `balance = 4500`) rather than clicking **Load one existing example** (`data/processed/validation.csv`, row 0). Reloading the validation sample in the UI, re-exporting `data/reports/step6_ui_prediction.csv`, and re-running `python src/compare_step6_predictions.py` produced a clean `PASS`.
- **Module export alignment**: Exported `FEATURES`, `NUMERIC_FEATURES`, and `CATEGORICAL_FEATURES` at the top level of `src/product_predict.py` so `app.py`, `src/check_product_prediction.py`, and `src/compare_step6_predictions.py` share a single source of truth for the 16-feature schema.
- **Intentional missing-model check (T07–T08)**: Executed in an isolated temporary directory on port `8502` so the primary repository artifacts remained untouched; restoring the model filename confirmed clean recovery.

## Known limits
- The CSV consistency check (`T09`) verifies alignment between the Streamlit UI and the Python inference script on the first validation row (`data/processed/validation.csv`, row 0); batch consistency across all 15,000 validation records was not tested through the browser UI.
- Automated unit tests (`T10`) validate data types, required field presence, and numeric boundaries (`age` 18–120, non-negative `duration`/`campaign`/`previous`), but do not enforce strict closed-set whitelists for unseen categorical strings (handled instead via `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`).
- Tested locally on Windows with Python `.venv` and Streamlit (`localhost:8501` / `localhost:8502`); multi-user concurrent load and cloud deployment environments were not tested.
- This application provides decision-support credit risk scores only and does not replace formal regulatory underwriting or manual compliance audits.

## Contributions
- **Ayush Dubey (`24SCSE1410094`)**: Configured the `step-6-testing` branch, implemented `src/check_product_prediction.py`, `src/compare_step6_predictions.py`, and `test_product_predict.py`, executed all UI boundary/error/recovery tests (`T01`–`T08`), verified CLI/UI parity (`T09`) and unit tests (`T10`), and documented the results in `docs/STEP-6.md`.