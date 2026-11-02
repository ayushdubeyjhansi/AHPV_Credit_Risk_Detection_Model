from pathlib import Path
import pandas as pd
import streamlit as st
from src.product_predict import FEATURES, load_model, predict_one

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="SC09 Credit Risk - Product V1", layout="centered")
st.title("Credit Risk Loan Approval Prediction: Product V1")
st.write("Enter the applicant financial and demographic attributes used by our saved Step 3 ANN.")
st.warning("Decision support demonstration only. Predictions do not replace manual underwriting or regulatory credit compliance checks.")

# Field labels and units matching the SC09 Banking Loan Approval schema
LABELS = {
    "age": "Age (years, 18-120)",
    "job": "Job Sector (e.g., management, technician, blue-collar, admin., services)",
    "marital": "Marital Status (married, single, divorced)",
    "education": "Education Level (primary, secondary, tertiary, unknown)",
    "default": "Has Credit in Default? (yes, no)",
    "balance": "Average Yearly Balance (currency units)",
    "housing_loan": "Has Housing Loan? (yes, no)",
    "personal_loan": "Has Personal Loan? (yes, no)",
    "contact": "Contact Communication Type (cellular, telephone, unknown)",
    "day": "Last Contact Day of the Month (1-31)",
    "month": "Last Contact Month (jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec)",
    "duration": "Last Contact Duration (seconds)",
    "campaign": "Number of Contacts Performed During Campaign",
    "pdays": "Days Passed Since Last Contact (-1 means not contacted)",
    "previous": "Number of Contacts Performed Before Campaign",
    "poutcome": "Outcome of Previous Marketing Campaign (success, failure, other, unknown)",
}

# Default initial values for input fields
DEFAULT_VALUES = {
    "age": "35",
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": "4500",
    "housing_loan": "no",
    "personal_loan": "no",
    "contact": "cellular",
    "day": "15",
    "month": "may",
    "duration": "250",
    "campaign": "1",
    "pdays": "-1",
    "previous": "0",
    "poutcome": "unknown",
}


@st.cache_resource
def saved_model():
    return load_model()


try:
    model = saved_model()
except Exception as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.info("Ensure you ran 'python src/train_ann.py' and your virtual environment is active.")
    st.stop()

# Helper button to load an actual applicant row from processed validation data
if st.button("Load one existing example"):
    try:
        val_file = ROOT / "data" / "processed" / "validation.csv"
        example = pd.read_csv(val_file, nrows=1)
        for name in FEATURES:
            if name in example.columns:
                st.session_state[name] = str(example.iloc[0][name])
        st.success("Loaded applicant sample from validation.csv!")
    except Exception as error:
        st.error(f"Example could not be loaded: {error}")

# Input Form
with st.form("loan_applicant_inputs"):
    values = {}
    for name in FEATURES:
        default_val = st.session_state.get(name, DEFAULT_VALUES.get(name, ""))
        label_text = LABELS.get(name, name)
        values[name] = st.text_input(label_text, value=default_val, key=name)

    submitted = st.form_submit_button("Predict Loan Approval (ANN)")

if submitted:
    try:
        result = predict_one(model, values)
    except ValueError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
    else:
        # Status Badge & Outcome
        is_approved = result.get("status") == "APPROVED" or result.get("predicted_class") == 1
        if is_approved:
            st.success(f"**Decision: {result.get('label', 'Loan Approved')}**")
        else:
            st.error(f"**Decision: {result.get('label', 'Loan Rejected')}**")

        st.write(f"**Model Class Output**: `{result.get('predicted_class')}`")
        if "approval_probability" in result:
            st.write(f"**Approval Confidence**: `{result['approval_probability']}%` (Policy Threshold: `{result.get('policy_threshold', '>= 75.0%')}`)")
            st.write(f"**Default Risk Exposure**: `{result['default_risk']}%`")
            st.write(f"**Recommended Action**: **{result.get('action', 'N/A')}**")

        st.caption(result.get("notice", ""))

        # Output Summary Table
        output_display = {k: v for k, v in result.items() if k != "input_features"}
        output_df = pd.DataFrame([output_display])
        st.dataframe(output_df)

        st.download_button(
            "Download this result as CSV",
            output_df.to_csv(index=False).encode("utf-8"),
            file_name="step5_prediction.csv",
            mime="text/csv",
        )

with st.expander("How this demonstration works"):
    st.write("The interface validates inputs and calls `predict_one()` on the saved pipeline.")
    st.write("The pipeline includes the fitted preprocessor (`StandardScaler` and `OrdinalEncoder`). No training runs here.")
    st.write("SC09 models credit risk for bank loan approvals using 16 demographic and financial features.")