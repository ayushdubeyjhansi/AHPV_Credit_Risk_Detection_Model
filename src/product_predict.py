"""
AHPV Credit Risk Detection Model - Inference Module (Project SC09)
"""

from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "ann_pipeline_model.joblib"

# Top-level export expected by app.py
NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing_loan",
    "personal_loan",
    "contact",
    "month",
    "poutcome"
]

FEATURES = [
    "age", "job", "marital", "education", "default", "balance",
    "housing_loan", "personal_loan", "contact", "day", "month",
    "duration", "campaign", "pdays", "previous", "poutcome"
]


def load_model():
    """Load the trained model pipeline or deployment bundle."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved model missing at {MODEL_PATH}. Run 'python src/train_ann.py' first."
        )

    bundle = joblib.load(MODEL_PATH)
    
    if isinstance(bundle, dict) and "pipeline" in bundle:
        model = bundle["pipeline"]
    else:
        model = bundle

    return model


def validate_inputs(values: dict) -> dict:
    """Validate data types and presence before inference."""
    if not isinstance(values, dict):
        raise TypeError("Applicant inputs must be provided as a dictionary.")

    clean = {}

    for col in NUMERIC_FEATURES:
        if col not in values or values[col] is None:
            raise ValueError(f"Missing required numeric feature: '{col}'")
        try:
            val = float(values[col])
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{col}' must be a valid number, got: {values[col]}")

        if col == "age" and (val < 18 or val > 120):
            raise ValueError(f"Invalid age '{val}'. Age must be between 18 and 120.")
        if col in ["duration", "campaign", "previous"] and val < 0:
            raise ValueError(f"Feature '{col}' cannot be negative, got: {val}")

        clean[col] = val

    for col in CATEGORICAL_FEATURES:
        if col not in values or values[col] is None:
            raise ValueError(f"Missing required categorical feature: '{col}'")
        val_str = str(values[col]).strip()
        if not val_str:
            raise ValueError(f"Feature '{col}' cannot be empty.")
        clean[col] = val_str

    return clean


def predict_one(model, values: dict) -> dict:
    """Run inference for a single applicant record."""
    clean = validate_inputs(values)

    # Frame in exact feature sequence
    frame = pd.DataFrame([[clean[name] for name in FEATURES]], columns=FEATURES)

    prediction = int(model.predict(frame)[0])
    if hasattr(model, "predict_proba"):
        approval_prob = float(model.predict_proba(frame)[0, 1])
    else:
        approval_prob = float(prediction)

    default_risk = 1.0 - approval_prob
    threshold = 0.75
    status = "APPROVED" if approval_prob >= threshold else "REJECTED"

    return {
        "status": status,
        "predicted_class": prediction,
        "label": "Loan Approved" if prediction == 1 else "Loan Rejected",
        "approval_probability": round(approval_prob * 100, 2),
        "default_risk": round(default_risk * 100, 2),
        "policy_threshold": f">= {threshold * 100:.1f}%",
        "action": "Disburse Loan" if status == "APPROVED" else "Refer to Credit Underwriter",
        "notice": "Decision support score only. This does not replace formal regulatory checks.",
        **clean,
    }