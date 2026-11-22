from pathlib import Path
import pandas as pd
from product_predict import load_model, predict_one

ROOT = Path(__file__).resolve().parents[1]

CUSTOM_INPUT = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 4500.0,
    "housing_loan": "no",
    "personal_loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 250.0,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}

def main():
    result = predict_one(load_model(), CUSTOM_INPUT)

    output = ROOT / "data" / "reports" / "step5_ui_reference_prediction.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(output, index=False)

    print(f"Predicted class: {result['predicted_class']} ({result['label']})")
    print("Saved:", output)

if __name__ == "__main__":
    main()