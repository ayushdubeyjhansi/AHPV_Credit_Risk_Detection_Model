import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# 1. Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

for d in [MODELS_DIR, RESULTS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "ann_pipeline_model.joblib"
METRIC_PATH = RESULTS_DIR / "step3_ann_metrics.csv"
META_PATH = RESULTS_DIR / "step3_metadata.json"
LOSS_PLOT_PATH = RESULTS_DIR / "step3_ann_loss_curve.png"
CM_PLOT_PATH = RESULTS_DIR / "step3_val_test_confusion_matrices.png"
DECISION_CARD_PATH = DOCS_DIR / "product-v1-sketch.png"

TARGET_COL = "LOAN_APPROVAL"


def load_dataset():
    """Load train, validation, and test splits."""
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"
    test_path = DATA_DIR / "test.csv"

    if not (train_path.is_file() and val_path.is_file() and test_path.is_file()):
        raise FileNotFoundError(
            f"Dataset splits missing in {DATA_DIR}. Ensure train.csv, validation.csv, and test.csv exist."
        )

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL].astype(int)

    X_val = df_val.drop(columns=[TARGET_COL])
    y_val = df_val[TARGET_COL].astype(int)

    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL].astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_pipeline(X_train: pd.DataFrame):
    """Explicitly assign numeric vs categorical features and assemble pipeline."""
    # Ensure binary flag columns are explicitly treated as categorical strings
    flag_cols = ["default", "housing_loan", "personal_loan"]
    
    numeric_features = [
        col for col in X_train.select_dtypes(include=[np.number]).columns
        if col not in flag_cols
    ]
    categorical_features = [
        col for col in X_train.columns
        if col not in numeric_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
        ],
        remainder="drop"
    )

    ann_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10,
        verbose=False
    )

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", ann_model)
    ])

    return pipeline, numeric_features, categorical_features


def evaluate_split(pipeline, X, y, split_name: str):
    y_pred = pipeline.predict(X)
    return {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
    }


def render_loan_decision_card(pipeline, applicant_df, applicant_id="APP-10492"):
    prob_approved = pipeline.predict_proba(applicant_df)[0, 1]
    prob_default = 1.0 - prob_approved
    decision = "APPROVED" if prob_approved >= 0.75 else "REJECTED"
    badge_color = "#2ca02c" if decision == "APPROVED" else "#d62728"

    fig = plt.figure(figsize=(9, 4.5), facecolor="#f8f9fa")
    
    # Left Panel: Text Card
    ax_card = fig.add_axes([0.05, 0.1, 0.45, 0.8])
    ax_card.axis("off")
    card_text = (
        f"LOAN DECISION CARD\n"
        f"-----------------------------------------\n"
        f"Applicant ID:     {applicant_id}\n"
        f"System Status:    {decision}\n"
        f"Threshold Policy: >= 75.0% for Approval\n"
        f"Confidence Score: {prob_approved * 100:.2f}%\n"
        f"Default Risk:     {prob_default * 100:.2f}%\n"
        f"-----------------------------------------\n"
        f"Action: {'Disburse Loan' if decision == 'APPROVED' else 'Refer to Underwriter'}"
    )
    ax_card.text(0.05, 0.5, card_text, fontsize=11, family="monospace", va="center",
                 bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="#cccccc", lw=1.5))

    # Right Panel: Risk Gauges
    ax_gauge = fig.add_axes([0.55, 0.2, 0.4, 0.6])
    categories = ["Default Risk", "Approval Prob."]
    scores = [prob_default * 100, prob_approved * 100]
    bar_colors = ["#d62728" if prob_default > 25 else "#ff7f0e", badge_color]

    y_pos = np.arange(len(categories))
    bars = ax_gauge.barh(y_pos, scores, color=bar_colors, height=0.45, edgecolor="black", linewidth=0.5)
    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_yticks(y_pos)
    ax_gauge.set_yticklabels(categories, fontsize=11, fontweight="bold")
    ax_gauge.set_xlabel("Probability Score (%)", fontsize=10)
    ax_gauge.set_title("Real-Time Risk Meters", fontsize=12, fontweight="bold", pad=12)
    ax_gauge.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, scores):
        ax_gauge.text(bar.get_width() - 12 if score > 15 else bar.get_width() + 2,
                      bar.get_y() + bar.get_height() / 2,
                      f"{score:.1f}%",
                      va="center",
                      ha="left" if score <= 15 else "right",
                      color="black" if score <= 15 else "white",
                      fontweight="bold", fontsize=10)

    plt.savefig(DECISION_CARD_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Product Decision Card saved to: {DECISION_CARD_PATH}")


def main():
    print("Loading data splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    print("Configuring pipeline...")
    pipeline, num_cols, cat_cols = build_pipeline(X_train)

    print("Fitting ANN pipeline...")
    pipeline.fit(X_train, y_train)

    print("Evaluating models...")
    val_metrics = evaluate_split(pipeline, X_val, y_val, "Validation")
    test_metrics = evaluate_split(pipeline, X_test, y_test, "Test")

    # Save metrics
    df_metrics = pd.DataFrame([val_metrics, test_metrics])
    df_metrics.to_csv(METRIC_PATH, index=False)
    print(f"Metrics saved to: {METRIC_PATH}")

    # Save permanent deployment bundle
    bundle = {
        "pipeline": pipeline,
        "features": list(X_train.columns),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "target": TARGET_COL,
        "threshold": 0.75,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Deployment bundle saved to: {MODEL_PATH}")

    # Save metadata JSON
    meta_info = {
        "features": list(X_train.columns),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "architecture": "MLPClassifier(32, 16)",
        "decision_threshold": 0.75,
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics
        }
    }
    META_PATH.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")
    print(f"Metadata saved to: {META_PATH}")

    # Generate product visualization
    sample_row = X_test.iloc[[0]]
    render_loan_decision_card(pipeline, sample_row, applicant_id="APP-SAMPLE-01")
    print("Training and artifact export complete.")


if __name__ == "__main__":
    main()