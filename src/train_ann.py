import os
import json
from pathlib import Path
from datetime import datetime
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for directory in [DATA_DIR, PROCESSED_DATA_DIR, REPORT_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

TARGET_COL = 'LOAN_APPROVAL'
RANDOM_STATE = 42


def load_split(split_name: str, data_dir: Path = PROCESSED_DATA_DIR, target_col: str = TARGET_COL) -> pd.DataFrame:
    file_path = data_dir / f"{split_name}.csv"
    if not file_path.is_file():
        raise FileNotFoundError(f"[Validation Error] Split file not found: {file_path}")

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"[Validation Error] Split '{file_path.name}' is empty.")

    if target_col and target_col not in df.columns:
        raise KeyError(f"[Validation Error] Expected target '{target_col}' missing from '{file_path.name}'.")

    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        raise ValueError(f"[Validation Error] '{file_path.name}' has {missing_count} missing values.")

    print(f"Loaded '{file_path.name}': {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def metric_row(model, X, y_true, split_name: str, threshold: float = None) -> pd.DataFrame:
    if threshold is not None and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return pd.DataFrame([{
        'Split': split_name,
        'Threshold': f"{threshold:.2f}" if threshold is not None else "0.50 (default)",
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4)
    }])


def save_loss_curve(pipeline, output_path: Path):
    loss_history = pipeline.named_steps['ann'].loss_curve_
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='#1f77b4', linewidth=2, label='Training Loss')
    plt.title('ANN Training Loss Curve (Loan Approval Model)', fontsize=12, fontweight='bold')
    plt.xlabel('Iteration / Epoch')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Loss curve saved to: {output_path}")


def save_confusion_matrices(y_val, y_val_pred, y_test, y_test_pred, output_path: Path):
    cm_val = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
    cm_test = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    class_names = ['Rejected (0)', 'Approved (1)']

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Validation Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('Actual Label')

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Final Test Confusion Matrix', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('Actual Label')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrices saved to: {output_path}")


def main():
    print("=" * 60)
    print("Starting AHPV ANN Model Training Pipeline")
    print("=" * 60)

    train_df = load_split('train')
    val_df = load_split('validation')
    test_df = load_split('test')

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_val = val_df.drop(columns=[TARGET_COL])
    y_val = val_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=['number']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ]
    )

    ann_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ann', MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=RANDOM_STATE
        ))
    ])

    print("\nTraining the ANN pipeline...")
    ann_pipeline.fit(X_train, y_train)

    trained_ann = ann_pipeline.named_steps['ann']
    print(f"Training completed: {trained_ann.n_iter_} iterations, final loss: {trained_ann.loss_:.5f}")

    val_row = metric_row(ann_pipeline, X_val, y_val, split_name="Validation")
    test_row = metric_row(ann_pipeline, X_test, y_test, split_name="Test")
    final_metrics_df = pd.concat([val_row, test_row], ignore_index=True)

    print("\nModel Evaluation Summary:")
    print(final_metrics_df.to_string(index=False))

    y_val_pred = ann_pipeline.predict(X_val)
    y_test_pred = ann_pipeline.predict(X_test)

    print("\nTest Classification Report:")
    print(classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=['Rejected (0)', 'Approved (1)'],
        zero_division=0
    ))

    save_loss_curve(ann_pipeline, RESULTS_DIR / "step3_ann_loss_curve.png")
    save_confusion_matrices(
        y_val, y_val_pred, y_test, y_test_pred,
        RESULTS_DIR / "step3_val_test_confusion_matrices.png"
    )

    model_file = MODEL_DIR / "ann_pipeline_model.joblib"
    joblib.dump(ann_pipeline, model_file)

    metrics_file = RESULTS_DIR / "step3_ann_metrics.csv"
    final_metrics_df.to_csv(metrics_file, index=False)

    val_preds_df = pd.DataFrame({
        'actual': y_val.values,
        'pred_label': y_val_pred,
        'approval_probability': ann_pipeline.predict_proba(X_val)[:, 1]
    })
    val_preds_df.to_csv(RESULTS_DIR / "validation_predictions.csv", index=False)

    test_preds_df = pd.DataFrame({
        'actual': y_test.values,
        'pred_label': y_test_pred,
        'approval_probability': ann_pipeline.predict_proba(X_test)[:, 1]
    })
    test_preds_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    metadata = {
        "project_name": "AHPV Credit Risk Detection Model",
        "timestamp": datetime.now().isoformat(),
        "target_column": TARGET_COL,
        "feature_count": int(X_train.shape[1]),
        "architecture": {
            "hidden_layer_sizes": list(trained_ann.hidden_layer_sizes),
            "activation": trained_ann.activation,
            "solver": trained_ann.solver,
            "max_iter": trained_ann.max_iter,
            "random_state": trained_ann.random_state
        },
        "training_run": {
            "iterations_completed": int(trained_ann.n_iter_),
            "final_loss": float(trained_ann.loss_)
        },
        "dataset_sizes": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df))
        }
    }

    with open(RESULTS_DIR / "step3_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved model and Step 3 evidence files.")
    print(f"Model saved to: {model_file}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
