# Step 3: Artificial Neural Network (ANN) Baseline Model

## 1. Project Goal
The goal of this stage is to build, train, and evaluate a feed-forward Artificial Neural Network (ANN) pipeline to predict whether an applicant should be granted loan approval based on credit and demographic indicators[cite: 2, 6].

## 2. Model Inputs & Features
The model consumes tabular features from the banking loan approval dataset[cite: 2]:
- Categorical attributes encoded via ordinal transformation (e.g., job sector, marital status, education, housing)[cite: 2].
- Continuous financial and temporal features (e.g., age, account balance, contact duration, campaign metrics) scaled to zero mean and unit variance via `StandardScaler`[cite: 2, 6, 7].
- Fuzzy logic membership representations capturing non-linear boundary transitions across age, balance, and duration[cite: 2].

## 3. Output Definition
The target variable is binary classification:
- **`LOAN_APPROVAL = 1`**: Approved (Low credit risk)[cite: 2].
- **`LOAN_APPROVAL = 0`**: Rejected (High credit risk)[cite: 2].

## 4. ANN Architecture
- **Pipeline Structure**: Feature preprocessor (`StandardScaler` + `OrdinalEncoder`) feeding into an `MLPClassifier`[cite: 6, 7].
- **Hidden Layers**: Two fully connected hidden layers: `(32, 16)` neurons[cite: 2, 6].
- **Activation Function**: ReLU (`Rectified Linear Unit`)[cite: 2, 6].
- **Optimizer / Solver**: Adam[cite: 2, 6].
- **Reproducibility**: Random seed fixed to `42`[cite: 2, 6].

## 5. Data Splitting & Leakage Strategy
- Data is strictly partitioned into three independent sets: **Train**, **Validation**, and **Test**[cite: 3, 4].
- Preprocessors and standardizers are fitted exclusively on the training split and applied downstream to prevent data leakage[cite: 7, 15].
- The final test split remains uninspected during development and is evaluated only once for unbiased performance estimation[cite: 15].

## 6. Generated Evidence Files
All training artifacts and evaluation outputs are persisted to disk[cite: 17]:
- `models/ann_pipeline_model.joblib`: Trained pipeline model artifact[cite: 1, 17].
- `results/step3_ann_metrics.csv`: Summary metrics table (Accuracy, Precision, Recall, F1)[cite: 1, 11, 17].
- `results/step3_ann_loss_curve.png`: Training loss trajectory across iterations[cite: 1, 13].
- `results/step3_val_test_confusion_matrices.png`: Validation and test confusion matrix comparisons[cite: 1, 16].
- `results/validation_predictions.csv` & `results/test_predictions.csv`: Model predictions and probability outputs[cite: 1, 17].
- `results/step3_metadata.json`: Complete training configuration, layer parameters, and runtime metrics[cite: 1, 17].

## 7. Model Limitations
This model provides automated statistical risk scoring based on historical training data[cite: 10, 15]. It does not replace manual credit underwriting, full institutional audits, or real-time regulatory compliance checks. Decisions with borderline confidence thresholds require human underwriter review to prevent unintended financial exposure or discriminatory bias[cite: 2, 16].