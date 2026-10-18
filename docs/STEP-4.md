# Step 4 — Baseline Comparison

## 1. Project Details

- **Project:** SC09 — Credit Risk Detection Model
- **Dataset:** Banking Loan Approval Dataset
- **Dataset Source:** https://www.kaggle.com/datasets/parthangare/banking-loan-approval-dataset
- **Target Column:** `LOAN_APPROVAL`
- **Target Labels:** `0 = Rejected`, `1 = Approved`
- **Step 3 Model:** Artificial Neural Network (ANN)
- **ANN Architecture:** 32 and 16 hidden neurons
- **Random State:** 42
- **Test Examples:** 15,002

## 2. Comparison Question

Does the Step 3 Artificial Neural Network perform better than a majority-class baseline when predicting `LOAN_APPROVAL` on the same test dataset?

Both methods will be evaluated using accuracy, precision, recall, and F1-score.

## 3. Baseline Rule

The baseline is a **Majority-Class Classifier**. It identifies the most frequent class in the training labels and predicts that class for every test example.

This method is simple, reproducible, and provides a reference point for evaluating the ANN. It does not learn relationships between applicant features and loan approval.

The baseline must use training labels only and must not use test labels to select the majority class.

## 4. Dataset and Experimental Setup

- Training dataset: `data/processed/train.csv`
- Test dataset: `data/processed/test.csv`
- Target column: `LOAN_APPROVAL`
- Training and testing splits must remain identical for both methods.
- The Step 3 implementation must remain unchanged.
- The test dataset must not be used for training or tuning.

## 5. Step 3 ANN Results

The following results were obtained from the existing model evaluation output.

| Metric | ANN Result |
|---|---:|
| Accuracy | 0.7489 |
| Precision (Class 1) | 0.7715 |
| Recall (Class 1) | 0.9316 |
| F1-score (Class 1) | 0.8440 |
| Test Examples | 15,002 |

### Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Rejected (0) | 0.58 | 0.26 | 0.36 | 4,060 |
| Approved (1) | 0.77 | 0.93 | 0.84 | 10,942 |

## 6. Evaluation Metrics

- **Accuracy:** Percentage of all predictions that are correct.
- **Precision:** Percentage of predicted approvals that are actually approved, for Class 1.
- **Recall:** Percentage of actual approvals correctly predicted, for Class 1.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for both classes.

Accuracy alone may be misleading because the dataset contains unequal numbers of approved and rejected examples.

## 7. Baseline Comparison

Run the baseline and evaluate it on the same 15,002 test examples before completing this table.

| Method | Accuracy | Precision | Recall | F1-score | Test Samples |
|---|---:|---:|---:|---:|---:|
| Majority-Class Baseline | 0.729369  |  0.729369  | 1.0000 | 0.843509 | 15,002 |
| Step 3 ANN | 0.7489 | 0.7715 | 0.9316 | 0.8440 | 15,002 |
| Difference (ANN - Baseline) | 0.019531 | 0.042131 | -0.0684 | 0.000491 | 15,002 |

**Difference Formula:**

Difference = ANN Score - Baseline Score

A positive difference means the ANN has a higher score for that metric.

## 8. Results Interpretation

- **Where the ANN helps:** Complete after comparing actual baseline and ANN scores.
- **Where the ANN struggles:** The ANN has a recall of 0.26 for Rejected (Class 0), indicating that many rejected examples are incorrectly predicted as approved.
- **Practical Meaning:** False approvals and false rejections can have different consequences in a lending process.
- **Overall Conclusion:** Determine whether the ANN outperforms the baseline after measuring both methods on the same test set.

Do not claim improvement unless the results support it.

## 9. Limitations

- The majority-class baseline ignores applicant features.
- The dataset has an imbalanced target distribution.
- The dataset documentation does not specify the currency of the balance field.
- The dataset lacks complete year-level temporal context.
- Results depend on data quality, preprocessing, and the test split.
- The model should not be the sole basis for real lending decisions.

## 10. Reproducibility and Run Instructions

- **Baseline File:** `src/baseline.py`
- **Comparison Results:** `reports/baseline_comparison.csv`
- **Comparison Graph:** `reports/baseline_comparison.png`
- **Dependencies:** Install packages listed in `requirements.txt`.
- **Run Command:** Add the verified command after implementing and testing the baseline.

Keep the baseline separate from the Step 3 ANN code.

## 11. Team Contributions

| Team Member | Contribution |
|---|---|
| Pradhuman Dixit | Baseline implementation |
| Ayush Dubey | Evaluation and comparison |
| Himanshu | Documentation and graph generation |

Enter the actual contributions of each team member.

## 12. Submission Checklist

- [ ] Step 3 code remains accessible.
- [ ] Baseline is implemented separately.
- [ ] Baseline uses training labels only.
- [ ] Both methods use the same test data.
- [ ] Actual baseline metrics are recorded.
- [ ] Metric differences are calculated.
- [ ] Comparison CSV is generated.
- [ ] Comparison graph is generated.
- [ ] Results and limitations are explained.
- [ ] Run instructions are documented.
- [ ] Team contributions are recorded.
- [ ] Changes are committed and pushed to the `step-4-baseline` branch.
- [ ] Step 4 is submitted through the team dashboard.

---
