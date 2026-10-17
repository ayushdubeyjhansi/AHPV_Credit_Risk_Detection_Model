from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
BASELINE_PATH = REPORTS_DIR / "baseline_comparison.csv"
COMPARISON_PATH = REPORTS_DIR / "baseline_vs_ann_comparison.csv"
GRAPH_PATH = REPORTS_DIR / "baseline_comparison.png"

baseline_df = pd.read_csv(BASELINE_PATH)
baseline = baseline_df.iloc[0]

# Values taken from the Step 3 ANN test evaluation output.
ann = {
    "Method": "Step 3 ANN",
    "Accuracy": 0.7489,
    "Precision": 0.7715,
    "Recall": 0.9316,
    "F1-score": 0.8440,
    "Test_samples": 15002,
}

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
rows = [
    {
        "Method": "Majority-class baseline",
        **{metric: float(baseline[metric]) for metric in metrics},
        "Test_samples": int(baseline["Test_samples"]),
    },
    ann,
]
comparison = pd.DataFrame(rows)

if comparison["Test_samples"].nunique() != 1:
    raise ValueError("Baseline and ANN must use the same number of test examples.")

difference = {"Method": "Difference (ANN - baseline)"}
for metric in metrics:
    difference[metric] = float(comparison.loc[1, metric] - comparison.loc[0, metric])
difference["Test_samples"] = int(comparison.loc[0, "Test_samples"])
comparison = pd.concat([comparison, pd.DataFrame([difference])], ignore_index=True)
comparison.to_csv(COMPARISON_PATH, index=False)

plot_df = comparison.iloc[:2].set_index("Method")[metrics]
ax = plot_df.plot(kind="bar", figsize=(9, 5))
ax.set_title("Majority-Class Baseline vs Step 3 ANN")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xlabel("")
ax.legend(loc="lower right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(GRAPH_PATH, dpi=300, bbox_inches="tight")
plt.close()

print("Comparison table:")
print(comparison.to_string(index=False))
print(f"Saved comparison CSV: {COMPARISON_PATH}")
print(f"Saved comparison graph: {GRAPH_PATH}")
