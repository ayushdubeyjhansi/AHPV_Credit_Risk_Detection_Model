from pathlib import Path
import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "generated_water_quality_test_scenarios.csv"

RANDOM_SEED = 42
TARGET_COL = "LOAN_APPROVAL"


def generate_scenarios(
    base_df: pd.DataFrame, num_scenarios: int = 256, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generates synthetic stress-test scenarios (edge cases/perturbations)

    derived from the cleaned baseline data.
    """
    np.random.seed(seed)

    # Sample reference profiles from baseline data
    sample_df = base_df.sample(
        n=num_scenarios, replace=True, random_state=seed
    ).copy()

    numeric_cols = sample_df.select_dtypes(include="number").columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    # Add Gaussian jitter/perturbation to continuous features to create edge cases
    for col in numeric_cols:
        std = sample_df[col].std()
        jitter = np.random.normal(0, 0.05 * (std if std > 0 else 1.0), size=num_scenarios)
        sample_df[col] = sample_df[col] + jitter

        # Keep domain bounds valid (e.g. non-negative fields)
        if col in ["age", "duration", "campaign", "previous"]:
            sample_df[col] = sample_df[col].clip(lower=0)
        if col == "age":
            sample_df[col] = sample_df[col].clip(lower=18, upper=120)

    # Round integer-like columns
    int_cols = ["age", "day", "duration", "campaign", "pdays", "previous"]
    for col in int_cols:
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].round().astype(int)

    return sample_df.reset_index(drop=True)


def run_generator() -> None:
    """Orchestrates test scenario creation and saves the CSV artifact."""
    if not CLEANED_DATA_PATH.exists():
        # Fallback to raw data if cleaned_data.csv hasn't been written
        raw_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "banking_loan_approval_dataset_100k.csv"
        )
        base_df = pd.read_csv(raw_path, encoding="utf-8-sig")
    else:
        base_df = pd.read_csv(CLEANED_DATA_PATH)

    base_df.columns = base_df.columns.str.strip()

    # Generate 256 stress-test instances (~10% evaluation artifact per guidelines)
    scenarios_df = generate_scenarios(base_df, num_scenarios=256, seed=RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully generated {len(scenarios_df)} scenarios at: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_generator()