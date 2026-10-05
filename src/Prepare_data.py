from pathlib import Path
import numpy as np
import pandas as pd

# 1. Project paths (relative to script location, never hardcoded desktop/downloads)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = (
    PROJECT_ROOT / "data" / "raw" / "banking_loan_approval_dataset_100k.csv"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 2. Random seed
RANDOM_SEED = 42

# 3. Target and feature fields for SC09
TARGET_COL = "LOAN_APPROVAL"

FEATURE_COLS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing_loan",
    "personal_loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

REQUIRED_COLS = FEATURE_COLS + [TARGET_COL]


def inspect_data() -> None:
    """Inspects raw dataset characteristics prior to pipeline execution."""
    df = pd.read_csv(RAW_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    print("SC09 BANKING LOAN APPROVAL DATA INSPECTION")
    print("-" * 45)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nLoan approval class distribution:")
    print(df[TARGET_COL].value_counts().sort_index())
    print("\nFirst five rows:")
    print(df.head())


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by median-imputing numeric features and removing duplicates.

    Operates strictly on an independent copy.
    """
    clean_df = raw_df.copy()
    clean_df.columns = clean_df.columns.astype(str).str.strip()

    numeric_features = [
        col
        for col in clean_df.select_dtypes(include="number").columns
        if col != TARGET_COL
    ]

    clean_df[numeric_features] = clean_df[numeric_features].fillna(
        clean_df[numeric_features].median()
    )
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    return clean_df


def validate_data(df: pd.DataFrame) -> None:
    """Validates cleaned banking data against domain constraints, target labels,

    null counts, and duplicate rules. Raises ValueError if any check fails.
    """
    # 1. Required columns
    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Validation Failed: Missing required columns: {sorted(list(missing_cols))}"
        )

    # 2. Missing values
    total_nans = int(df.isna().sum().sum())
    if total_nans > 0:
        raise ValueError(
            f"Validation Failed: Found {total_nans} remaining NaN/missing values."
        )

    # 3. Domain range (Age: 18 to 120)
    invalid_age_count = int(((df["age"] < 18) | (df["age"] > 120)).sum())
    if invalid_age_count > 0:
        raise ValueError(
            f"Validation Failed: Found {invalid_age_count} records with out-of-range age (<18 or >120)."
        )

    # 4. Non-negative features
    non_neg_cols = ["age", "duration", "campaign", "previous"]
    for col in non_neg_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                raise ValueError(
                    f"Validation Failed: Column '{col}' contains {neg_count} negative values."
                )

    # 5. Target labels
    valid_labels = (
        {0, 1}
        if pd.api.types.is_numeric_dtype(df[TARGET_COL])
        else {"yes", "no", "0", "1"}
    )
    invalid_targets = int((~df[TARGET_COL].isin(valid_labels)).sum())
    if invalid_targets > 0:
        raise ValueError(
            f"Validation Failed: Target column '{TARGET_COL}' contains {invalid_targets} invalid labels."
        )

    # 6. Duplicate rows
    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        raise ValueError(
            f"Validation Failed: Found {duplicates} duplicate rows in dataset."
        )

    print("VALIDATION PASSED: Cleaned data satisfies all project rules.")


def create_stratified_splits(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
):
    """Splits data into train, val, and test partitions with preserved class ratios."""
    assert (
        np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    ), "Split ratios must sum to 1.0"

    train_chunks = []
    val_chunks = []
    test_chunks = []

    for class_label in df[target_col].unique():
        class_subset = df[df[target_col] == class_label].sample(
            frac=1.0, random_state=seed
        )
        total_len = len(class_subset)

        train_end = int(train_ratio * total_len)
        val_end = train_end + int(val_ratio * total_len)

        train_chunks.append(class_subset.iloc[:train_end])
        val_chunks.append(class_subset.iloc[train_end:val_end])
        test_chunks.append(class_subset.iloc[val_end:])

    train_df = (
        pd.concat(train_chunks)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_chunks)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat(test_chunks)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df


def run_pipeline() -> None:
    """Executes the complete data preparation pipeline from raw CSV to processed splits."""
    print("=" * 50)
    print("   SC09 DATA PREPARATION PIPELINE EXECUTION")
    print("=" * 50)

    # 1. Load raw dataset
    print(f"\n[1/5] Loading raw data from: {RAW_FILE.name}")
    raw_df = pd.read_csv(RAW_FILE, encoding="utf-8-sig")
    print(f"      Loaded {len(raw_df):,} rows and {raw_df.shape[1]} columns.")

    # 2. Clean data
    print("\n[2/5] Cleaning data (median imputation & duplicate removal)...")
    clean_df = clean_data(raw_df)
    print(f"      Cleaned dataset retains {len(clean_df):,} rows.")

    # 3. Validate cleaned data
    print("\n[3/5] Validating cleaned dataset...")
    validate_data(clean_df)

    # 4. Stratified splitting
    print("\n[4/5] Performing stratified train/validation/test split...")
    train_df, val_df, test_df = create_stratified_splits(
        clean_df,
        target_col=TARGET_COL,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=RANDOM_SEED,
    )

    # 5. Export processed CSV files
    print("\n[5/5] Exporting processed artifacts...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files_to_save = {
        "cleaned_data.csv": clean_df,
        "train.csv": train_df,
        "validation.csv": val_df,
        "test.csv": test_df,
    }

    for filename, dataset in files_to_save.items():
        output_path = PROCESSED_DIR / filename
        dataset.to_csv(output_path, index=False)

    # Print summary of sizes and verified files
    print("\n" + "=" * 50)
    print("           PIPELINE EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Random Seed        : {RANDOM_SEED}")
    print(f"Raw Input Rows     : {len(raw_df):,}")
    print(f"Cleaned Rows       : {len(clean_df):,}")
    print(f"Train Rows (70%)   : {len(train_df):,}")
    print(f"Val Rows   (15%)   : {len(val_df):,}")
    print(f"Test Rows  (15%)   : {len(test_df):,}")
    print("-" * 50)
    print("Saved Files in data/processed/:")
    for filename in files_to_save.keys():
        file_path = PROCESSED_DIR / filename
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  - {filename:<18} ({size_mb:.2f} MB)")
    print("=" * 50)
    print("Pipeline run completed successfully.")


if __name__ == "__main__":
    run_pipeline()