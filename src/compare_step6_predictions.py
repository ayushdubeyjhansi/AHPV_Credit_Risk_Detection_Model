import argparse
import csv
from math import isclose
from pathlib import Path
from product_predict import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "reports"

def read_prediction(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise ValueError(f"{path.name}: expected 1 row, found {len(rows)}.")
    row = rows[0]
    for field in [*FEATURES, "predicted_class", "label"]:
        if field not in row or row[field] == "":
            raise ValueError(f"{path.name}: missing value for '{field}'.")
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, nargs="?", default=REPORT_DIR / "step5_ui_reference_prediction.csv")
    parser.add_argument("app_download", type=Path, nargs="?", default=REPORT_DIR / "step6_ui_prediction.csv")
    args = parser.parse_args()

    try:
        ref = read_prediction(args.reference)
        app = read_prediction(args.app_download)
        diffs = []
        for col in [*NUMERIC_FEATURES, "approval_probability", "default_risk"]:
            if col in ref and col in app and not isclose(float(ref[col]), float(app[col]), rel_tol=1e-6, abs_tol=1e-6):
                diffs.append(col)
        for col in [*CATEGORICAL_FEATURES, "predicted_class", "label"]:
            if col in ref and col in app and ref[col].strip() != app[col].strip():
                diffs.append(col)
    except (OSError, ValueError, csv.Error) as err:
        parser.exit(1, f"FAIL: {err}\n")

    if diffs:
        parser.exit(1, f"FAIL: Differences in {', '.join(diffs)}.\n")
    print("PASS: All 16 SC09 features, predicted class, and label match.")
    print(f"Class {app['predicted_class']} - {app['label']}")

if __name__ == "__main__":
    main()