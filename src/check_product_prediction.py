from pathlib import Path
import pandas as pd
from product_predict import FEATURES, load_model, predict_one

ROOT = Path(__file__).resolve().parents[1]

def main():
    sample = pd.read_csv(ROOT / "data" / "processed" / "validation.csv", nrows=1)
    values = {name: sample.iloc[0][name] for name in FEATURES}
    result = predict_one(load_model(), values)

    output = ROOT / "data" / "reports" / "step10_prediction.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result]).to_csv(output, index=False)

    print(pd.DataFrame([result]).to_string(index=False))
    print("Saved:", output)
    print("Step 10 consistency check completed.")

if __name__ == "__main__":
    main()