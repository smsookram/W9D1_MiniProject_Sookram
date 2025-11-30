import sys
import pandas as pd
import joblib
import time
import os

def usage():
    print("Usage: python batch_infer.py input.csv output.csv")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    start = time.time()
    df = pd.read_csv(input_path)

    # Expect columns x1 and x2
    if not {"x1", "x2"}.issubset(df.columns):
        print("Input CSV must contain columns: x1, x2")
        sys.exit(1)

    saved = joblib.load("models/baseline.joblib")
    model = saved["model"]

    # Predict probabilities for class 1
    preds = model.predict_proba(df[["x1", "x2"]].values)[:, 1]
    df["prediction"] = preds

    df.to_csv(output_path, index=False)

    elapsed = time.time() - start
    print(f"Processed {len(df)} rows in {elapsed:.3f} seconds")
    print(f"Wrote predictions to {output_path}")