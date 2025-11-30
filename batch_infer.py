import sys
import pandas as pd
import joblib
import time

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_infer.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Load model
    saved = joblib.load("models/baseline.joblib")
    model = saved["model"]   # extract the actual LogisticRegression

    # Load input CSV
    df = pd.read_csv(input_csv)
    
    # Check that required columns exist
    if not {"x1", "x2"}.issubset(df.columns):
        print("Error: input CSV must have columns 'x1' and 'x2'")
        sys.exit(1)

    start_time = time.time()

    # Generate predictions
    df["prediction"] = model.predict_proba(df[["x1", "x2"]])[:, 1]

    # Save output CSV
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"Processed {len(df)} rows in {elapsed:.2f} seconds")
    print(f"Output saved to {output_csv}")

if __name__ == "__main__":
    main()
