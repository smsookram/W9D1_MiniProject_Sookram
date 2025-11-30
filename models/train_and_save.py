from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import os

os.makedirs("models", exist_ok=True)

# Very small synthetic dataset (2 features)
X = np.array([
    [0.1, 1.2],
    [1.0, 0.9],
    [2.1, 2.2],
    [1.5, 1.8],
    [0.5, 0.2],
    [3.0, 3.1],
    [2.5, 0.5],
    [0.0, 0.0],
    [1.7, 2.8],
    [2.2, 1.9],
])
# binary target just for model purposes
y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

joblib.dump({"model": model, "version": "v1.0"}, "models/baseline.joblib")
print("Saved models/baseline.joblib")