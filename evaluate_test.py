import json
import os
import sys

import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Import the original feature engineering function.
sys.path.insert(
    0,
    os.path.join(
        os.getcwd(),
        "src",
        "train_random_forest",
    ),
)

from feature_engineering import prepare_features


MODEL_PATH = "src/train_random_forest/model_export"
TEST_PATH = "test_data.csv"
OUTPUT_PATH = "artifacts/test_evaluation.json"


print("\n1. Loading exported MLflow model...")

model = mlflow.pyfunc.load_model(MODEL_PATH)

print("Model loaded successfully.")


print("\n2. Loading held-out test dataset...")

df = pd.read_csv(TEST_PATH)

print("Original test dataset shape:", df.shape)


print("\n3. Preparing test features and target...")

X_test, y_test = prepare_features(df)

# Match the training pipeline's valid-target filtering.
valid_rows = y_test.notna() & np.isfinite(y_test)

X_test = X_test.loc[valid_rows].copy()
y_test = y_test.loc[valid_rows].copy()

# Match the numerical input types required by MLflow.
numeric_columns = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

for column in numeric_columns:
    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="raise",
    ).astype("float64")

print("Test observations after filtering:", len(X_test))
print("Feature shape:", X_test.shape)


print("\n4. Generating test predictions...")

predictions = np.asarray(model.predict(X_test)).reshape(-1)

y_true = y_test.to_numpy(dtype=float)

assert len(predictions) == len(y_true), (
    "Prediction count does not match actual target count."
)

assert np.isfinite(predictions).all(), (
    "Predictions contain NaN or infinity."
)

print("Predictions generated successfully.")


print("\n5. Calculating test metrics...")

test_rmse = float(
    np.sqrt(mean_squared_error(y_true, predictions))
)

test_mae = float(
    mean_absolute_error(y_true, predictions)
)

test_r2 = float(
    r2_score(y_true, predictions)
)

results = {
    "model": "RandomForestRegressor",
    "evaluation_dataset": TEST_PATH,
    "test_observations": int(len(y_true)),
    "test_rmse": test_rmse,
    "test_mae": test_mae,
    "test_r2": test_r2,
    "mean_actual_price": float(np.mean(y_true)),
    "mean_predicted_price": float(np.mean(predictions)),
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 4,
        "min_samples_leaf": 3,
        "max_features": 0.5,
        "criterion": "squared_error",
        "oob_score": True,
        "random_state": 42,
    },
}


print("\n========== FINAL TEST RESULTS ==========")

print(f"Test observations: {len(y_true)}")
print(f"Test RMSE:         {test_rmse:.4f}")
print(f"Test MAE:          {test_mae:.4f}")
print(f"Test R2:           {test_r2:.4f}")

print("========================================")


print("\n6. Saving evaluation results...")

os.makedirs("artifacts", exist_ok=True)

with open(OUTPUT_PATH, "w") as fp:
    json.dump(results, fp, indent=4)

print("Results saved to:", OUTPUT_PATH)

print("\nSUCCESS: Held-out test evaluation completed.")

