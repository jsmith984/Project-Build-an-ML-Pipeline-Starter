import argparse
import logging
import os
import pandas as pd
import numpy as np
import wandb
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

class SimplePipeline:
    def __init__(self, cat_cols, num_cols, target_col):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col
        self.means = {}
        self.modes = {}
        self.weights = None

    def fit(self, df):
        for col in self.num_cols:
            self.means[col] = df[col].median()
        
        X = pd.get_dummies(df[self.cat_cols + self.num_cols].fillna(self.means), drop_first=True)
        self.feature_columns = X.columns
        y = df[self.target_col].values
        
        X_mat = X.values.astype(float)
        self.x_mean = np.mean(X_mat, axis=0)
        self.x_std = np.std(X_mat, axis=0) + 1e-8
        X_norm = (X_mat - self.x_mean) / self.x_std
        X_design = np.hstack([np.ones((len(X_norm), 1)), X_norm])
        
        I = np.eye(X_design.shape[1])
        I[0, 0] = 0
        self.weights = np.linalg.solve(X_design.T @ X_design + 1.0 * I, X_design.T @ y)

    def predict(self, df):
        X = pd.get_dummies(df[self.cat_cols + self.num_cols].fillna(self.means), drop_first=True)
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        X_mat = X.values.astype(float)
        X_norm = (X_mat - self.x_mean) / self.x_std
        X_design = np.hstack([np.ones((len(X_norm), 1)), X_norm])
        return X_design @ self.weights

def go(args):
    run = wandb.init(job_type="test_regression_model")

    logger.info("Loading test data and trained model export")
    test_df = pd.read_csv("test_data.csv")
    model = joblib.load("model_export/model.joblib")

    logger.info("Evaluating model on test dataset...")
    test_preds = model.predict(test_df)
    y_test = test_df["price"].values

    mae = np.mean(np.abs(y_test - test_preds))
    ss_res = np.sum((y_test - test_preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test R2: {r2:.4f}")

    wandb.log({"test_mae": mae, "test_r2": r2})
    logger.info("Final model evaluation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test regression model")
    parser.add_argument("--mlflow_model", type=str, required=True)
    parser.add_argument("--test_artifact", type=str, required=True)

    args = parser.parse_args()
    go(args)
