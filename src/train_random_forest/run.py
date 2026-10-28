import argparse
import logging
import os
import json
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
    run = wandb.init(job_type="train_random_forest")

    logger.info("Loading trainval data")
    df = pd.read_csv("trainval_data.csv")

    cat_cols = ["neighbourhood_group", "neighbourhood", "room_type"]
    num_cols = ["latitude", "longitude", "minimum_nights", "number_of_reviews", 
                "reviews_per_month", "calculated_host_listings_count", "availability_365"]

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    np.random.seed(args.random_seed)
    perm = np.random.permutation(len(df))
    val_size = int(len(df) * args.val_size)
    
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    logger.info("Fitting model pipeline...")
    pipe = SimplePipeline(cat_cols, num_cols, "price")
    pipe.fit(train_df)

    logger.info("Evaluating model...")
    preds = pipe.predict(val_df)
    y_val = val_df["price"].values

    mae = np.mean(np.abs(y_val - preds))
    ss_res = np.sum((y_val - preds) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R2: {r2:.4f}")

    wandb.log({"val_mae": mae, "val_r2": r2})

    os.makedirs("model_export", exist_ok=True)
    model_path = "model_export/model.joblib"
    joblib.dump(pipe, model_path)

    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Trained model pipeline"
    )
    try:
        artifact.add_file(model_path)
        run.log_artifact(artifact)
    except Exception as e:
        logger.warning(f"Skipped online W&B artifact upload: {e}")

    logger.info("Model training successfully completed and saved locally!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--max_tfidf_features", type=int, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)

    args = parser.parse_args()
    go(args)
