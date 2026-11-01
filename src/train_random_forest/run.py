#!/usr/bin/env python

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import wandb

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from feature_engineering import (
    build_preprocessor,
    prepare_features,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT = "nyc_airbnb"
ENTITY = "jsm394-western-governors-university"


def find_file(directory, extension):
    """Find exactly one file with the specified extension."""

    files = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))

    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one {extension} file, found {files}"
        )

    return files[0]


def go(args):

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="train_random_forest",
    )

    try:

        run.config.update(vars(args))

        # Download training artifact.
        logger.info(
            "Downloading training artifact: %s",
            args.trainval_artifact,
        )

        artifact = run.use_artifact(args.trainval_artifact)
        artifact_dir = artifact.download()

        csv_file = find_file(artifact_dir, ".csv")

        df = pd.read_csv(csv_file)

        logger.info("Dataset dimensions: %s", df.shape)

        # Prepare features and target.
        X, y = prepare_features(df)

        valid_rows = y.notna() & np.isfinite(y)

        X = X.loc[valid_rows].copy()
        y = y.loc[valid_rows].copy()

        if len(X) < 2:
            raise ValueError("Not enough valid rows to train.")

        # Split into training and validation sets.
        stratify = None

        if args.stratify_by.lower() != "none":
            if args.stratify_by not in df.columns:
                raise ValueError(
                    f"Stratification column '{args.stratify_by}' "
                    "does not exist."
                )

            stratify = df.loc[valid_rows, args.stratify_by]

            if stratify.isna().any():
                raise ValueError(
                    "Stratification column contains missing values."
                )

            if stratify.value_counts().min() < 2:
                raise ValueError(
                    "Every stratification group needs at least "
                    "two observations."
                )

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=float(args.val_size),
            random_state=int(args.random_seed),
            stratify=stratify,
        )

        logger.info("Training dimensions: %s", X_train.shape)
        logger.info("Validation dimensions: %s", X_val.shape)

        # Load Random Forest parameters.
        with open(args.rf_config, "r") as fp:
            rf_config = json.load(fp)

        logger.info("Random Forest configuration: %s", rf_config)

        # Build preprocessing pipeline.
        preprocessor = build_preprocessor(
            X_train.assign(price=y_train),
            max_tfidf_features=int(args.max_tfidf_features),
        )

        # Initialize model.
        rf_model = RandomForestRegressor(
            **rf_config
        )

        # Combine preprocessing and model.
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", rf_model),
            ]
        )

        # Train model.
        logger.info("Training Random Forest model.")

        model.fit(X_train, y_train)

        # Validation predictions.
        predictions = model.predict(X_val)

        # Calculate metrics.
        rmse = np.sqrt(
            mean_squared_error(y_val, predictions)
        )

        mae = mean_absolute_error(
            y_val,
            predictions,
        )

        r2 = r2_score(
            y_val,
            predictions,
        )

        logger.info("Validation RMSE: %.4f", rmse)
        logger.info("Validation MAE: %.4f", mae)
        logger.info("Validation R2: %.4f", r2)

        # Log metrics to W&B.
        run.log(
            {
                "validation_rmse": float(rmse),
                "validation_mae": float(mae),
                "validation_r2": float(r2),
            }
        )

        # Export model.
        output_dir = "model_export"
        os.makedirs(output_dir, exist_ok=True)

        model_file = os.path.join(
            output_dir,
            "model.joblib",
        )

        joblib.dump(model, model_file)

        logger.info("Saved model: %s", model_file)

        # Register model artifact.
        model_artifact = wandb.Artifact(
            name=args.output_artifact,
            type="model",
            description="NYC Airbnb Random Forest regression model",
        )

        model_artifact.add_file(model_file)

        run.log_artifact(
            model_artifact,
            aliases=["latest", "prod"],
        )

        logger.info("Model registered successfully.")

    finally:
        run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val_size",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        default="none",
    )

    parser.add_argument(
        "--rf_config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max_tfidf_features",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
    )

    go(parser.parse_args())