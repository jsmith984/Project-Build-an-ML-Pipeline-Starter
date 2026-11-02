#!/usr/bin/env python

import argparse
import json
import logging
import os
import shutil

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import wandb

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import prepare_features


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT = "nyc_airbnb"
ENTITY = "jsm394-western-governors-university"


# --------------------------------------------------
# FIND FILE
# --------------------------------------------------

def find_file(directory, extension):

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


# --------------------------------------------------
# INFERENCE PIPELINE
# --------------------------------------------------

def get_inference_pipeline(
    rf_config,
    max_tfidf_features,
    random_seed,
):

    numerical_preproc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preproc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_preproc,
                [
                    "latitude",
                    "longitude",
                    "minimum_nights",
                    "number_of_reviews",
                    "reviews_per_month",
                    "calculated_host_listings_count",
                    "availability_365",
                ],
            ),
            (
                "categorical",
                categorical_preproc,
                [
                    "neighbourhood_group",
                    "neighbourhood",
                    "room_type",
                ],
            ),
            (
                "title",
                TfidfVectorizer(
                    max_features=int(max_tfidf_features),
                    stop_words="english",
                    lowercase=True,
                    strip_accents="unicode",
                ),
                "name",
            ),
        ],
        remainder="drop",
    )

    # Copy configuration to avoid modifying
    # the original dictionary.

    rf_params = dict(rf_config)

    # Always use the pipeline's configured seed.
    # Remove any existing random_state to avoid
    # duplicate keyword arguments.

    rf_params.pop("random_state", None)

    rf_params["random_state"] = int(random_seed)

    logger.info(
        "Effective Random Forest configuration: %s",
        rf_params,
    )

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "random_forest",
                RandomForestRegressor(**rf_params),
            ),
        ]
    )

    return sk_pipe


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def go(args):

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="train_random_forest",
    )

    try:

        run.config.update(
            vars(args),
            allow_val_change=True,
        )

        # ------------------------------------------
        # DOWNLOAD TRAINING DATA
        # ------------------------------------------

        logger.info("Downloading training artifact.")

        artifact = run.use_artifact(
            args.trainval_artifact
        )

        artifact_dir = artifact.download()

        csv_file = find_file(
            artifact_dir,
            ".csv",
        )

        df = pd.read_csv(csv_file)

        logger.info(
            "Dataset dimensions: %s",
            df.shape,
        )

        # ------------------------------------------
        # PREPARE FEATURES AND TARGET
        # ------------------------------------------

        X, y = prepare_features(df)

        valid_rows = (
            y.notna()
            & np.isfinite(y)
        )

        X = X.loc[valid_rows].copy()
        y = y.loc[valid_rows].copy()

        if len(X) < 2:
            raise ValueError(
                "Not enough valid observations to train."
            )

        # ------------------------------------------
        # STRATIFICATION
        # ------------------------------------------

        stratify = None

        if args.stratify_by.lower() != "none":

            if args.stratify_by not in df.columns:
                raise ValueError(
                    f"Stratification column "
                    f"{args.stratify_by} does not exist."
                )

            stratify = df.loc[
                X.index,
                args.stratify_by,
            ]

            if stratify.isna().any():
                raise ValueError(
                    "Stratification column contains missing values."
                )

            if stratify.value_counts().min() < 2:
                raise ValueError(
                    "Every stratification group needs "
                    "at least two observations."
                )

        # ------------------------------------------
        # TRAIN / VALIDATION SPLIT
        # ------------------------------------------

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=float(args.val_size),
            random_state=int(args.random_seed),
            stratify=stratify,
        )

        logger.info(
            "Training dimensions: %s",
            X_train.shape,
        )

        logger.info(
            "Validation dimensions: %s",
            X_val.shape,
        )

        # ------------------------------------------
        # LOAD RANDOM FOREST CONFIGURATION
        # ------------------------------------------

        with open(args.rf_config, "r") as fp:
            rf_config = json.load(fp)

        if not isinstance(rf_config, dict):
            raise ValueError(
                "Random Forest configuration must be a JSON object."
            )

        # Create effective configuration for tracking.

        effective_rf_config = dict(rf_config)

        effective_rf_config["random_state"] = int(
            args.random_seed
        )

        logger.info(
            "Random Forest configuration: %s",
            effective_rf_config,
        )

        run.config.update(
            effective_rf_config,
            allow_val_change=True,
        )

        # ------------------------------------------
        # BUILD MODEL PIPELINE
        # ------------------------------------------

        sk_pipe = get_inference_pipeline(
            rf_config,
            args.max_tfidf_features,
            args.random_seed,
        )

        # ------------------------------------------
        # TRAIN MODEL
        # ------------------------------------------

        logger.info("Training Random Forest model.")

        sk_pipe.fit(
            X_train,
            y_train,
        )

        # ------------------------------------------
        # VALIDATION PREDICTIONS
        # ------------------------------------------

        predictions = sk_pipe.predict(X_val)

        # ------------------------------------------
        # EVALUATION METRICS
        # ------------------------------------------

        rmse = np.sqrt(
            mean_squared_error(
                y_val,
                predictions,
            )
        )

        mae = mean_absolute_error(
            y_val,
            predictions,
        )

        r2 = r2_score(
            y_val,
            predictions,
        )

        logger.info(
            "Validation RMSE: %.4f",
            rmse,
        )

        logger.info(
            "Validation MAE: %.4f",
            mae,
        )

        logger.info(
            "Validation R2: %.4f",
            r2,
        )

        # ------------------------------------------
        # LOG METRICS TO WANDB
        # ------------------------------------------

        run.log(
            {
                "MAE": float(mae),
                "validation_mae": float(mae),
                "validation_rmse": float(rmse),
                "validation_r2": float(r2),
            }
        )

        # ------------------------------------------
        # EXPORT MODEL USING MLFLOW
        # ------------------------------------------

        export_path = "model_export"

        if os.path.exists(export_path):
            shutil.rmtree(export_path)

        logger.info(
            "Exporting model using MLflow."
        )

        # Prepare input example.

        input_example = X_train.head(5).copy()

        # Convert integer columns to float64
        # to support possible missing values.

        for column in input_example.select_dtypes(
            include=["integer"]
        ).columns:

            input_example[column] = (
                input_example[column].astype("float64")
            )

        # Generate predictions for signature.

        signature_predictions = sk_pipe.predict(
            input_example
        )

        # Infer MLflow model signature.

        signature = mlflow.models.infer_signature(
            input_example,
            signature_predictions,
        )

        # Export complete sklearn pipeline.

        mlflow.sklearn.save_model(
            sk_pipe,
            export_path,
            input_example=input_example,
            signature=signature,
        )

        logger.info(
            "MLflow model exported successfully."
        )

        # ------------------------------------------
        # REGISTER MODEL ARTIFACT IN WANDB
        # ------------------------------------------

        model_artifact = wandb.Artifact(
            name=args.output_artifact,
            type="model",
            description=(
                "NYC Airbnb Random Forest "
                "MLflow sklearn model"
            ),
        )

        model_artifact.add_dir(
            export_path
        )

        run.log_artifact(
            model_artifact,
            aliases=["latest"],
        )

        logger.info(
            "Model registered successfully."
        )

    finally:

        run.finish()


# --------------------------------------------------
# COMMAND-LINE ARGUMENTS
# --------------------------------------------------

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
