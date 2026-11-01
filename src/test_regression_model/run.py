#!/usr/bin/env python

"""
Evaluate the production NYC Airbnb regression model
against the held-out test dataset.

Steps:
1. Initialize a W&B run.
2. Download the registered production model.
3. Download the held-out test dataset.
4. Validate and prepare the test data.
5. Generate predictions.
6. Calculate regression metrics.
7. Log metrics to W&B.
8. Save and register test predictions.
"""

import argparse
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import wandb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# 1. Configure project paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

TRAINING_DIR = os.path.join(
    PROJECT_ROOT,
    "src",
    "train_random_forest",
)

if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)


# Import shared feature engineering.
from feature_engineering import prepare_features


# --------------------------------------------------
# 2. Configure logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. W&B configuration
# --------------------------------------------------

PROJECT = "nyc_airbnb"

ENTITY = "jsm394-western-governors-university"


# --------------------------------------------------
# 4. Helper function to locate downloaded files
# --------------------------------------------------

def find_file(directory, extension):
    """
    Find exactly one file with the requested extension.
    """

    files = []

    for root, _, filenames in os.walk(directory):

        for filename in filenames:

            if filename.lower().endswith(extension.lower()):

                files.append(
                    os.path.join(root, filename)
                )

    if len(files) != 1:

        raise ValueError(
            f"Expected exactly one {extension} file "
            f"in {directory}, found: {files}"
        )

    return files[0]


# --------------------------------------------------
# 5. Main evaluation function
# --------------------------------------------------

def go(args):

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="test_regression_model",
    )

    try:

        run.config.update(vars(args))

        # ------------------------------------------
        # STEP 1: Download production model
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("STEP 1: DOWNLOADING PRODUCTION MODEL")
        logger.info("=" * 65)

        logger.info(
            "Model artifact: %s",
            args.mlflow_model,
        )

        model_artifact = run.use_artifact(
            args.mlflow_model
        )

        model_dir = model_artifact.download()

        model_file = find_file(
            model_dir,
            ".joblib",
        )

        if not os.path.isfile(model_file):

            raise FileNotFoundError(
                f"Model file not found: {model_file}"
            )

        model = joblib.load(model_file)

        if not hasattr(model, "predict"):

            raise TypeError(
                "Loaded model does not implement predict()."
            )

        logger.info(
            "Production model loaded successfully: %s",
            model_file,
        )

        # ------------------------------------------
        # STEP 2: Download held-out test dataset
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("STEP 2: DOWNLOADING TEST DATASET")
        logger.info("=" * 65)

        logger.info(
            "Test artifact: %s",
            args.test_artifact,
        )

        test_artifact = run.use_artifact(
            args.test_artifact
        )

        test_dir = test_artifact.download()

        csv_file = find_file(
            test_dir,
            ".csv",
        )

        df = pd.read_csv(csv_file)

        logger.info(
            "Test dataset loaded: %s",
            csv_file,
        )

        logger.info(
            "Original test dimensions: %s",
            df.shape,
        )

        if df.empty:

            raise ValueError(
                "Test dataset is empty."
            )

        # ------------------------------------------
        # STEP 3: Validate required columns
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("STEP 3: VALIDATING TEST DATA")
        logger.info("=" * 65)

        required_columns = [
            "price",
            "latitude",
            "longitude",
            "neighbourhood_group",
            "room_type",
            "name",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Test dataset is missing required columns: "
                f"{missing_columns}"
            )

        logger.info(
            "Required columns validated successfully."
        )

        # ------------------------------------------
        # STEP 4: Prepare features and target
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("STEP 4: PREPARING TEST FEATURES")
        logger.info("=" * 65)

        X_test, y_test = prepare_features(df)

        if len(X_test) != len(y_test):

            raise ValueError(
                "Feature and target row counts do not match."
            )

        # Convert target to numeric and remove invalid values.
        y_test = pd.to_numeric(
            y_test,
            errors="coerce",
        )

        valid_rows = (
            y_test.notna()
            & np.isfinite(y_test)
        )

        X_test = X_test.loc[
            valid_rows
        ].copy()

        y_test = y_test.loc[
            valid_rows
        ].copy()

        if len(X_test) == 0:

            raise ValueError(
                "No valid test observations remain."
            )

        logger.info(
            "Valid test observations: %d",
            len(X_test),
        )

        logger.info(
            "Test feature dimensions: %s",
            X_test.shape,
        )

        # ------------------------------------------
        # STEP 5: Generate predictions
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("STEP 5: GENERATING PREDICTIONS")
        logger.info("=" * 65)

        predictions = model.predict(X_test)

        predictions = np.asarray(
            predictions,
            dtype=float,
        )

        if len(predictions) != len(y_test):

            raise ValueError(
                "Prediction count does not match "
                "the number of test observations."
            )

        if not np.isfinite(predictions).all():

            raise ValueError(
                "Model generated invalid predictions."
            )

        logger.info(
            "Predictions generated successfully."
        )

        # ------------------------------------------
        # STEP 6: Calculate regression metrics
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("STEP 6: CALCULATING TEST METRICS")
        logger.info("=" * 65)

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        logger.info("-----------------------------------------")
        logger.info("FINAL NYC AIRBNB TEST RESULTS")
        logger.info("-----------------------------------------")

        logger.info(
            "Test observations: %d",
            len(y_test),
        )

        logger.info(
            "RMSE: %.4f",
            rmse,
        )

        logger.info(
            "MAE: %.4f",
            mae,
        )

        logger.info(
            "R2: %.4f",
            r2,
        )

        logger.info("-----------------------------------------")

        # ------------------------------------------
        # STEP 7: Log metrics to W&B
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("STEP 7: LOGGING METRICS TO W&B")
        logger.info("=" * 65)

        run.log(
            {
                "test_rmse": float(rmse),
                "test_mae": float(mae),
                "test_r2": float(r2),
                "test_observations": int(len(y_test)),
            }
        )

        logger.info(
            "Evaluation metrics logged successfully."
        )

        # ------------------------------------------
        # STEP 8: Save test predictions
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("STEP 8: SAVING TEST PREDICTIONS")
        logger.info("=" * 65)

        results = pd.DataFrame(
            {
                "actual_price": y_test.to_numpy(),
                "predicted_price": predictions,
                "prediction_error": (
                    y_test.to_numpy() - predictions
                ),
            }
        )

        output_file = "test_predictions.csv"

        results.to_csv(
            output_file,
            index=False,
        )

        logger.info(
            "Predictions saved: %s",
            output_file,
        )

        # ------------------------------------------
        # STEP 9: Register prediction artifact
        # ------------------------------------------

        logger.info("=" * 65)
        logger.info("STEP 9: REGISTERING PREDICTIONS")
        logger.info("=" * 65)

        results_artifact = wandb.Artifact(
            name="test_predictions",
            type="evaluation",
            description=(
                "NYC Airbnb regression model "
                "held-out test predictions"
            ),
        )

        results_artifact.add_file(
            output_file
        )

        run.log_artifact(
            results_artifact
        )

        logger.info(
            "Test prediction artifact registered successfully."
        )

        logger.info("=" * 65)
        logger.info("REGRESSION EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 65)

    except Exception:

        logger.exception(
            "Regression model evaluation failed."
        )

        raise

    finally:

        run.finish()


# --------------------------------------------------
# 6. Command-line arguments
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the production NYC Airbnb "
            "Random Forest regression model."
        )
    )

    parser.add_argument(
        "--mlflow_model",
        type=str,
        required=True,
        help="W&B production model artifact reference.",
    )

    parser.add_argument(
        "--test_artifact",
        type=str,
        required=True,
        help="W&B held-out test dataset artifact reference.",
    )

    args = parser.parse_args()

    go(args)
