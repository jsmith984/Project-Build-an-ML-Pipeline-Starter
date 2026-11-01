#!/usr/bin/env python

import argparse
import logging
import os
import sys

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

# Make the shared W&B utilities available.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)

logger = logging.getLogger(__name__)

PROJECT = "nyc_airbnb"
ENTITY = "jsm394-western-governors-university"


def go(args):

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="train_val_test_split"
    )

    try:
        run.config.update(vars(args))

        # Download the cleaned dataset.
        logger.info("Downloading artifact: %s", args.input)

        artifact = run.use_artifact(args.input)
        artifact_dir = artifact.download()

        csv_files = [
            os.path.join(root, filename)
            for root, _, files in os.walk(artifact_dir)
            for filename in files
            if filename.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected exactly one CSV file, found {csv_files}"
            )

        df = pd.read_csv(csv_files[0])

        logger.info("Loaded dataset with shape: %s", df.shape)

        if len(df) < 2:
            raise ValueError("Dataset must contain at least two rows.")

        # Determine whether stratification is enabled.
        stratify = None

        if args.stratify_by.lower() != "none":
            if args.stratify_by not in df.columns:
                raise ValueError(
                    f"Stratification column '{args.stratify_by}' "
                    "does not exist."
                )

            stratify = df[args.stratify_by]

            if stratify.isna().any():
                raise ValueError(
                    "Stratification column contains missing values."
                )

            if stratify.value_counts().min() < 2:
                raise ValueError(
                    "Every stratification group must have at least "
                    "two observations."
                )

        # Split into train/validation and test.
        trainval, test = train_test_split(
            df,
            test_size=float(args.test_size),
            random_state=int(args.random_seed),
            stratify=stratify
        )

        logger.info("Train/validation dimensions: %s", trainval.shape)
        logger.info("Test dimensions: %s", test.shape)

        # Export datasets.
        trainval_file = "trainval_data.csv"
        test_file = "test_data.csv"

        trainval.to_csv(trainval_file, index=False)
        test.to_csv(test_file, index=False)

        # Register train/validation artifact.
        log_artifact(
            "trainval_data.csv",
            "trainval_data",
            "Training and validation dataset",
            trainval_file,
            run
        )

        # Register held-out test artifact.
        log_artifact(
            "test_data.csv",
            "test_data",
            "Held-out test dataset",
            test_file,
            run
        )

        logger.info("Data splitting completed successfully.")

    finally:
        run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Split Airbnb data into train/validation and test sets."
    )

    parser.add_argument("input", type=str)

    parser.add_argument("test_size", type=float)

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        default="none"
    )

    args = parser.parse_args()

    go(args)
