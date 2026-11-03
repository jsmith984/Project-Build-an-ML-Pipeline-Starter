#!/usr/bin/env python

"""
Download the raw dataset from W&B, clean the data,
and upload the cleaned dataset as a new W&B artifact.
"""

import argparse
import logging
import os

import pandas as pd
import wandb


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)

logger = logging.getLogger(__name__)


def go(args):

    # Initialize W&B run
    run = wandb.init(job_type="basic_cleaning")

    try:

        # Log configuration parameters
        run.config.update(vars(args))

        # Download input artifact
        logger.info("Downloading input artifact...")

        artifact = run.use_artifact(args.input_artifact)
        artifact_dir = artifact.download()

        # Locate CSV file dynamically
        csv_files = [
            os.path.join(root, filename)
            for root, _, files in os.walk(artifact_dir)
            for filename in files
            if filename.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected exactly one CSV file, "
                f"found: {csv_files}"
            )

        # Read dataset
        df = pd.read_csv(csv_files[0])

        logger.info(f"Original dataset shape: {df.shape}")

        # Filter price outliers
        logger.info(
            f"Filtering prices between "
            f"{args.min_price} and {args.max_price}"
        )

        idx = df["price"].between(
            args.min_price,
            args.max_price
        )

        df = df[idx].copy()

        logger.info(
            f"After price filtering: {df.shape}"
        )

        # Convert last_review to datetime
        df["last_review"] = pd.to_datetime(
            df["last_review"],
            errors="coerce"
        )

        # Remove listings outside NYC geographic boundaries
        logger.info("Filtering geographic boundaries...")

        idx = (
            df["longitude"].between(-74.25, -73.50)
            & df["latitude"].between(40.5, 41.2)
        )

        df = df[idx].copy()

        logger.info(
            f"After geographic filtering: {df.shape}"
        )

        # Save cleaned dataset
        output_file = "clean_sample.csv"

        df.to_csv(
            output_file,
            index=False
        )

        logger.info(
            f"Saved cleaned dataset to {output_file}"
        )

        # Create output W&B artifact
        cleaned_artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )

        cleaned_artifact.add_file(output_file)

        # Upload cleaned artifact
        logger.info("Uploading cleaned artifact...")

        run.log_artifact(cleaned_artifact)

        logger.info(
            "Basic cleaning completed successfully!"
        )

    except Exception:

        logger.exception("Basic cleaning failed!")

        run.finish(exit_code=1)

        raise

    else:

        run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Basic data cleaning"
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True,
        help="Input W&B artifact containing the raw dataset"
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name of the output W&B artifact"
    )

    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        help="Type of the output dataset"
    )

    parser.add_argument(
        "--output_description",
        type=str,
        required=True,
        help="Description of the cleaned dataset artifact"
    )

    parser.add_argument(
        "--min_price",
        type=float,
        required=True,
        help="Minimum nightly rental price to retain"
    )

    parser.add_argument(
        "--max_price",
        type=float,
        required=True,
        help="Maximum nightly rental price to retain"
    )

    args = parser.parse_args()

    go(args)
