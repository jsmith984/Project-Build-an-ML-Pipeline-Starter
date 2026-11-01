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

        # Download the input artifact from W&B
        logger.info("Downloading input artifact...")

        artifact = run.use_artifact(args.input_artifact)
        artifact_dir = artifact.download()

        # Read the downloaded CSV file
        df = pd.read_csv(
            os.path.join(artifact_dir, "sample1.csv")
        )

        logger.info(f"Original dataset shape: {df.shape}")

        # Filter price outliers
        min_price = args.min_price
        max_price = args.max_price

        idx = df["price"].between(min_price, max_price)
        df = df[idx].copy()

        logger.info(f"After price filtering: {df.shape}")

        # Convert last_review to datetime
        df["last_review"] = pd.to_datetime(
            df["last_review"],
            errors="coerce"
        )

        # Remove listings outside NYC geographic boundaries
        idx = (
            df["longitude"].between(-74.25, -73.50)
            & df["latitude"].between(40.5, 41.2)
        )

        df = df[idx].copy()

        logger.info(f"After geographic filtering: {df.shape}")

        # Save the cleaned dataset
        output_file = "clean_sample.csv"

        df.to_csv(output_file, index=False)

        logger.info(f"Saved cleaned dataset to {output_file}")

        # Create the output artifact
        cleaned_artifact = wandb.Artifact(
            args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )

        cleaned_artifact.add_file(output_file)

        # Upload cleaned artifact to W&B
        logger.info("Uploading cleaned artifact...")

        run.log_artifact(cleaned_artifact)

        # Wait for artifact upload to finish
        run.finish()

        logger.info("Basic cleaning completed successfully!")

    except Exception:
        logger.exception("Basic cleaning failed!")
        run.finish(exit_code=1)
        raise


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
        help="Type of the output W&B artifact"
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