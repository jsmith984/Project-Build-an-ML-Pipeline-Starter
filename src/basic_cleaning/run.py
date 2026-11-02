#!/usr/bin/env python

"""
Download raw Airbnb data, clean it,
and register the cleaned dataset in W&B.
"""

import argparse
import logging
import os

import pandas as pd
import wandb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT = "nyc_airbnb"
ENTITY = "jsm394-western-governors-university"


def find_csv(directory):

    csv_files = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(".csv"):
                csv_files.append(
                    os.path.join(root, filename)
                )

    if len(csv_files) != 1:
        raise ValueError(
            f"Expected one CSV file, found: {csv_files}"
        )

    return csv_files[0]


def go(args):

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="basic_cleaning",
    )

    try:

        run.config.update(vars(args))

        # Download raw artifact.
        logger.info(
            "Downloading: %s",
            args.input_artifact,
        )

        artifact = run.use_artifact(
            args.input_artifact
        )

        artifact_dir = artifact.download()

        input_file = find_csv(artifact_dir)

        df = pd.read_csv(input_file)

        logger.info(
            "Original dimensions: %s",
            df.shape,
        )

        required_columns = [
            "id",
            "latitude",
            "longitude",
            "price",
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # Clean price.
        df["price"] = pd.to_numeric(
            df["price"],
            errors="coerce",
        )

        df = df[
            df["price"].between(
                args.min_price,
                args.max_price,
            )
        ].copy()

        logger.info(
            "After price filtering: %s",
            df.shape,
        )

        # Clean coordinates.
        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce",
        )

        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce",
        )

        geographic_mask = (
            df["longitude"].between(-74.25, -73.50)
            & df["latitude"].between(40.5, 41.2)
        )

        df = df[geographic_mask].copy()

        # Convert review dates.
        df["last_review"] = pd.to_datetime(
            df["last_review"],
            errors="coerce",
        )

        # Remove duplicate listing IDs.
        df = df.drop_duplicates(
            subset=["id"]
        )

        if df.empty:
            raise ValueError(
                "Cleaning produced an empty dataset."
            )

        logger.info(
            "Final cleaned dimensions: %s",
            df.shape,
        )

        # Export cleaned dataset.
        output_file = "clean_data.csv"

        df.to_csv(
            output_file,
            index=False,
        )

        # Register cleaned artifact.
        cleaned_artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )

        cleaned_artifact.add_file(
            output_file
        )

        logged_artifact = run.log_artifact(
            cleaned_artifact,
            aliases=["latest"],
        )

        logged_artifact.wait()

        logger.info(
            "Registered artifact: %s",
            logged_artifact.name,
        )

    finally:
        run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_artifact",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        required=True,
    )

    go(parser.parse_args())
