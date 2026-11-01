#!/usr/bin/env python

"""
Upload a local dataset to Weights & Biases.
"""

import argparse
import logging
import os
import sys

import wandb

# Locate repository root.
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

# Import shared artifact utility.
sys.path.insert(
    0,
    os.path.join(PROJECT_ROOT, "components")
)

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT = "nyc_airbnb"
ENTITY = "jsm394-western-governors-university"


def go(args):

    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        job_type="download_file",
    )

    try:

        run.config.update(vars(args))

        # Resolve dataset from repository root.
        dataset_path = os.path.join(
            PROJECT_ROOT,
            "data",
            args.sample,
        )

        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        logger.info(
            "Using dataset: %s",
            dataset_path,
        )

        logger.info(
            "Uploading artifact: %s",
            args.artifact_name,
        )

        # Upload dataset to W&B.
        log_artifact(
            args.artifact_name,
            args.artifact_type,
            args.artifact_description,
            dataset_path,
            run,
        )

        logger.info(
            "Data acquisition completed successfully."
        )

    finally:
        run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "sample",
        type=str,
    )

    parser.add_argument(
        "artifact_name",
        type=str,
    )

    parser.add_argument(
        "artifact_type",
        type=str,
    )

    parser.add_argument(
        "artifact_description",
        type=str,
    )

    go(parser.parse_args())