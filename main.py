#!/usr/bin/env python

import json
import logging
import os

import hydra
import mlflow
from omegaconf import DictConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s",
)

logger = logging.getLogger(__name__)


def run_component(root_dir, component_path, parameters):
    """
    Execute an MLflow component.

    Args:
        root_dir: Absolute project root directory.
        component_path: Relative path to the component.
        parameters: Parameters passed to the MLproject entry point.

    Returns:
        MLflow run result.
    """

    component_dir = os.path.join(
        root_dir,
        component_path,
    )

    logger.info(
        "Starting component: %s",
        component_path,
    )

    result = mlflow.run(
        component_dir,
        "main",
        env_manager="local",
        parameters=parameters,
    )

    logger.info(
        "Component completed successfully: %s",
        component_path,
    )

    return result


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="config",
)
def go(config: DictConfig):
    """
    Run the NYC Airbnb machine learning pipeline.

    Available steps:
        download
        basic_cleaning
        data_check
        data_split
        train_random_forest
        test_regression_model

    Use 'all' to execute every pipeline stage.
    """

    # Get original project directory.
    root_dir = hydra.utils.get_original_cwd()

    os.chdir(root_dir)

    # Determine active pipeline steps.
    active_steps = [
        step.strip().lower()
        for step in config.main.steps.split(",")
        if step.strip()
    ]

    def should_run(step):
        return (
            "all" in active_steps
            or step.lower() in active_steps
        )

    logger.info("=" * 60)
    logger.info("STARTING NYC AIRBNB ML PIPELINE")
    logger.info("=" * 60)

    logger.info(
        "Active pipeline steps: %s",
        active_steps,
    )

    # --------------------------------------------------
    # 1. DOWNLOAD DATA
    # --------------------------------------------------

    if should_run("download"):

        run_component(
            root_dir,
            "components/get_data",
            {
                "sample": config.etl.sample,
                "artifact_name": "sample.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw Airbnb data",
            },
        )

    # --------------------------------------------------
    # 2. BASIC DATA CLEANING
    # --------------------------------------------------

    if should_run("basic_cleaning"):

        run_component(
            root_dir,
            "src/basic_cleaning",
            {
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Cleaned Airbnb data",
                "min_price": config.etl.min_price,
                "max_price": config.etl.max_price,
            },
        )

    # --------------------------------------------------
    # 3. DATA QUALITY VALIDATION
    # --------------------------------------------------

    if should_run("data_check"):

        run_component(
            root_dir,
            "src/data_check",
            {
                "csv": "clean_sample.csv:latest",
                "ref": "clean_sample.csv:reference",
                "kl_threshold": config.data_check.kl_threshold,
                "min_price": config.etl.min_price,
                "max_price": config.etl.max_price,
            },
        )

    # --------------------------------------------------
    # 4. TRAIN / VALIDATION / TEST SPLITTING
    # --------------------------------------------------

    if should_run("data_split"):

        run_component(
            root_dir,
            "components/train_val_test_split",
            {
                "input": "clean_sample.csv:latest",
                "test_size": config.modeling.test_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by,
            },
        )

    # --------------------------------------------------
    # 5. RANDOM FOREST MODEL TRAINING
    # --------------------------------------------------

    if should_run("train_random_forest"):

        rf_config = os.path.join(
            root_dir,
            "rf_config.json",
        )

        with open(rf_config, "w") as fp:

            json.dump(
                dict(config.modeling.random_forest),
                fp,
                indent=4,
            )

        run_component(
            root_dir,
            "src/train_random_forest",
            {
                "trainval_artifact": "trainval_data.csv:latest",
                "val_size": config.modeling.val_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by,
                "rf_config": rf_config,
                "max_tfidf_features": config.modeling.max_tfidf_features,
                "output_artifact": "model_export",
            },
        )

    # --------------------------------------------------
    # 6. FINAL REGRESSION MODEL EVALUATION
    # --------------------------------------------------

    if should_run("test_regression_model"):

        run_component(
            root_dir,
            "src/test_regression_model",
            {
                "model_artifact": "model_export:prod",
                "test_artifact": "test_data.csv:latest",
            },
        )

    # --------------------------------------------------
    # PIPELINE COMPLETED
    # --------------------------------------------------

    logger.info("=" * 60)
    logger.info("NYC AIRBNB ML PIPELINE COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":

    go()
