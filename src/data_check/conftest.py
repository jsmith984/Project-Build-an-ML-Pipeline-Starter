import os

import pandas as pd
import pytest
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store", required=True)
    parser.addoption("--ref", action="store", required=True)
    parser.addoption("--kl_threshold", action="store", default="0.2")
    parser.addoption("--min_price", action="store", default="10")
    parser.addoption("--max_price", action="store", default="350")


@pytest.fixture(scope="session")
def kl_threshold(request):
    return float(request.config.getoption("--kl_threshold"))


@pytest.fixture(scope="session")
def min_price(request):
    return float(request.config.getoption("--min_price"))


@pytest.fixture(scope="session")
def max_price(request):
    return float(request.config.getoption("--max_price"))


@pytest.fixture(scope="session")
def wandb_run():
    run = wandb.init(
        project="nyc_airbnb",
        entity="jsm394-western-governors-university",
        job_type="data_check",
    )

    try:
        yield run
    finally:
        run.finish()


def load_artifact_csv(run, artifact_name):
    artifact = run.use_artifact(artifact_name)
    artifact_dir = artifact.download()

    csv_files = [
        os.path.join(root, filename)
        for root, _, files in os.walk(artifact_dir)
        for filename in files
        if filename.lower().endswith(".csv")
    ]

    if len(csv_files) != 1:
        raise ValueError(
            f"Expected exactly one CSV in {artifact_name}, "
            f"found: {csv_files}"
        )

    return pd.read_csv(csv_files[0])


@pytest.fixture(scope="session")
def data(request, wandb_run):
    artifact_name = request.config.getoption("--csv")
    return load_artifact_csv(wandb_run, artifact_name)


@pytest.fixture(scope="session")
def ref_data(request, wandb_run):
    artifact_name = request.config.getoption("--ref")
    return load_artifact_csv(wandb_run, artifact_name)
