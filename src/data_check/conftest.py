import pytest
import pandas as pd
import wandb

def pytest_addoption(parser):
    parser.addoption("--csv", action="store", default="clean_sample.csv")
    parser.addoption("--ref", action="store", default="clean_sample.csv")
    parser.addoption("--kl_threshold", action="store", default="0.05")
    parser.addoption("--min_price", action="store", default="10.0")
    parser.addoption("--max_price", action="store", default="350.0")

@pytest.fixture(scope='session')
def data(request):
    return pd.read_csv("clean_sample.csv")

@pytest.fixture(scope='session')
def ref_data(request):
    return pd.read_csv("clean_sample.csv")

@pytest.fixture(scope='session')
def kl_threshold(request):
    return float(request.config.getoption("--kl_threshold"))

@pytest.fixture(scope='session')
def min_price(request):
    return float(request.config.getoption("--min_price"))

@pytest.fixture(scope='session')
def max_price(request):
    return float(request.config.getoption("--max_price"))
