import argparse
import logging
import pytest
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data quality")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--kl_threshold", type=float, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)

    args = parser.parse_args()

    # Load local cleaned sample
    data = pd.read_csv("clean_sample.csv")
    ref_data = pd.read_csv("clean_sample.csv")

    pytest.data = data
    pytest.ref_data = ref_data
    pytest.kl_threshold = args.kl_threshold
    pytest.min_price = args.min_price
    pytest.max_price = args.max_price

    res = pytest.main(["-s", "src/data_check/test_data.py"])
    if res != 0:
        raise ValueError("Data check failed")
