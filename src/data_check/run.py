import logging
import pytest
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s"
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logger.info("Starting data quality checks")

    result = pytest.main([
        "-vv",
        "-s",
        "test_data.py",
        *sys.argv[1:],
    ])

    if result != pytest.ExitCode.OK:
        raise RuntimeError(
            f"Data quality checks failed. "
            f"Pytest exit code: {result}"
        )

    logger.info("All data quality checks passed")
