from setuptools import setup, find_packages

setup(
    name="wandb-utils",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "mlflow",
        "wandb",
    ],
)
