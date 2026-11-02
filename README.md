NYC Airbnb Project
Project Links:

Weights & Biases Project: https://forge.coreweave.com/wandb/jsm394-western-governors-university/nyc_airbnb/reports/NYC-AirBnB-Reports--VmlldzoxODIzODI1NA

GitHub Repository: https://github.com/jsmith984/Project-Build-an-ML-Pipeline-Starter
--------------------------------------------------------------------------------------------------------------------------------------
NYC Airbnb Price Prediction – End-to-End ML Pipeline
Project Links
Weights & Biases Project: NYC Airbnb – W&B Dashboard
GitHub Repository: Project-Build-an-ML-Pipeline-Starter
(Finalize README with NYC Airbnb pipeline results)
Project Overview

This project implements a reusable, end-to-end machine learning pipeline to predict NYC Airbnb listing prices using a Random Forest regression model.

The goal is to estimate the typical nightly rental price of an Airbnb listing based on the characteristics of similar properties.

The pipeline automates data acquisition, cleaning, validation, preprocessing, model training, and evaluation. It uses MLflow Projects for pipeline orchestration, Hydra for configuration management, and Weights & Biases (W&B) for experiment tracking and artifact management.

The completed pipeline successfully executes all components from raw data acquisition through final model evaluation.

Technologies Used
Python 3.13
Pandas and NumPy
Scikit-learn
MLflow Projects
Hydra
Weights & Biases (W&B)
Conda
Git and GitHub
Pipeline Architecture

The pipeline consists of the following components:

Component	Description
get_data	Downloads and registers the raw Airbnb dataset.
basic_cleaning	Cleans the dataset and filters listing prices.
data_check	Validates data quality and dataset consistency.
data_split	Separates the dataset into training/validation and test datasets.
train_random_forest	Trains and exports the Random Forest regression model.
test_regression_model	Evaluates the trained model using held-out test data.
Pipeline Workflow
Acquire the raw NYC Airbnb dataset.
Clean the data and remove listings outside the configured price range.
Validate the cleaned dataset using automated tests.
Split the data into training/validation and test datasets.
Train a Random Forest regression model.
Export and register the trained model.
Evaluate the model against the held-out test dataset.
Log evaluation metrics and register prediction artifacts in W&B.
Environment Setup
Requirements
Python 3.13
Conda
A Weights & Biases account and API key
1. Clone the Repository
bash
git clone https://github.com/jsmith984/Project-Build-an-ML-Pipeline-Starter.git

cd Project-Build-an-ML-Pipeline-Starter

2. Create and Activate the Environment
bash
conda env create -f environment.yml

conda activate nyc_airbnb_dev

3. Configure Weights & Biases

Obtain an API key from:

https://wandb.ai/authorize

Log in using:

bash
wandb login


The pipeline uses W&B to track experiments, store datasets, register trained models, and save evaluation results.

Running the Pipeline

All commands should be executed from the root directory of the repository with the nyc_airbnb_dev environment activated.

Run the Entire Pipeline

To execute all pipeline components:

bash
python main.py main.steps=all

Run Individual Components

Data Splitting

bash
python main.py main.steps=data_split


Random Forest Model Training

bash
python main.py main.steps=train_random_forest modeling.random_forest.max_depth=15 modeling.random_forest.n_estimators=100


Regression Model Evaluation

bash
python main.py main.steps=test_regression_model


Individual components can also be executed independently after their required W&B artifacts have been registered.

Configuration Management

Pipeline parameters are managed through Hydra using the project's configuration files.

For example, Random Forest parameters can be overridden directly from the command line:

bash
python main.py main.steps=train_random_forest modeling.random_forest.max_depth=15 modeling.random_forest.n_estimators=100


This allows model configurations to be adjusted without modifying the underlying source code.

Model Configuration

The final Random Forest model uses the following configuration:

Parameter	Value
n_estimators	100
max_depth	15
min_samples_split	4
min_samples_leaf	3
max_features	0.5
criterion	squared_error
random_state	42
oob_score	True
n_jobs	-1
Model Evaluation Results

The trained Random Forest model was evaluated using a held-out test dataset containing 9,286 observations.

Final Test Metrics
Metric	Result
RMSE	47.9119
MAE	33.6050
R²	0.5743
Test Observations	9,286
Validation Metrics
Metric	Result
RMSE	46.1633
MAE	32.3759
R²	0.5785
Model Performance Analysis

The final test R² score of 0.5743 indicates that the model explains approximately 57.4% of the variation in Airbnb listing prices.

The model achieved a test Mean Absolute Error (MAE) of approximately $33.61, meaning its predictions differed from actual listing prices by approximately $33.61 per observation on average.

The validation and test metrics are relatively consistent, suggesting that the model generalizes reasonably well to previously unseen data.

Data Validation

The data validation component successfully executed all seven automated tests.

The tests cover:

Column name validation
Data boundary checks
Unique listing IDs
Missing neighborhood validation
Neighborhood distribution consistency
Dataset row count validation

Result: 7 tests passed.

The validation process helps ensure that the data meets the expected quality and consistency requirements before model training.

Experiment Tracking and Artifact Management

Weights & Biases is integrated throughout the pipeline to track experiments and manage data and model artifacts.

The pipeline registers and tracks:

Raw Airbnb dataset
Cleaned Airbnb dataset
Training/validation dataset
Held-out test dataset
Exported Random Forest model
Test prediction results

The following evaluation metrics are logged to W&B:

test_rmse
test_mae
test_r2
test_observations

The final test predictions are registered in Weights & Biases
under the artifact name test_predictions.

The artifact contains the prediction output file generated
by the evaluation component.

MLflow is used to orchestrate the pipeline components and export the trained model.

Final Pipeline Verification

The complete pipeline was successfully executed using:

bash
python main.py main.steps=all


All six pipeline components completed successfully:

Data acquisition
Data cleaning
Data validation
Data splitting
Random Forest model training
Regression model evaluation

The final test evaluation completed successfully with an R² score of 0.5743.

The repository was also verified after execution:

bash
git status


Result:

text
On branch main
Your branch is up to date with 'origin/main'.

Final repository synchronization and working-tree status
were verified after the final documentation and verification
updates were committed.


This confirms that the pipeline ran successfully without leaving uncommitted repository changes.

Conclusion

This project demonstrates the development of a reusable end-to-end machine learning pipeline for predicting NYC Airbnb listing prices.

By integrating MLflow, Hydra, Scikit-learn, and Weights & Biases, the project provides an automated workflow for data processing, validation, model training, evaluation, and artifact management.

The successful execution of the complete pipeline demonstrates that the individual components work together as a reproducible machine learning workflow.

Original Project Description

This project is based on the Udacity Build an ML Pipeline for Short-Term Rental Prices in NYC starter project.

The original project focuses on developing a reusable machine learning pipeline for a property management company that receives new rental data in bulk every week and needs to retrain its model regularly.

Original Starter Repository:
https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter

License

See LICENSE.txt.