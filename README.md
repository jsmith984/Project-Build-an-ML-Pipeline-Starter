NYC Airbnb Price Prediction – End-to-End ML Pipeline
Project Overview

This project implements a reusable, end-to-end machine learning pipeline to predict nightly Airbnb listing prices in New York City using a Random Forest regression model.

The project is based on the Udacity Build an ML Pipeline for Short-Term Rental Prices in NYC starter project.

The business scenario involves a property management company that receives new rental listing data regularly and needs a repeatable process for acquiring, cleaning, validating, processing, and evaluating data before training machine learning models.

The pipeline automates the complete machine learning workflow, from raw data acquisition through model training and final evaluation.

The project integrates MLflow Projects for pipeline orchestration, Hydra for configuration management, Scikit-learn for machine learning, and Weights & Biases (W&B) for experiment tracking and artifact management.

Project Links
GitHub Repository: https://github.com/jsmith984/Project-Build-an-ML-Pipeline-Starter
Weights & Biases Project: https://wandb.ai/jsm394-western-governors-university/nyc_airbnb
Weights & Biases Reports: https://forge.coreweave.com/wandb/jsm394-western-governors-university/nyc_airbnb/reports/NYC-AirBnB-Reports--VmlldzoxODIzODI1NA
Original Udacity Repository: https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices
Technologies Used
Technology	Purpose
Python 3.13	Application development
Pandas	Data processing and manipulation
NumPy	Numerical computations
Scikit-learn	Machine learning
Random Forest Regressor	Price prediction
MLflow Projects	Pipeline orchestration and model export
Hydra	Configuration management
Weights & Biases	Experiment tracking and artifact management
Conda	Python environment management
Git	Version control
GitHub	Source code hosting
Project Architecture

The machine learning pipeline consists of six major components.

Component	Description
Data Acquisition	Downloads and registers the raw Airbnb dataset.
Basic Cleaning	Cleans the data and filters listing prices.
Data Validation	Executes automated data quality tests.
Data Splitting	Creates training/validation and held-out test datasets.
Random Forest Training	Trains and exports the regression model.
Model Evaluation	Generates predictions, calculates evaluation metrics, and registers results.
Pipeline Workflow
Acquire the raw NYC Airbnb dataset.
Clean the dataset and filter listing prices.
Validate the cleaned dataset using automated tests.
Split the data into training/validation and test datasets.
Train the Random Forest regression model.
Export the trained model using MLflow.
Evaluate the model using previously unseen test observations.
Log evaluation metrics and register prediction artifacts in Weights & Biases.
Repository Structure

The repository contains the following important directories and files:

text
Project-Build-an-ML-Pipeline-Starter/
│
├── components/
│   ├── get_data/
│   ├── basic_cleaning/
│   ├── train_val_test_split/
│   └── wandb_utils/
│
├── src/
│   ├── basic_cleaning/
│   ├── data_check/
│   ├── eda/
│   ├── train_random_forest/
│   └── test_regression_model/
│
├── data/
├── artifacts/
├── images/
│
├── config.yaml
├── environment.yml
├── conda.yml
├── MLproject
├── main.py
├── rf_config.json
├── wandb_utils.py
├── check_wandb_artifacts.py
└── README.md

Environment Setup
Requirements
Python 3.13
Conda
Git
Weights & Biases account
1. Clone the Repository
bash
git clone https://github.com/jsmith984/Project-Build-an-ML-Pipeline-Starter.git

cd Project-Build-an-ML-Pipeline-Starter

2. Create the Conda Environment
bash
conda env create -f environment.yml

3. Activate the Environment
bash
conda activate nyc_airbnb_dev

4. Verify Hydra Installation
bash
python -c "import hydra; print('Hydra installed successfully')"


Expected output:

text
Hydra installed successfully

5. Configure Weights & Biases

Create or access a Weights & Biases account:

https://wandb.ai/

Obtain an API key:

https://wandb.ai/authorize

Authenticate using:

bash
wandb login


Weights & Biases is used to track pipeline experiments, store datasets, register trained models, and manage evaluation results.

Running the Machine Learning Pipeline

All commands should be executed from the repository's root directory with the nyc_airbnb_dev Conda environment activated.

Execute the Complete Pipeline
bash
python main.py main.steps=all


This command executes all six pipeline components in sequence.

Execute Individual Pipeline Components

Individual components can also be executed through the main pipeline configuration when their required upstream artifacts are available.

For example, model training:

bash
python main.py main.steps=train_random_forest


Model evaluation:

bash
python main.py main.steps=test_regression_model


The exact available step names are determined by the project's Hydra configuration.

Data Processing Results

The original NYC Airbnb dataset contained 48,895 observations and 16 columns.

The basic cleaning component filtered listing prices using the configured minimum and maximum price boundaries.

Dataset Cleaning
Processing Stage	Observations
Original dataset	48,895
After price filtering	46,428
Final cleaned dataset	46,427

Approximately 94.95% of the original observations were retained after cleaning.

Data Validation

The data validation component executed seven automated tests.

The tests covered:

Column name validation.
Data boundary checks.
Unique listing IDs.
Missing neighborhood validation.
Neighborhood distribution consistency.
Dataset row count validation.
Remaining configured validation checks.

Validation Result: 7 tests passed.

Nine deprecation warnings appeared during execution. These warnings originated from the Weights & Biases integration and did not cause test failures.

Dataset Splitting

The cleaned dataset was divided into training/validation and held-out testing datasets.

Training and Testing Split
Dataset	Observations
Training/validation	37,141
Held-out testing	9,286
Total	46,427

The training/validation dataset was further divided:

Training and Validation Split
Subset	Observations
Training	29,712
Validation	7,429
Total	37,141

The split used:

Random seed: 42
Stratification column: neighbourhood_group
Validation proportion: 20% of the training/validation dataset
Random Forest Model Configuration

The final model uses Scikit-learn's Random Forest Regressor.

The following configuration was used:

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

The model was trained successfully using the training dataset.

Validation performance was calculated before exporting the model.

Validation Results
Metric	Result
RMSE	46.1633
MAE	32.3759
R²	0.5785
Final Model Evaluation

The trained Random Forest model was evaluated using the held-out test dataset containing 9,286 observations.

The evaluation component successfully:

Downloaded the registered model artifact.
Loaded the exported MLflow model.
Downloaded the held-out test dataset.
Validated the required test columns.
Prepared the test features.
Generated the test predictions.
Calculated regression evaluation metrics.
Logged the results to Weights & Biases.
Saved the test predictions.
Registered the prediction artifact.
Final Test Results
Metric	Result
RMSE	47.9119
MAE	33.6050
R²	0.5743
Test Observations	9,286
Validation Versus Test Performance
Metric	Validation	Test
RMSE	46.1633	47.9119
MAE	32.3759	33.6050
R²	0.5785	0.5743
Model Performance Analysis

The final test R² score of 0.5743 indicates that the model explains approximately 57.4% of the variation in Airbnb listing prices within the held-out test dataset.

The Mean Absolute Error (MAE) of approximately $33.61 indicates that the model's predictions differ from actual listing prices by approximately $33.61 per observation on average.

The Root Mean Squared Error (RMSE) of approximately $47.91 gives greater weight to larger prediction errors.

The validation and test metrics are relatively consistent, suggesting that the model maintains similar performance on previously unseen data.

The results demonstrate a functioning machine learning pipeline, although there is room for further improvement in predictive accuracy.

Test Predictions

The evaluation component generated a CSV file containing the actual prices, predicted prices, and prediction errors.

Output File
text
src/test_regression_model/test_predictions.csv


The CSV contains:

Column	Description
actual_price	Actual Airbnb listing price
predicted_price	Model-predicted listing price
prediction_error	Difference between actual and predicted prices

The file contains 9,286 prediction records, plus the CSV header.

The generated prediction file was approximately 372 KB during the November 2, 2026 execution.

The prediction results were also registered as a Weights & Biases artifact.

Weights & Biases Integration

Weights & Biases was successfully integrated throughout the machine learning pipeline.

The following artifacts were registered and verified:

Artifact	Type	Verification
sample.csv	raw_data	Verified
clean_data.csv	clean_data	Verified
trainval_data.csv	trainval_data	Verified
test_data.csv	test_data	Verified
model_export	model	Verified
test_predictions	evaluation	Verified

All six artifacts were successfully located using the project's W&B verification script.

Final Evaluation Metrics Logged to W&B

The following metrics were recorded:

text
test_rmse
test_mae
test_r2
test_observations

November 2, 2026 Experiment Runs
Pipeline Component	W&B Run Name
Data Acquisition	radiant-meadow-122
Data Cleaning	cool-deluge-123
Data Validation	laced-firebrand-124
Data Splitting	floral-frost-125
Model Training	glad-plant-126
Model Evaluation	rose-oath-127
Weights & Biases Resources
Project Dashboard: https://wandb.ai/jsm394-western-governors-university/nyc_airbnb
Reports: https://forge.coreweave.com/wandb/jsm394-western-governors-university/nyc_airbnb/reports/NYC-AirBnB-Reports--VmlldzoxODIzODI1NA
Model Export and Reproducibility

MLflow Projects orchestrates the execution of the individual pipeline components.

The trained Random Forest model was successfully exported through MLflow and registered as a W&B artifact.

The pipeline uses Hydra to manage configuration settings, allowing model parameters and pipeline steps to be adjusted without modifying the underlying source code.

The complete pipeline can be executed using:

bash
python main.py main.steps=all


This provides a repeatable workflow for acquiring new data, validating records, training a model, and evaluating its performance.

Final Pipeline Verification

The complete machine learning pipeline was successfully executed on November 2, 2026.

The final terminal output confirmed:

text
============================================================
NYC AIRBNB ML PIPELINE COMPLETED
============================================================


All six pipeline components completed successfully:

Data acquisition.
Basic data cleaning.
Data validation.
Dataset splitting.
Random Forest model training.
Regression model evaluation.

Additional verification confirmed:

Hydra was successfully installed and imported.
Seven automated data validation tests passed.
The MLflow model export completed successfully.
The final test predictions were generated.
The prediction artifact was registered successfully.
All six expected W&B artifacts were found.
The prediction CSV contained 9,286 records.
Git reported a clean working tree at the time of the post-execution check.

The Git working-tree check returned:

text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean


This confirms that the repository was clean and synchronized at that verification point. Any subsequent documentation changes should be committed and pushed separately.

Conclusion

This project demonstrates the development and execution of a reusable end-to-end machine learning pipeline for predicting NYC Airbnb listing prices.

By integrating Python, Pandas, NumPy, Scikit-learn, MLflow, Hydra, Conda, and Weights & Biases, the project provides an automated workflow for data acquisition, cleaning, validation, dataset splitting, model training, model export, and evaluation.

The Random Forest model achieved a test R² score of 0.5743, an MAE of $33.6050, and an RMSE of $47.9119 across 9,286 held-out test observations.

The successful execution of all six pipeline components, completion of seven validation tests, registration of the trained model, and verification of all six W&B artifacts demonstrate that the pipeline operates successfully as an integrated machine learning workflow.

The project establishes a foundation for future model improvements, additional experimentation, and repeatable evaluation of new Airbnb listing data.

Original Project Description

This project is based on the Udacity Build an ML Pipeline for Short-Term Rental Prices in NYC starter project.

The original business scenario focuses on a property management company that receives new rental listing data in bulk every week and needs an automated, reusable machine learning pipeline to process incoming data and retrain its price prediction model regularly.
