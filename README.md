NYC Airbnb Price Prediction – End-to-End ML Pipeline
Project Overview

This project implements a reusable, end-to-end machine learning pipeline to predict nightly Airbnb listing prices in New York City using a Random Forest regression model.

The project is based on Udacity's Build an ML Pipeline for Short-Term Rental Prices in NYC starter project.

The business scenario involves a property management company that receives new rental listing data regularly and needs a repeatable process for acquiring, cleaning, validating, processing, and evaluating data before training machine learning models.

The pipeline automates the machine learning workflow, from raw data acquisition through model training and final evaluation.

The project integrates MLflow Projects for pipeline orchestration, Hydra for configuration management, Scikit-learn for machine learning, and Weights & Biases (W&B) for experiment tracking and artifact management.

Project Links
GitHub Repository: https://github.com/jsmith984/Project-Build-an-ML-Pipeline-Starter
Weights & Biases Reports: [https://forge.coreweave.com/wandb/jsm394-western-governors-university/nyc_airbnb/reports/NYC-AirBnB-Reports--VmlldzoxODIzODI1NA]
(https://forge.coreweave.com/wandb/jsm394-western-governors-university/nyc_airbnb/reports/NYC-Airbnb-ML-Pipeline-Evaluation-Report--VmlldzoxODI0ODc4MA?accessToken=v76ln741vmlx55eo3n0ec2mmfzj6so7ob0f7ak9qbx46dph6jbrxk1y2dh2n2v1o)
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

All commands should be executed from the repository root directory with the nyc_airbnb_dev Conda environment activated.

Execute the Complete Pipeline
bash
python main.py main.steps=all


This command executes all six pipeline components in sequence.

Execute Individual Pipeline Components

Individual components can also be executed through the main pipeline configuration when their required upstream artifacts are available.

Model training:

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

The one-record difference between the intermediate and final cleaned datasets reflects an additional processing or cleaning step.

Data Validation

The data validation component executed seven automated tests.

The tests covered:

Column name validation.
Data boundary checks.
Unique listing IDs.
Missing neighborhood validation.
Neighborhood distribution consistency.
Dataset row count validation.
Additional configured validation requirements.

Validation result: 7 of 7 tests passed.

Nine deprecation warnings appeared during execution. These warnings originated from the Weights & Biases integration and did not prevent successful execution.

Dataset Splitting

The cleaned dataset was divided into training/validation and held-out testing datasets.

Original Pipeline Training Split

The original November 2 training experiment used the following dataset counts:

Dataset                 Observations
Training/validation     37,141
Held-out testing        9,286
Total                   46,427

The training/validation dataset was further divided:

Subset          Observations
Training        29,712
Validation      7,429
Total           37,141

The original split used:

Random seed: 42
Stratification column: neighbourhood_group
Validation proportion: 20% of the training/validation dataset

Current Local Dataset Verification – November 3, 2026

A subsequent inspection of the current root-level CSV files identified:

Dataset                 Observations
Training/validation     37,143
Held-out testing        9,284
Total                   46,427

Both datasets contain valid price targets. No duplicate listing IDs or overlapping listing IDs were found.

The current local test dataset contains 9,284 observations, which were used for the independent November 3 model evaluation.

The original training experiment and the subsequent independent evaluation are documented separately because their dataset counts differ.

Random Forest Model Configuration

The final model uses Scikit-learn's Random Forest Regressor.

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

The model was trained successfully using the training dataset. Validation performance was calculated before exporting the model.

Validation Results
Metric	Result
RMSE	46.1633
MAE	32.3759
R²	0.5785
Final Model Evaluation

The exported Random Forest model was independently evaluated on November 3, 2026, using the current held-out test dataset and the existing MLflow model artifact.

The evaluation used evaluate_test.py and did not retrain the model or modify its hyperparameters.

The evaluation successfully:

Loaded the exported MLflow model.
Loaded the held-out test dataset.
Applied the original feature engineering function.
Generated predictions for previously unseen observations.
Calculated RMSE, MAE, and R².
Saved the evaluation results as JSON.

Final Test Results – November 3, 2026

Metric          Result
RMSE            39.6831
MAE             27.7701
R²              0.6913
Test observations 9,284

Validation Versus Test Performance

Metric          Validation      Test
RMSE            46.1633         39.6831
MAE             32.3759         27.7701
R²              0.5785          0.6913

Model Performance Analysis

The final test R² score of 0.6913 indicates that the model explains approximately 69.1% of the variation in Airbnb listing prices within the held-out test dataset.

The Mean Absolute Error (MAE) of approximately $27.77 indicates that predictions differ from actual listing prices by approximately $27.77 per observation on average.

The Root Mean Squared Error (RMSE) of approximately $39.68 gives greater weight to larger prediction errors.

Test performance was better than validation performance. This suggests encouraging generalization performance, although differences in the difficulty of the validation and test samples may contribute to the results.

The actual mean test price was $122.02, compared with a mean predicted price of $122.50.

Evaluation Reproducibility

The independent evaluation script is:

text
evaluate_test.py

Run it from the repository root using:

bash
python evaluate_test.py

The script loads the existing MLflow model and evaluates the held-out test dataset without retraining.

Evaluation results are saved to:

text
artifacts/test_evaluation.json

The artifacts directory is excluded from Git through .gitignore. The evaluation script and its documented results are maintained in the repository.

Historical Test Prediction Artifact

An earlier pipeline execution generated:

text
src/test_regression_model/test_predictions.csv

That historical prediction artifact contained 9,286 observations. Its metrics are not the results of the November 3 independent evaluation documented above. The current evaluation uses the 9,284 observations in test_data.csv.

Weights & Biases Integration

Weights & Biases was successfully integrated throughout the machine learning pipeline.

Registered and Verified Artifacts
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

The pipeline supports repeatable execution. Scheduled automatic retraining has not been implemented or verified.

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
Git reported a clean working tree at the post-execution verification point.
Git Repository Verification

The repository was checked using:

bash
git status


The previously verified output was:

text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean


The repository has since received additional documentation changes, which are handled through separate Git commits.

Conclusion

This project demonstrates the development and execution of a reusable end-to-end machine learning pipeline for predicting NYC Airbnb listing prices.

By integrating Python, Pandas, NumPy, Scikit-learn, MLflow, Hydra, Conda, and Weights & Biases, the project provides an automated workflow for data acquisition, cleaning, validation, dataset splitting, model training, model export, and evaluation.

The Random Forest model achieved:

Test R²: 0.6913
Test MAE: $27.7701
Test RMSE: $39.6831
Test observations: 9,284

The successful execution of all six pipeline components, completion of seven validation tests, registration of the trained model, and verification of all six W&B artifacts demonstrate that the pipeline operates successfully as an integrated machine learning workflow.

The project establishes a foundation for future model improvements, additional experimentation, and repeatable evaluation of new Airbnb listing data.

Original Project Description

This project is based on the Udacity Build an ML Pipeline for Short-Term Rental Prices in NYC starter project.

The original business scenario focuses on a property management company that receives new rental listing data in bulk every week and needs an automated, reusable machine learning pipeline to process incoming data and retrain its price prediction model regularly.

Original Starter Repository:

https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices

License

See LICENSE.txt for license information.
