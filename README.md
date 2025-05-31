# Gold Recovery Prediction: Optimizing Mining Operations

## Project Description

This project aims to develop a machine learning model to accurately predict the gold recovery rate from gold ore at different crucial stages of the purification process. The data is obtained from a mining operation and includes various parameters related to the ore composition, processing conditions, and final product quality. Accurate prediction of recovery rates can help optimize the mining process, reduce waste, and improve economic efficiency.

## Gold Recovery Process Overview

The gold recovery process involves two main stages:

1.  **Flotation (Rougher Process):** This is the initial stage where the raw gold ore mixture is treated. The goal here is to separate the rougher gold concentrate (containing a higher percentage of gold) from the rougher tails (waste material).
2.  **Purification (Cleaner Process):** This is a subsequent two-stage process that further refines the rougher concentrate to produce the final, high-grade gold concentrate.

## Prediction Goals

The primary goal is to predict the gold recovery rate at two specific points within this process:

* `rougher.output.recovery`: The recovery rate achieved after the initial flotation stage.
* `final.output.recovery`: The ultimate recovery rate after the final purification stage, representing the overall efficiency of the gold extraction.

## Evaluation Metric

The final evaluation metric for the model will be the **final sMAPE (symmetric Mean Absolute Percentage Error)**. This is a custom, weighted average of the sMAPE calculated for both the rougher concentrate recovery and the final concentrate recovery, providing a balanced measure of prediction accuracy across both critical stages.

The `final sMAPE` is calculated as:
$$
\text{final\_sMAPE} = 0.25 \times \text{sMAPE}(\text{rougher.output.recovery}) + 0.75 \times \text{sMAPE}(\text{final.output.recovery})
$$

Where sMAPE for a single recovery is:
$$
\text{sMAPE} = \frac{1}{N} \sum_{i=1}^{N} \frac{|P_i - A_i|}{(|A_i| + |P_i|) / 2} \times 100\%
$$
* $P_i$: Predicted value
* $A_i$: Actual value
* $N$: Number of samples

## Technologies Used

* **Python**
* **Pandas:** For robust data loading, manipulation, and analysis.
* **NumPy:** For efficient numerical operations and array handling.
* **Scikit-learn:** For machine learning model implementation (Linear Regression, Decision Tree Regressor, RandomForest Regressor), data preprocessing (StandardScaler), feature selection (SelectFromModel), and model evaluation utilities (`cross_val_score`, `make_scorer`).
* **XGBoost (`xgboost`):** For powerful and efficient gradient boosting regression.
* **Matplotlib:** For creating static data visualizations, including trend analysis and model performance plots.
* **Seaborn:** For enhancing the aesthetic quality and informativeness of statistical graphics.
* **Jupyter Notebook:** The primary environment for conducting the analysis, experimentation, and presenting findings.

## Dataset

The project utilizes historical data from a gold mining operation, comprising various parameters related to ore composition, processing conditions, and gold recovery rates.

* **File Names:**
    * `gold_recovery_train.csv`: Training dataset.
    * `gold_recovery_test.csv`: Test dataset.
    * `gold_recovery_full.csv`: Full dataset for initial analysis and understanding feature availability.
* **Content:** The datasets include columns representing input parameters (e.g., feed concentrations, process parameters) and output parameters (e.g., product concentrations, recovery rates).
* **Original Location (on training platform):** `/datasets/gold_recovery_train.csv`, `/datasets/gold_recovery_test.csv`, `/datasets/gold_recovery_full.csv`
