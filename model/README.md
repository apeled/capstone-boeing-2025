Comprehensive analysis and prediction system for ordinal ranks generated from multiple decision criteria. The project focuses on identifying key rank drivers, predicting future ranks, applying proper ordinal modeling techniques, handling high dimensionality data, and maintaining interpretability for decision analysis.

## Repository Structure

### Folder Directory

- **enhanced-rank-prediction.ipynb**: Enhanced notebook with additional features and improvements over the base model
- **mcda-rank-prediction.ipynb**: Main notebook containing the Multi-Criteria Decision Analysis and rank prediction methodology
- **model_summary.csv**: Summary of all trained models and their performance metrics
- **Rank.csv**: Original dataset with rank information and drivers

### Data Folder

- **original_dataset.csv**: Original dataset after initial preprocessing
- **pca_features.csv**: Dataset with PCA-reduced features
- **X_train_positional.csv**: Training set with positional feature encoding
- **X_test_positional.csv**: Test set with positional feature encoding
- **X_train_onehot.csv**: Training set with one-hot feature encoding
- **X_test_onehot.csv**: Test set with one-hot feature encoding
- **X_train_pca.csv**: Training set with PCA-reduced features
- **X_test_pca.csv**: Test set with PCA-reduced features
- **y_train.csv**: Target variable (ranks) for training
- **y_test.csv**: Target variable (ranks) for testing

### Saved Models & Features Folder

- **best_rank_prediction_model.pkl**: Best performing model based on evaluation metrics
- **feature_columns.pkl**: Lists of feature columns for different encoding methods
- **gradient_boosting_one-hot.pkl**: Gradient Boosting model trained with one-hot encoding
- **gradient_boosting_pca.pkl**: Gradient Boosting model trained with PCA features
- **gradient_boosting_positional.pkl**: Gradient Boosting model trained with positional encoding
- **lightgbm_one-hot.pkl**: LightGBM model trained with one-hot encoding
- **lightgbm_positional.pkl**: LightGBM model trained with positional encoding
- **mord_logisticit_one-hot.pkl**: Ordinal regression model (Mord LogisticIT) with one-hot encoding

## Methodology

This project implements a comprehensive approach to analyzing and predicting ordinal ranks:

1. **Data Preprocessing**: Cleaning and structuring the raw rank data
2. **Feature Engineering**: Creating meaningful features that capture temporal patterns and driver positions
3. **Dimension Reduction**: Applying PCA to handle high-dimensional one-hot encoded data
4. **Driver Analysis**: Identifying which factors contribute most significantly to rank scores
5. **Model Training**: Experimenting with multiple models including tree-based algorithms and ordinal regression
6. **Evaluation**: Comparing models using metrics appropriate for ordinal prediction tasks
7. **Forecasting**: Predicting future ranks based on historical patterns

## Models Implemented

- Random Forest Regression
- Gradient Boosting Regression
- LightGBM Regression
- XGBoost Regression
- Mord LogisticIT (Ordinal Regression)

Each model is implemented with different feature encoding approaches (positional, one-hot, PCA) to determine the optimal combination of model and feature representation.

## Usage

The main analysis and code are contained in the Jupyter notebooks. To replicate the analysis:

1. Start with `mcda-rank-prediction.ipynb` for the core methodology
2. Explore `enhanced-rank-prediction.ipynb` for additional features and improvements
3. The saved models can be loaded for making new predictions on unseen data

## Data Description

The dataset (`Rank.csv`) contains:
- Rank scores (ordinal target variable)
- Multiple decision criteria (Drivers 1-17)
- Subject IDs for longitudinal tracking
- Timestamps for tracking rank changes over time

The models aim to understand how these drivers influence rank and predict future rank positions.
