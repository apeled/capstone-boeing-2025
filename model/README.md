# MCDA Rank Prediction Project

This folder contains resources and analyses focused on predicting positional rankings using machine learning techniques and Multi-Criteria Decision Analysis (MCDA). It is structured to guide users clearly through our modeling process, from data exploration to predictive modeling and evaluation.

## Contents

### Notebooks

- **`mcda_model_prediction.ipynb`**: Provides a comprehensive workflow for predicting rankings using MCDA. The notebook covers data exploration, preprocessing, extensive feature engineering, hyperparameter optimization, and model evaluation across multiple machine learning algorithms.

- **`enhanced-positional-notebook.ipynb`**: Focuses specifically on advanced positional feature engineering, including PCA transformations and positional encodings. This notebook implements and compares various models (Random Forest, Gradient Boosting, LightGBM, XGBoost) with these enhanced features to improve predictive accuracy.

### Data Outputs (`data_outputs` Folder)

- Contains processed datasets ready for modeling:
  - Training and test datasets (`X_train`, `X_test`, `y_train`, `y_test`)
  - Original and PCA-transformed datasets (`original_dataset.csv`, `pca_features.csv`)
  - Various feature engineering outputs (positional, PCA-based, and one-hot encoded features)

### Saved Models

- **Main Models (`*.pkl` files)**:
  - Gradient Boosting, LightGBM, Random Forest, and XGBoost models trained using enhanced positional features and PCA transformations.

- **`saved_models_mcda_prediction` folder**:
  - Contains the trained models based on different feature sets (one-hot, PCA, positional).
  - Feature metadata (`feature_columns.pkl`) and model-specific encodings to reproduce or extend the analyses.

### Model Evaluation and Navigation

To navigate effectively:

1. Start with `mcda_model_rank_prediction.ipynb` to follow the full modeling workflow.
2. Move to **`positional_models_notebook.ipynb`** for detailed insights into enhanced positional modeling and PCA-based feature transformations.
3. Refer to saved models in the `saved_models_mcda_model_rank_prediction` folder for immediate application or further experimentation.
4. Use datasets in the `data_outputs` folder for consistent data reference across analyses.

This setup facilitates a comprehensive understanding of MCDA rank prediction modeling, including the impact of advanced feature engineering and systematic hyperparameter optimization.

### Folder Structure

```
├── README.md
├── data_outputs
│   ├── X_test_onehot.csv
│   ├── X_test_pca.csv
│   ├── X_test_positional.csv
│   ├── X_train_onehot.csv
│   ├── X_train_pca.csv
│   ├── X_train_positional.csv
│   ├── original_dataset.csv
│   ├── pca_features.csv
│   ├── y_test.csv
│   └── y_train.csv
├── gradient_boosting_enhanced.pkl
├── lightgbm_enhanced.pkl
├── mcda_model_rank_prediction.ipynb
├── model_summary.csv
├── model_summary_enhanced.csv
├── model_summary_enhanced_combined.csv
├── positional_models_notebook.ipynb
├── random_forest_enhanced.pkl
├── requirements.txt
├── xgboost_enhanced.pkl
├── saved_models_mcda_model_rank_prediction
    ├── best_rank_prediction_model.pkl
    ├── feature_columns.pkl
    ├── gradient_boosting_one-hot.pkl
    ├── gradient_boosting_pca.pkl
    ├── gradient_boosting_positional.pkl
    ├── lightgbm_one-hot.pkl
    ├── lightgbm_positional.pkl
    ├── mord_logisticit_one-hot.pkl
    ├── onehot_features.csv
    ├── pca_model.pkl
    ├── positional_features.csv
    ├── random_forest_one-hot.pkl
    ├── random_forest_pca.pkl
    ├── random_forest_positional.pkl
    └── xgboost_positional.pkl
```
