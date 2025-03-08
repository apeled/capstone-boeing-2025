import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional, Any, Union
import sqlite3
import datetime
import random
from data.seed_db import insert_into_rankings, load_sqlite_to_dataframe
import plotly.graph_objects as go


# Define paths according to your repository structure
MODEL_PATH = "model/saved_models_features"
MODEL_SUMMARY_PATH = "model/model_summary.csv"
DATA_PATH = "data/rankings.db"

# Load model summary for model selection
try:
    model_summary = pd.read_csv(MODEL_SUMMARY_PATH)
    print(f"Loaded model summary with {len(model_summary)} models")
except FileNotFoundError:
    print(f"WARNING: Could not find {MODEL_SUMMARY_PATH}. Using default model information.")
    # Create a dummy model summary if file not found
    model_summary = pd.DataFrame({
        'model_name': ['Random Forest (One-Hot)'],
        'filename': ['random_forest_one-hot.pkl'],
        'mae': [17.45],
        'rmse': [25.01],
        'r2': [0.805],
        'spearman': [0.894],
        'kendall': [0.738],
        'accuracy_5': [0.226],
        'accuracy_10': [0.428],
        'accuracy_20': [0.694],
        'is_best_model': [True]
    })
except Exception as e:
    print(f"Error loading model summary: {str(e)}")
    # Create a dummy model summary if any other error occurs
    model_summary = pd.DataFrame({
        'model_name': ['Random Forest (One-Hot)'],
        'filename': ['random_forest_one-hot.pkl'],
        'mae': [17.45],
        'rmse': [25.01],
        'r2': [0.805],
        'spearman': [0.894],
        'kendall': [0.738],
        'accuracy_5': [0.226],
        'accuracy_10': [0.428],
        'accuracy_20': [0.694],
        'is_best_model': [True]
    })

# Determine best model from summary and available model files
best_model_info = model_summary[model_summary['is_best_model'] == True].iloc[0] if not model_summary.empty else None
available_models: Dict[str, str] = {}

def find_model_files() -> Dict[str, str]:
    """Scan model directory for available model files"""
    model_files: Dict[str, str] = {}
    
    try:
        files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pkl') and f != 'pca_model.pkl' and f != 'feature_columns.pkl']
        for file in files:
            # Match with model summary if possible
            model_name = None
            for _, row in model_summary.iterrows():
                if row['filename'] == file:
                    model_name = row['model_name']
                    break
            
            if model_name is None:
                # Use filename as model name if not found in summary
                model_name = file.replace('.pkl', '').replace('_', ' ').title()
            
            model_files[model_name] = os.path.join(MODEL_PATH, file)
        
        print(f"Found {len(model_files)} model files in {MODEL_PATH}")
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing model directory {MODEL_PATH}: {str(e)}")
    except Exception as e:
        print(f"Error scanning model directory {MODEL_PATH}: {str(e)}")
    
    return model_files

available_models = find_model_files()
if not available_models:
    print("WARNING: No model files found. App will not be able to make predictions.")
    # Add a dummy model for UI testing
    available_models = {"Random Forest (One-Hot)": ""}

# Function to load a specific model
def load_model(model_path: str) -> Optional[Any]:
    """Load a model from the specified path"""
    if not model_path:
        return None
        
    try:
        print(f"Attempting to load model from {model_path}")
        with open(model_path, 'rb') as f:
            loaded_object = pickle.load(f)
            
        # Check if the loaded object is a dictionary containing the model
        if isinstance(loaded_object, dict):
            print(f"Loaded a dictionary with keys: {list(loaded_object.keys())}")
            
            # Check for common model keys
            if 'model' in loaded_object:
                print("Found 'model' key in dictionary")
                return loaded_object['model']
            elif 'estimator' in loaded_object:
                print("Found 'estimator' key in dictionary")
                return loaded_object['estimator']
            elif 'classifier' in loaded_object:
                print("Found 'classifier' key in dictionary")
                return loaded_object['classifier']
            elif 'regressor' in loaded_object:
                print("Found 'regressor' key in dictionary")
                return loaded_object['regressor']
            elif any(key.endswith('_model') for key in loaded_object.keys()):
                # Find any key ending with _model
                model_key = next(key for key in loaded_object.keys() if key.endswith('_model'))
                print(f"Found model key: {model_key}")
                return loaded_object[model_key]
            else:
                # Check if any value is a model-like object with a predict method
                for key, value in loaded_object.items():
                    if hasattr(value, 'predict'):
                        print(f"Found model in key: {key}")
                        return value
                
                print("Could not find model in dictionary, returning dictionary itself")
                return loaded_object
        else:
            # If it's already a model object, return it directly
            print(f"Loaded object type: {type(loaded_object).__name__}")
            return loaded_object
            
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None

# Helper function to make predictions with different model types
def make_prediction_with_model(model, X):
    """
    Try different approaches to make a prediction with the model
    """
    # If model is None, return an error
    if model is None:
        raise ValueError("Model is None")
        
    # If model is a dictionary, try to find a way to use it
    if isinstance(model, dict):
        print(f"Model is a dictionary with keys: {list(model.keys())}")
        
        # If it has a 'predict' function directly (some libraries wrap models this way)
        if callable(getattr(model, 'predict', None)):
            return model.predict(X)
        
        # Check for common model keys
        if 'model' in model and hasattr(model['model'], 'predict'):
            return model['model'].predict(X)
        elif 'estimator' in model and hasattr(model['estimator'], 'predict'):
            return model['estimator'].predict(X)
        elif 'best_estimator_' in model and hasattr(model['best_estimator_'], 'predict'):
            return model['best_estimator_'].predict(X)
        elif 'best_model' in model and hasattr(model['best_model'], 'predict'):
            return model['best_model'].predict(X)
        
        # Look for any key that contains a model with predict method
        for key, value in model.items():
            if hasattr(value, 'predict'):
                print(f"Using model found in key: {key}")
                return value.predict(X)
                
        # If the dictionary contains prediction data directly
        if 'predictions' in model:
            print("Using precomputed predictions")
            return model['predictions']
            
        # If we have feature importances, we might be able to use them
        if 'feature_importances' in model and 'raw_predictions' in model:
            print("No predict method found but dictionary contains prediction data")
            # This is a very simple fallback for decision trees
            if 'raw_predictions' in model and isinstance(model['raw_predictions'], np.ndarray):
                return np.mean(model['raw_predictions'], axis=0)
        
        raise ValueError(f"Could not find a way to predict with dictionary model: {list(model.keys())}")
    
    # Standard case: model has predict method
    if hasattr(model, 'predict'):
        return model.predict(X)
    
    # If model is a sklearn Pipeline
    if hasattr(model, 'steps') and isinstance(getattr(model, 'steps', None), list):
        final_step = model.steps[-1][1]
        if hasattr(final_step, 'predict'):
            print("Using final step of pipeline")
            return final_step.predict(X)
    
    # If all else fails
    raise ValueError(f"Model type {type(model).__name__} does not support prediction")

# Load the default model (best model or first available)
default_model_name = None
if best_model_info is not None:
    default_model_name = best_model_info['model_name']
elif available_models:
    default_model_name = next(iter(available_models.keys()))

default_model_path = available_models.get(default_model_name, "") if default_model_name else ""
current_model = load_model(default_model_path) if default_model_path else None

# Try to load scaler (you might not have one, in which case we'll use a new one)
scaler = None
try:
    # Try to find a scaler in your model directory
    scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
except Exception as e:
    print(f"No scaler found: {str(e)}")

if scaler is None:
    print("Creating a new StandardScaler")
    scaler = StandardScaler()

# Load dataset for reference data and one-hot encoding
df = pd.DataFrame()
try:
    df = load_sqlite_to_dataframe(DATA_PATH)
    print(f"Loaded data from {DATA_PATH} with {len(df)} rows")
except FileNotFoundError:
    print(f"WARNING: Could not load {DATA_PATH}. Using empty DataFrame.")
except Exception as e:
    print(f"Error loading {DATA_PATH}: {str(e)}")

# Load subject IDs from reference data if available
subject_ids = []
if not df.empty and 'SubjectID' in df.columns:
    subject_ids = sorted(df['SubjectID'].unique().tolist())
    print(f"Loaded {len(subject_ids)} subject IDs from reference data")
print(f"DATA_PATH: {DATA_PATH}")
print(f"CSV exists: {os.path.exists(DATA_PATH)}")
if not df.empty:
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"First few subject IDs: {df['SubjectID'].head().tolist() if 'SubjectID' in df.columns else 'No SubjectID column'}")


# Process reference data if available
all_factors = []
if not df.empty:
    # Convert date column if exists
    if 'UpdateDT' in df.columns:
        try:
            df['UpdateDT'] = pd.to_datetime(df['UpdateDT'])
        except Exception as e:
            print(f"Error converting UpdateDT to datetime: {str(e)}")
    
    # Get driver columns (adjust based on your data structure)
    driver_columns = [col for col in df.columns if col.startswith('Driver') and col.replace('Driver', '').isdigit()]
    if not driver_columns and len(df.columns) > 2:
        # Fallback: assume columns 2-19 are driver columns if naming pattern not found
        driver_columns = df.columns[2:19].tolist()
    
    # Get all unique factors from reference data
    all_factors = set()
    for col in driver_columns:
        if col in df.columns:
            all_factors.update([f for f in df[col].dropna().unique() if f])
    all_factors = sorted(list(all_factors))
    print(f"Found {len(all_factors)} unique factors across all drivers")
else:
    # Default driver columns if no data
    driver_columns = [f'Driver{i}' for i in range(1, 18)]
    all_factors = ["Factor1", "Factor2", "Factor3", "Factor4", "Factor5"]  # Default sample factors

# Function to get previous rank data for a subject
def get_subject_history(subject_id, reference_df):
    """
    Get previous rank information for a specific subject ID
    
    Args:
        subject_id: The subject ID to look up
        reference_df: The reference dataset
        
    Returns:
        Dict containing previous rank and other historical data
    """
    if reference_df.empty or 'SubjectID' not in reference_df.columns:
        return None
    
    # Filter for this subject
    subject_data = reference_df[reference_df['SubjectID'] == subject_id]
    
    if subject_data.empty:
        return None
    
    # Sort by date to get most recent entry
    if 'UpdateDT' in subject_data.columns:
        try:
            subject_data = subject_data.sort_values('UpdateDT', ascending=False)
        except:
            pass  # If sorting fails, continue without sorting
    
    # Get most recent record
    latest_record = subject_data.iloc[0]
    
    # Collect relevant history data
    history = {
        'previous_rank': latest_record['Rank'] if 'Rank' in latest_record else None,
        'last_update': latest_record['UpdateDT'] if 'UpdateDT' in latest_record else None,
        'num_records': len(subject_data),
        'avg_rank': subject_data['Rank'].mean() if 'Rank' in subject_data.columns else None,
        'min_rank': subject_data['Rank'].min() if 'Rank' in subject_data.columns else None,
        'max_rank': subject_data['Rank'].max() if 'Rank' in subject_data.columns else None,
    }
    
    return history

def engineer_features(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform feature engineering similar to your engineer_features function
    
    Args:
        input_df: DataFrame with user input
        reference_df: Original training data for reference
        
    Returns:
        Tuple of processed DataFrames (positional, one-hot)
    """
    print("Engineering features for prediction...")
    
    # Create a copy to avoid modifying the original
    processed_df = input_df.copy()
    
    # Add required columns that might be missing
    if 'SubjectID' not in processed_df.columns or processed_df['SubjectID'].iloc[0] == "":
        processed_df['SubjectID'] = 'new_input'
    
    processed_df['UpdateDT'] = datetime.now()
    processed_df['UpdateDT'] = pd.to_datetime(processed_df['UpdateDT'])
    
    # 1. Add time-based features
    processed_df['Year'] = processed_df['UpdateDT'].dt.year
    processed_df['Month'] = processed_df['UpdateDT'].dt.month
    processed_df['DayOfMonth'] = processed_df['UpdateDT'].dt.day
    processed_df['DayOfWeek'] = processed_df['UpdateDT'].dt.dayofweek
    
    # Days since first observation
    min_date = reference_df['UpdateDT'].min() if 'UpdateDT' in reference_df.columns else processed_df['UpdateDT'].min()
    processed_df['DaysSinceFirst'] = (processed_df['UpdateDT'] - pd.to_datetime(min_date)).dt.days
    
    # Days since previous observation - for new input this will be 0
    processed_df['DaysSincePrev'] = 0
    
    # 2. Add lag features
    if 'PrevRank' in processed_df.columns:
        # Use the provided previous rank
        print(f"Using provided PrevRank: {processed_df['PrevRank'].iloc[0]}")
    else:
        # Use average rank from reference data as fallback
        avg_rank = reference_df['Rank'].mean() if 'Rank' in reference_df.columns else 50
        processed_df['PrevRank'] = avg_rank
        print(f"Using average rank as PrevRank: {avg_rank}")
    
    # Calculate rank change (will be 0 since we don't know new rank yet)
    processed_df['RankChange'] = 0
    
    # 3. Create features based on driver positions
    # Get all unique factors from reference data and input data
    all_factors_combined = set()
    
    # Add factors from reference data
    for col in driver_columns:
        if col in reference_df.columns:
            all_factors_combined.update([f for f in reference_df[col].dropna().unique() if f])
    
    # Add factors from input data
    for col in driver_columns:
        if col in processed_df.columns:
            all_factors_combined.update([f for f in processed_df[col].dropna().unique() if f])
    
    print(f"Found {len(all_factors_combined)} unique factors for feature engineering")
    
    # For each factor, create a position feature
    for factor in all_factors_combined:
        # Initialize position as 0 (not present)
        processed_df[f'{factor}_Position'] = 0
        
        # Update position for each driver where this factor appears
        for i, col in enumerate(driver_columns, 1):
            if col in processed_df.columns:
                processed_df.loc[processed_df[col] == factor, f'{factor}_Position'] = i
    
    # Drop original driver columns as we now have positional features
    processed_df_no_drivers = processed_df.drop(columns=driver_columns)
    
    # Create a version with one-hot encoding for models that use it
    processed_df_onehot = pd.get_dummies(
        processed_df, 
        columns=driver_columns, 
        prefix=driver_columns,
        drop_first=False
    )
    
    # Fill missing dummy columns based on reference data
    if 'Rank' in reference_df.columns:
        # Try to load feature columns from saved file if available
        feature_columns = []
        try:
            feature_cols_path = os.path.join(MODEL_PATH, "feature_columns.pkl")
            if os.path.exists(feature_cols_path):
                with open(feature_cols_path, 'rb') as f:
                    feature_columns = pickle.load(f)
                    print(f"Loaded {len(feature_columns)} feature columns from {feature_cols_path}")
        except Exception as e:
            print(f"Could not load feature columns: {str(e)}")
        
        # If we have saved feature columns, use them
        if feature_columns:
            for col in feature_columns:
                if col not in processed_df_onehot.columns:
                    processed_df_onehot[col] = 0
        else:
            # Find all one-hot columns from reference data
            ref_onehot_cols = []
            for col in reference_df.columns:
                if any(col.startswith(f"{driver}_") for driver in driver_columns):
                    ref_onehot_cols.append(col)
            
            # Add missing one-hot columns
            for col in ref_onehot_cols:
                if col not in processed_df_onehot.columns:
                    processed_df_onehot[col] = 0
    
    # Apply standardization to numerical features
    numeric_cols = processed_df_onehot.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Exclude one-hot encoded columns from standardization
    numeric_non_onehot = [col for col in numeric_cols 
                         if not any(col.startswith(f"{driver}_") for driver in driver_columns)]
    
    if numeric_non_onehot and scaler is not None:
        try:
            # Transform only non-one-hot numeric columns
            processed_df_onehot[numeric_non_onehot] = scaler.transform(
                processed_df_onehot[numeric_non_onehot].values.reshape(1, -1)
            )
        except Exception as e:
            print(f"Error applying scaler: {str(e)}")
            # Fallback to simple standardization if transform fails
            for col in numeric_non_onehot:
                try:
                    mean = reference_df[col].mean() if col in reference_df.columns else processed_df_onehot[col].mean()
                    std = reference_df[col].std() if col in reference_df.columns else processed_df_onehot[col].std()
                    if std > 0:
                        processed_df_onehot[col] = (processed_df_onehot[col] - mean) / std
                except Exception as e2:
                    print(f"Error standardizing column {col}: {str(e2)}")
    
    return processed_df_no_drivers, processed_df_onehot

def get_model_features(model_name: str) -> str:
    """
    Determine which processed dataframe to use based on model name
    """
    if 'One-Hot' in model_name:
        return 'onehot'
    elif 'Positional' in model_name:
        return 'positional'
    elif 'PCA' in model_name:
        # PCA models would require the additional PCA transformation
        # For now, default to one-hot as most models use it
        return 'onehot'
    else:
        # Default to one-hot if unsure
        return 'onehot'

# Helper function to extract feature importances
def get_feature_importances(model, feature_names):
    """
    Extract feature importances from the model if available
    
    Args:
        model: The trained model
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature names and importance scores, sorted by importance
    """
    importances = None
    
    try:
        # For tree-based models that have feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        
        # For linear models that have coef_ attribute
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
        
        # For dictionary-stored models
        elif isinstance(model, dict):
            # Check if feature importances are stored directly
            if 'feature_importances_' in model:
                importances = model['feature_importances_']
            elif 'feature_importances' in model:
                importances = model['feature_importances']
            # Try to find a model inside the dictionary
            else:
                for key, value in model.items():
                    if hasattr(value, 'feature_importances_'):
                        importances = value.feature_importances_
                        break
                    elif hasattr(value, 'coef_'):
                        importances = np.abs(value.coef_)
                        if importances.ndim > 1:
                            importances = importances.mean(axis=0)
                        break
        
        # If we found importances, create a DataFrame
        if importances is not None:
            # Make sure lengths match
            if len(importances) != len(feature_names):
                print(f"Warning: Length mismatch between importances ({len(importances)}) and feature names ({len(feature_names)})")
                # Use only the number of features we have importances for
                min_len = min(len(importances), len(feature_names))
                importances = importances[:min_len]
                feature_names = feature_names[:min_len]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            return importance_df
    
    except Exception as e:
        print(f"Error extracting feature importances: {str(e)}")
    
    # If all methods fail, return None
    return None

# Function to create a feature importance chart
def create_importance_chart(model, X_pred, selected_factors, approach_used):
    """
    Create a chart showing the most influential features for the prediction
    
    Args:
        model: The model used for prediction
        X_pred: The feature DataFrame used for prediction
        selected_factors: List of factors selected by the user
        approach_used: String describing which prediction approach was used
    
    Returns:
        Plotly figure showing feature importances
    """
    # Get feature names from the DataFrame
    feature_names = X_pred.columns.tolist()
    
    # Try to get feature importances
    importance_df = get_feature_importances(model, feature_names)
    
    if importance_df is None:
        # If we couldn't get importances, create a basic chart of selected factors
        return px.bar(
            x=selected_factors,
            y=[1] * len(selected_factors),
            labels={"x": "Selected Factors", "y": "Count"},
            title=f"Selected Factors (Feature importance not available for {approach_used})"
        )
    
    # Filter for just the features related to the driver factors
    factor_importance_df = importance_df[importance_df['Feature'].str.contains('|'.join(driver_columns))]
    
    # Get the top 10 most important features
    top_features = factor_importance_df.head(10).copy()
    
    # Clean up feature names for display (remove the "Driver1_" prefix, etc.)
    top_features['Display_Name'] = top_features['Feature'].apply(
        lambda x: x.split('_', 1)[1] if '_' in x else x
    )
    
    # Create importance plot
    fig = px.bar(
        top_features,
        x='Importance',
        y='Display_Name',
        orientation='h',
        labels={'Importance': 'Importance Score', 'Display_Name': 'Factor'},
        title='Most Influential Factors for Prediction',
        color='Importance',
        color_continuous_scale='blues',
    )
    
    # Improve layout
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
    )
    
    return fig

# Define the model selector dropdown
model_selector = html.Div([
    html.Label("Select Prediction Model", className="fw-bold mb-2"),
    dcc.Dropdown(
        id="model-selector",
        options=[{"label": "None", "value": "None"}] + [{"label": name, "value": name} for name in available_models.keys()],
        # value=default_model_name,
        value="None",
        clearable=False,
        className="mb-3"
    ),
    html.Div(id="selected-model-info", className="small text-muted")
])

# Create datetime selector
date_time_ui = html.Div([
    dbc.Row(
        dcc.DatePickerSingle(
            id="date-picker",
            date=datetime.datetime.now().date(),
            display_format="YYYY-MM-DD",
        ),
    ),
    dbc.Row(
        dbc.Col(
            dcc.Input(
                id="time-input",
                type="text",
                value=datetime.datetime.now().strftime("%H:%M"),
                placeholder="HH:MM",
            ),
            width=6
        ),
    )
], style={"display": "none"}, id="date-time-container")

# Create subject ID selector
subject_id_selector = html.Div([
    html.Label("Select Subject ID", className="fw-bold mb-2"),
    dcc.Dropdown(
        id="subject-id-selector",
        options=[{"label": sid, "value": sid} for sid in subject_ids] + [{"label": "New Subject", "value": "new_subject"}],
        value=subject_ids[0] if subject_ids else "new_subject",
        clearable=False,
        className="mb-3"
    ),
    # Add new subject ID input field (will be shown/hidden based on selection)
    html.Div([
        html.Label("Enter New Subject ID", className="fw-bold mb-2"),
        dbc.Input(id="new-subject-id-input", type="text", placeholder="Enter new Subject ID", className="mb-3"),
    ], id="new-subject-container", style={"display": "none"}),
])

import plotly.graph_objects as go
import plotly.express as px

# Function to create a history chart for a subject
def create_subject_history_chart(subject_id, reference_df):
    """
    Create a line chart showing the rank history of a subject, with separate traces for each model and historical data
    
    Args:
        subject_id: The subject ID to look up
        reference_df: The reference dataset
    
    Returns:
        Plotly figure showing rank history
    """
    if reference_df.empty or 'SubjectID' not in reference_df.columns:
        return go.Figure(data=[go.Scatter(x=[], y=[], mode='lines', name="No history data available")])
    
    # Filter for this subject
    subject_data = reference_df[reference_df['SubjectID'] == subject_id]
    
    if subject_data.empty or 'Rank' not in subject_data.columns:
        return go.Figure(data=[go.Scatter(x=[], y=[], mode='lines', name="No rank history available for this subject")])
    
    # Ensure we have dates
    if 'UpdateDT' in subject_data.columns:
        try:
            # Sort by date
            subject_data = subject_data.sort_values('UpdateDT')
            
            # Create a list to store traces
            traces = []
            
            # Add the Historical Data trace for rows with empty 'Model' value
            historical_data = subject_data[subject_data['Model'].isna()]
            if not historical_data.empty:
                historical_trace = go.Scatter(
                    x=historical_data['UpdateDT'],
                    y=historical_data['Rank'],
                    mode='lines+markers',
                    name="Historical Data",
                    line=dict(color='gray'),  # A neutral color for historical data
                    marker=dict(symbol='circle')
                )
                traces.append(historical_trace)
            
            # Use a color scale to ensure distinct colors
            color_scale = px.colors.qualitative.Set1  # Set1 provides distinct colors
            
            # Loop through all unique model values (excluding NaN) and create a trace for each
            for i, model in enumerate(subject_data['Model'].dropna().unique()):
                model_data = subject_data[subject_data['Model'] == model]
                
                # Assign a color from the color scale
                model_color = color_scale[i % len(color_scale)]
                
                # Add a trace for the current model
                model_trace = go.Scatter(
                    x=model_data['UpdateDT'],
                    y=model_data['Rank'],
                    mode='lines+markers',
                    name=f"Model: {model}",
                    line=dict(color=model_color),
                    marker=dict(symbol='diamond')
                )
                traces.append(model_trace)
            
            # Create the figure with all traces
            fig = go.Figure(data=traces)
            
            # Invert y-axis so that higher ranks (lower numbers) are at the top
            fig.update_layout(
                yaxis_autorange="reversed",
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10),
                height=200,
                title=f"Rank History for Subject {subject_id}",
                xaxis_title="Date",
                yaxis_title="Rank"
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating history chart: {str(e)}")
            return go.Figure(data=[go.Scatter(x=[], y=[], mode='lines', name=f"Error: {str(e)}")])
    else:
        # No date column
        return go.Figure(data=[go.Scatter(x=[], y=[], mode='lines', name="Cannot create history chart: missing date information")])


    
# Create improved factor selection interface
def create_factor_selectors():
    """
    Create a set of improved factor selection dropdowns for each driver
    """
    selectors = []
    
    for i, driver in enumerate(driver_columns):
        selectors.append(
            dbc.Col([
                html.Label(driver, className="fw-bold mb-1 small"),
                dcc.Dropdown(
                    id={"type": "factor-dropdown", "index": i},
                    options=[{"label": factor, "value": factor} for factor in all_factors],
                    value="",
                    clearable=True,
                    placeholder="Select factor",
                    className="mb-3",
                )
            ], md=3, lg=2, className="mb-2")
        )
    
    # Group selectors into rows of 6 columns each
    rows = []
    for i in range(0, len(selectors), 6):
        rows.append(dbc.Row(selectors[i:i+6], className="mb-2"))
    
    return html.Div(rows)

# Define the "Rank Prediction" tab content
rank_prediction_tab = dbc.Container([
    # Top section with model and subject selection
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Model selector
                dbc.Col([
                    model_selector
                ], md=6, className="border-end"),
                
                # Subject selector
                dbc.Col([
                    subject_id_selector,
                    dbc.Checkbox(
                        id="toggle-checkbox",
                        label="Show Date and Time Input (Store in Database)",
                        className="mb-2"
                    ),
                    date_time_ui
                ], md=6),
            ]),
        ])
    ], className="shadow-sm mb-4"),
    
    # Subject history section (moved below model/subject selection)
    html.Div(
        id="subject-history-container",
        style={"display": "none"},
        children=[
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="bi bi-clock-history me-2"),
                        "Subject History"
                    ], className="mb-0"),
                ]),
                dbc.CardBody([
                    dbc.Row([
                        # Subject history information
                        dbc.Col([
                            html.Div(id="subject-history-content"),
                        ], md=5),
                        
                        # Subject history chart
                        dbc.Col([
                            dcc.Graph(id="subject-history-chart", config={'displayModeBar': False})
                        ], md=7),
                    ]),
                ]),
            ], className="shadow-sm mb-4")
        ]
    ),
    
    # Decision factors section
    dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="bi bi-list-check me-2"),
                "Input Decision Factors"
            ], className="mb-0"),
            html.P("Select a unique factor for each driver position to predict rank", 
                  className="text-muted mb-0 small"),
        ]),
        dbc.CardBody([
            # Factor selectors
            create_factor_selectors(),
            
            # Action buttons
            html.Div([
                dbc.Button([
                    html.I(className="bi bi-calculator me-2"),
                    "Predict Rank"
                ], id="predict-btn", color="primary", size="lg", className="mt-3"),
                dbc.Button([
                    html.I(className="bi bi-shuffle me-2"),
                    "Randomize Factors"
                ], id="randomize-btn", color="warning", size="lg", className="mt-3 ms-2"),
            ], className="d-flex justify-content-center mt-4")
        ]),
    ], className="shadow-sm mb-4"),
    
    # Results section
    dbc.Row([
        # Prediction result
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="bi bi-graph-up-arrow me-2"),
                        "Prediction Result"
                    ], className="mb-0"),
                ]),
                dbc.CardBody([
                    html.Div([
                        html.Div(id="prediction-output", 
                                className="display-4 text-center mb-3"),
                        html.Div(id="prediction-confidence", 
                                className="text-muted text-center"),
                    ], className="py-4 text-center"),
                ], style={"min-height": "300px"}),
            ], className="shadow-sm h-100"),
        ], md=4),
        
        # Influential factors
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="bi bi-bar-chart-fill me-2"),
                        "Influential Factors"
                    ], className="mb-0"),
                    html.P("Factors with the most impact on prediction", 
                          className="text-muted mb-0 small"),
                ]),
                dbc.CardBody([
                    dcc.Graph(id="factor-distribution", style={"height": "300px"},
                             figure={"layout": {"title": "Make a prediction to see influential factors"}}),
                ]),
            ], className="shadow-sm h-100"),
        ], md=8),
    ]),
], className="py-3")

def get_model_description(model_name: str) -> str:
    """Get description of model based on its name"""
    if 'One-Hot' in model_name:
        return "This model uses one-hot encoding for categorical factors, which means each driver-factor combination is treated as a separate binary feature."
    elif 'Positional' in model_name:
        return "This model uses positional encoding, where the position of each factor in the driver list is used as a numerical feature."
    elif 'PCA' in model_name:
        return "This model uses Principal Component Analysis (PCA) to reduce the dimensionality of the feature space before training."
    else:
        return "This model uses a custom feature engineering approach based on the input factors."

# Define the "Model Information" tab content
def create_model_info_tab(model_name: str) -> html.Div:
    """Create model info tab content based on selected model"""
    # Get model metrics from model_summary
    model_row = model_summary[model_summary['model_name'] == model_name]
    
    if model_row.empty:
        # Default info if model not found in summary
        return dbc.Container([
            html.H2("Model Information"),
            html.P(f"Information for model '{model_name}' is not available in the model summary."),
            html.P("Please select a different model or contact the system administrator."),
        ], className="py-4")
    
    model_info = model_row.iloc[0]
    
    return dbc.Container([
        dbc.Card([
            dbc.CardHeader([
                html.H3(model_name, className="mb-0"),
                html.P("Model Performance Metrics", className="text-muted mb-0 small")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Error Metrics", className="border-bottom pb-2"),
                        dbc.Table([
                            html.Tbody([
                                html.Tr([html.Td("Mean Absolute Error (MAE)"), html.Td(f"{model_info['mae']:.2f}")]),
                                html.Tr([html.Td("Root Mean Squared Error (RMSE)"), html.Td(f"{model_info['rmse']:.2f}")]),
                            ])
                        ], bordered=False, hover=True, size="sm", className="mb-4"),
                        
                        html.H5("Correlation Metrics", className="border-bottom pb-2"),
                        dbc.Table([
                            html.Tbody([
                                html.Tr([html.Td("R² Score"), html.Td(f"{model_info['r2']:.3f}")]),
                                html.Tr([html.Td("Spearman Correlation"), html.Td(f"{model_info['spearman']:.3f}")]),
                                html.Tr([html.Td("Kendall Correlation"), html.Td(f"{model_info['kendall']:.3f}")]),
                            ])
                        ], bordered=False, hover=True, size="sm"),
                    ], md=6),
                    
                    dbc.Col([
                        html.H5("Accuracy Metrics", className="border-bottom pb-2"),
                        html.Div([
                            dcc.Graph(
                                figure={
                                    'data': [
                                        {
                                            'x': ['Within 5', 'Within 10', 'Within 20'],
                                            'y': [
                                                model_info['accuracy_5'] * 100,
                                                model_info['accuracy_10'] * 100,
                                                model_info['accuracy_20'] * 100
                                            ],
                                            'type': 'bar',
                                            'marker': {'color': '#007bff'},
                                        }
                                    ],
                                    'layout': {
                                        'title': 'Accuracy Percentages',
                                        'xaxis': {'title': 'Rank Range'},
                                        'yaxis': {'title': 'Accuracy (%)', 'range': [0, 100]},
                                        'margin': {'l': 40, 'r': 20, 't': 40, 'b': 30},
                                    }
                                },
                                config={'displayModeBar': False},
                                style={'height': '250px'}
                            ),
                            html.P("Shows accuracy within 5, 10, and 20 ranks", className="text-muted text-center small"),
                        ]),
                    ], md=6),
                ]),
                
                html.H5("Model Description", className="border-bottom pb-2 mt-4"),
                html.P([
                    f"This {model_name} model was trained on historical ranking data with "
                    f"multiple driver factors. It is designed to predict the rank based on "
                    f"the selected decision factors.",
                    html.Br(),
                    "The model was evaluated using a time-based split of the data, with the most "
                    "recent data used for testing.",
                ]),
                
                html.Div([
                    html.H5("Feature Type", className="border-bottom pb-2"),
                    html.P(get_model_description(model_name)),
                ], className="mt-4"),
            ]),
        ], className="shadow-sm"),
    ], className="py-4")

# Define the "About Me" tab content
about_me_tab = dbc.Container([
    dbc.Card([
        dbc.CardHeader([
            html.H3("About RankLab", className="mb-0"),
        ]),
        dbc.CardBody([
            html.P("RankLab is an interactive web application designed for Multi-Criteria Decision Analysis Rank Prediction."),
            html.P("This app allows users to enter decision factors and predict a rank based on historical data."),
            html.P("Developed using Python (Dash, Plotly, Pandas) with machine learning models for predictions."),
            
            html.Hr(),
            
            html.H5("Contributors", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Mark Rarlston Daniel", className="card-title"),
                            html.Img(src="/assets/mark.png", height="250px"),
                        ]),
                    ], className="shadow-sm h-100"),
                ], md=4, className="mb-3"),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Amit Peled", className="card-title"),
                            html.Img(src="/assets/amit.png", height="250px"),
                        ]),
                    ], className="shadow-sm h-100"),
                ], md=4, className="mb-3"),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Jake Flynn", className="card-title"),
                            html.Img(src="/assets/jake.png", height="250px"),
                        ]),
                    ], className="shadow-sm h-100"),
                ], md=4, className="mb-3"),
            ]),
            
            html.Hr(),
            
            html.H5("Technical Details", className="mb-3"),
            html.P([
                "Frontend: Dash, Dash Bootstrap Components, Plotly",
                html.Br(),
                "Backend: Python, scikit-learn",
                html.Br(),
                "Data Processing: Pandas, NumPy",
                html.Br(),
                "Visualization: Plotly, Dash AG Grid",
                html.Br(),
                "Deployment: Docker, Gunicorn"
            ]),
        ]),
    ], className="shadow-sm"),
], className="py-4")

# App Layout with Title & Tabs
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP
    ],
    suppress_callback_exceptions=True
)
app.title = "RankLab: Rank Prediction Tool"

app.layout = dbc.Container([
    # Dummy output used to trigger the second callback
    html.Div(id='dummy-output', style={'display': 'none'}),

    # Header with Logo and Title
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("RankLab", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
            ]),
            html.P("Multi-Criteria Decision Analysis Rank Prediction", className="text-muted"),
        ]),
    ], className="border-bottom pb-3"),
    
    # Tabs with shadow effect and rounded corners
    html.Div([
        dbc.Tabs([
            dbc.Tab(rank_prediction_tab, label="Rank Prediction", tab_id="tab-1", 
                   label_style={"fontWeight": "bold"}, 
                   active_label_style={"color": "#007bff"}),
            dbc.Tab(html.Div(id="model-info-tab"), label="Model Information", tab_id="tab-2",
                   label_style={"fontWeight": "bold"},
                   active_label_style={"color": "#007bff"}),
            dbc.Tab(about_me_tab, label="About Me", tab_id="tab-3",
                   label_style={"fontWeight": "bold"},
                   active_label_style={"color": "#007bff"}),
        ], id="tabs", active_tab="tab-1"),
    ], className="shadow-sm rounded bg-white p-3 mb-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.P("© 2025 RankLab - All Rights Reserved", className="text-center text-muted small mt-4"),
        ]),
    ], className="border-top pt-3"),
], fluid=True, className="px-4 py-3 bg-light min-vh-100")


# Callback to reset the input dropdowns
@app.callback(
    [Output({"type": "factor-dropdown", "index": idx}, "value") for idx in range(len(driver_columns))],
    Input("randomize-btn", "n_clicks"),
    prevent_initial_call=True
)
def randomize_factors(n_clicks):
    factor_options = [f"Factor{i}" for i in range(1, 18)]  # Factor1 to Factor17
    shuffled_factors = random.sample(factor_options, len(factor_options))  # Ensure unique random values
    return shuffled_factors  # Assign unique values to all dropdowns

# Callback to show/hide datetime input
@app.callback(
    Output("date-time-container", "style"),
    [Input("toggle-checkbox", "value")]
)
def toggle_new_date_time_input(selected: bool) -> Dict:
    """Show new subject input field when 'New Subject' is selected"""
    if selected:
        return {"display": "block"}
    return {"display": "none"}

# Callback to update model information based on selected model
@app.callback(
    Output("model-info-tab", "children"),
    [Input("model-selector", "value")]
)
def update_model_info_tab(selected_model: str) -> html.Div:
    """Update the model info tab based on selected model"""
    if not selected_model:
        return html.Div("No model selected")
    return create_model_info_tab(selected_model)

# Callback to update selected model info text
@app.callback(
    Output("selected-model-info", "children"),
    [Input("model-selector", "value")]
)
def update_selected_model_info(selected_model: str) -> Union[str, List]:
    """Show brief info about selected model"""
    if not selected_model:
        return "No model selected"
        
    model_row = model_summary[model_summary['model_name'] == selected_model]
    if model_row.empty:
        return f"Using {selected_model}"
    
    info = model_row.iloc[0]
    return [
        f"MAE: {info['mae']:.2f} | R²: {info['r2']:.3f}", html.Br(),
        f"Accuracy within 10 ranks: {info['accuracy_10']*100:.1f}%"
    ]

# Callback to update subject history display
@app.callback(
    [Output("subject-history-container", "style"),
     Output("subject-history-content", "children"),
     Output("subject-history-chart", "figure")],
    [Input("subject-id-selector", "value"),
     Input("model-selector", "value"),
     Input('dummy-output', 'children')]
)
def update_subject_history(selected_subject, selected_model, dummy_output):
    """Show subject history when a subject is selected"""
    if selected_model != "None":
        filtered_df = df[df['Model'] == selected_model]
    else:
        filtered_df = df

    if not selected_subject or selected_subject == "new_subject":
        empty_fig = px.line(title="No history available")
        empty_fig.update_layout(height=200)
        return {"display": "none"}, "", empty_fig
    
    # Get subject history
    history = get_subject_history(selected_subject, filtered_df)
    
    if not history:
        empty_fig = px.line(title="No history available")
        empty_fig.update_layout(height=200)
        return {"display": "block"}, "No history found for this subject.", empty_fig
    
    # Create history display with improved formatting
    content = [
        html.Div([
            html.P([
                html.Strong("Previous Rank: "), 
                html.Span(
                    f"{history['previous_rank']}" if history['previous_rank'] is not None else "Not available",
                    className="fs-4 text-primary"
                )
            ], className="mb-2"),
            
            html.P([
                html.Strong("Last Update: "), 
                f"{history['last_update'].strftime('%Y-%m-%d')}" if history['last_update'] is not None else "Not available"
            ], className="mb-2"),
            
            html.P([
                html.Strong("History: "), 
                f"{history['num_records']} records"
            ], className="mb-2"),
            
            dbc.Row([
                dbc.Col(html.P([
                    html.Strong("Average: "), 
                    html.Span(
                        f"{history['avg_rank']:.1f}" if history['avg_rank'] is not None else "N/A",
                        className="text-info"
                    )
                ]), width=4),
                dbc.Col(html.P([
                    html.Strong("Min: "), 
                    html.Span(
                        f"{history['min_rank']}" if history['min_rank'] is not None else "N/A",
                        className="text-success"
                    )
                ]), width=4),
                dbc.Col(html.P([
                    html.Strong("Max: "), 
                    html.Span(
                        f"{history['max_rank']}" if history['max_rank'] is not None else "N/A",
                        className="text-danger"
                    )
                ]), width=4),
            ], className="mb-2"),
            
            html.P(
                "This information will be used in the prediction.",
                className="text-muted small mt-2 fst-italic"
            ),
        ], className="h-100 d-flex flex-column justify-content-center")
    ]
    
    # Create history chart
    history_chart = create_subject_history_chart(selected_subject, df)
    
    return {"display": "block"}, content, history_chart

# Callback to show/hide new subject ID input field
@app.callback(
    Output("new-subject-container", "style"),
    [Input("subject-id-selector", "value")]
)
def toggle_new_subject_input(selected_subject):
    """Show new subject input field when 'New Subject' is selected"""
    if selected_subject == "new_subject":
        return {"display": "block"}
    return {"display": "none"}

@app.callback(
    [Output({"type": "factor-dropdown", "index": ALL}, "options")],
    [Input({"type": "factor-dropdown", "index": ALL}, "value")],
    [State({"type": "factor-dropdown", "index": ALL}, "id")]
)
def update_dropdown_options(selected_values, dropdown_ids):
    """
    Update dropdown options to hide factors that have already been selected.
    This ensures each factor can only be selected once across all dropdowns.
    """
    # List of all possible factors
    all_factors = [f"Factor{i}" for i in range(1, 18)]
    
    # Get the set of selected values, ignoring None values
    selected_values_set = set(filter(None, selected_values))
    
    # Generate updated options for each dropdown
    updated_options = []
    for dropdown_id, selected_value in zip(dropdown_ids, selected_values):
        # Available options exclude the selected values from other dropdowns
        available_options = [factor for factor in all_factors if factor not in selected_values_set or factor == selected_value]
        updated_options.append([{"label": factor, "value": factor} for factor in available_options])
    
    return [updated_options]

# # Callback to update dropdown options based on already selected factors
# @app.callback(
#     [Output({"type": "factor-dropdown", "index": ALL}, "options")],
#     [Input({"type": "factor-dropdown", "index": ALL}, "value")],
#     [State({"type": "factor-dropdown", "index": ALL}, "id")]
# )
# def update_dropdown_options(selected_values, dropdown_ids):
#     """
#     Update dropdown options to hide factors that have already been selected
#     This enforces the constraint that each factor can only be used once
#     """
#     ctx = callback_context
#     if not ctx.triggered or not all([isinstance(x, (list, tuple)) for x in [selected_values, dropdown_ids]]):
#         # Initialize with all options available
#         return [[{"label": factor, "value": factor} for factor in all_factors] for _ in dropdown_ids]
    
#     # Get all selected factors (excluding empty selections)
#     selected_factors = [v for v in selected_values if v]
    
#     # Create list of dropdown options for each dropdown
#     all_options = []
#     for i, dropdown_id in enumerate(dropdown_ids):
#         current_value = selected_values[i] if i < len(selected_values) else ""
        
#         # This dropdown's options should include all factors except those selected in other dropdowns
#         available_factors = all_factors.copy()
#         for j, value in enumerate(selected_values):
#             if value and j != i:  # Skip this dropdown's own value
#                 if value in available_factors:
#                     available_factors.remove(value)
        
#         # Always include the current selection in options (to prevent it from disappearing)
#         if current_value and current_value not in available_factors:
#             available_factors.append(current_value)
#             available_factors.sort()
        
#         # Create options list for this dropdown
#         dropdown_options = [{"label": factor, "value": factor} for factor in available_factors]
#         all_options.append(dropdown_options)
    
#     return [all_options]

# Callback for prediction & visualization
@app.callback(
    [
        Output("prediction-output", "children"), 
        Output("prediction-confidence", "children"),
        Output("factor-distribution", "figure"),  # We'll reuse this for feature importance
        Output('dummy-output', 'children')
    ],
    [Input("predict-btn", "n_clicks")],
    [
        State({"type": "factor-dropdown", "index": ALL}, "value"),
        State("model-selector", "value"),
        State("subject-id-selector", "value"),
        State("new-subject-id-input", "value"),
        State("date-picker", "date"),
        State("time-input", "value"),
        State("toggle-checkbox", "value")
    ],
    prevent_initial_call=True
)
def predict_rank(n_clicks, factor_values, selected_model, subject_id, new_subject_id, date_picker, time_input, toggle_checkbox):
    """Make predictions based on user input and selected model"""
    global df
    print(f"Date: {date_picker}")
    if selected_model != "None":
        filtered_df = df[df['Model'] == selected_model] # or model is blank... from sql table
    else:
        filtered_df = df

    if not n_clicks:
        return "No prediction yet", "", {}, 'Dummy signal for second callback'
    
    # Debug info
    print(f"Prediction requested for model: {selected_model}")
    print(f"Subject ID: {subject_id} (New: {new_subject_id})")
    print(f"Factor values: {factor_values}")
    
    # Determine actual subject ID to use
    used_subject_id = new_subject_id if subject_id == "new_subject" and new_subject_id else subject_id
    
    # Get subject history if available
    subject_history = None
    if used_subject_id and used_subject_id != "new_subject":
        subject_history = get_subject_history(used_subject_id, filtered_df)
        if subject_history:
            print(f"Found history for Subject {used_subject_id}: Previous Rank = {subject_history['previous_rank']}")
    
    # Create input dataframe from factor values
    input_data = {}
    for i, col in enumerate(driver_columns):
        if i < len(factor_values):
            factor_value = factor_values[i] if factor_values[i] else ""
        else:
            factor_value = ""
        input_data[col] = [factor_value]
    
    # Add subject ID
    input_data["SubjectID"] = [used_subject_id]
    
    # If we have subject history, add previous rank
    if subject_history and subject_history['previous_rank'] is not None:
        input_data["PrevRank"] = [subject_history['previous_rank']]
    
    # Convert to DataFrame
    df_new = pd.DataFrame(input_data)
    print(f"Created input DataFrame: {df_new.shape}")
    print(f"Input data: {df_new.to_dict(orient='records')}")
    df_new.to_csv('test1.csv')
    
    # Check if we have any factors selected
    has_factors = any(v for v in factor_values if v)
    
    if not has_factors:
        return "No prediction", "Please select at least one driver factor", {}, 'Dummy signal for second callback'
    
    try:
        # Ensure we have a model to use
        model_path = available_models.get(selected_model, "")
        
        if not model_path:
            return "Model error", f"Model {selected_model} not found", {}, 'Dummy signal for second callback'
        
        # Load the model
        print(f"Loading model from {model_path}")
        model = load_model(model_path)
        
        if model is None:
            return "Model error", "Could not load model", {}, 'Dummy signal for second callback'
        
        print(f"Successfully loaded model: {type(model).__name__}")
        
        # Try different approaches to make a prediction
        approach_used = "unknown"
        predicted_rank_value = None
        X_pred = None  # Store the prediction DataFrame for feature importance
        
        # APPROACH 1: Try using feature columns file (most reliable)
        try:
            print("Attempting prediction using feature columns file")
            feature_columns_path = os.path.join(MODEL_PATH, "feature_columns.pkl")
            
            if os.path.exists(feature_columns_path):
                with open(feature_columns_path, 'rb') as f:
                    feature_columns = pickle.load(f)
                    
                print(f"Loaded {len(feature_columns)} feature columns")
                
                # Create one-hot encoding for all driver columns
                encoded_df = pd.get_dummies(df_new, columns=driver_columns, prefix=driver_columns)
                
                # Create a DataFrame with all needed columns, filled with 0's
                X_pred = pd.DataFrame(0, index=[0], columns=feature_columns)
                
                # Update with values from our encoded input data
                for col in encoded_df.columns:
                    if col in X_pred.columns:
                        X_pred[col] = encoded_df[col].values
                
                print(f"Prediction data shape: {X_pred.shape}")
                
                # Make prediction using our helper function
                predicted_rank = make_prediction_with_model(model, X_pred)
                predicted_rank_value = int(round(predicted_rank[0]))
                print(f"Predicted rank: {predicted_rank_value}")
                
                approach_used = "feature columns"
            else:
                raise ValueError("Feature columns file not found")
                
        except Exception as e1:
            print(f"Feature columns approach failed: {str(e1)}")
            
            # APPROACH 2: Try to use model.feature_names_in_ if available
            try:
                print("Trying model.feature_names_in_ approach")
                
                # Create one-hot encoding
                encoded_df = pd.get_dummies(df_new, columns=driver_columns, prefix=driver_columns)
                
                # Check if the model (or a model inside a dict) has feature_names_in_
                feature_names = None
                
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                elif isinstance(model, dict):
                    # Look through the dictionary for an object with feature_names_in_
                    for key, value in model.items():
                        if hasattr(value, 'feature_names_in_'):
                            feature_names = value.feature_names_in_
                            break
                
                if feature_names is not None:
                    print(f"Found {len(feature_names)} feature names in model")
                    
                    # Create DataFrame with zeros for all expected features
                    X_pred = pd.DataFrame(0, index=[0], columns=feature_names)
                    
                    # Update with values we have
                    for col in encoded_df.columns:
                        if col in X_pred.columns:
                            X_pred[col] = encoded_df[col].values
                            
                    # Make prediction
                    predicted_rank = make_prediction_with_model(model, X_pred)
                    X_pred.to_csv("test.csv")
                    predicted_rank_value = int(round(predicted_rank[0]))
                    print(f"Predicted rank: {predicted_rank_value}")
                    
                    approach_used = "model features"
                else:
                    raise ValueError("Model doesn't have feature_names_in_ attribute")
                    
            except Exception as e2:
                print(f"Model features approach failed: {str(e2)}")
                
                # APPROACH 3: Let's try to directly extract the model from saved files
                try:
                    print("Trying to directly load the actual model file")
                    # Check which model file we should use
                    model_filename = None
                    
                    for _, row in model_summary.iterrows():
                        if row['model_name'] == selected_model:
                            model_filename = row['filename']
                            break
                    
                    if not model_filename:
                        model_filename = f"{selected_model.lower().replace(' ', '_')}.pkl"
                    
                    # Try to load the model directly
                    direct_model_path = os.path.join(MODEL_PATH, model_filename)
                    print(f"Looking for model at: {direct_model_path}")
                    
                    if os.path.exists(direct_model_path):
                        with open(direct_model_path, 'rb') as f:
                            direct_model = pickle.load(f)
                        
                        print(f"Loaded direct model of type: {type(direct_model).__name__}")
                        
                        # Create input features using simple encoding
                        X_pred = pd.get_dummies(df_new, columns=driver_columns, prefix=driver_columns)
                        
                        # Try to predict
                        predicted_rank = make_prediction_with_model(direct_model, X_pred)
                        predicted_rank_value = int(round(predicted_rank[0]))
                        print(f"Predicted rank: {predicted_rank_value}")
                        
                        # Update our model reference for feature importance
                        model = direct_model
                        approach_used = "direct model"
                    else:
                        raise ValueError(f"Direct model file not found: {direct_model_path}")
                        
                except Exception as e3:
                    print(f"Direct model approach failed: {str(e3)}")
                    
                    # APPROACH 4: Last resort - manually extract and use raw predictions if possible
                    try:
                        print("Attempting to use raw predictions from model")
                        
                        # Create a basic input representation for feature importance
                        X_pred = pd.get_dummies(df_new, columns=driver_columns, prefix=driver_columns)
                        
                        # If model is a dictionary with raw prediction values
                        if isinstance(model, dict) and 'raw_predictions' in model:
                            raw_preds = model['raw_predictions']
                            if isinstance(raw_preds, np.ndarray):
                                predicted_rank = np.mean(raw_preds)
                                predicted_rank_value = int(round(predicted_rank))
                                print(f"Used raw predictions, got rank: {predicted_rank_value}")
                                approach_used = "raw predictions"
                            else:
                                raise ValueError("Raw predictions not in expected format")
                        
                        # If the model dictionary has precomputed values
                        elif isinstance(model, dict) and 'ranks' in model:
                            # Just use the first rank from the precomputed values
                            ranks = model['ranks']
                            if isinstance(ranks, (list, np.ndarray)) and len(ranks) > 0:
                                predicted_rank_value = int(ranks[0])
                                print(f"Used precomputed rank: {predicted_rank_value}")
                                approach_used = "precomputed values"
                            else:
                                raise ValueError("Precomputed ranks not available")
                        else:
                            raise ValueError("No raw predictions or precomputed values found")
                        
                    except Exception as e4:
                        print(f"All approaches failed:\n1: {str(e1)}\n2: {str(e2)}\n3: {str(e3)}\n4: {str(e4)}")
                        return "Error", "All prediction approaches failed", {}, 'Dummy signal for second callback'
        
        # If we don't have a prediction by now, return an error
        if predicted_rank_value is None:
            return "Error", "Could not generate a prediction", {}, 'Dummy signal for second callback'
            
        # Get confidence/info text
        model_row = model_summary[model_summary['model_name'] == selected_model]
        if not model_row.empty:
            info = model_row.iloc[0]
            confidence_text = (f"Expected accuracy: ±{info['mae']:.1f} ranks | "
                              f"Within 10 ranks: {info['accuracy_10']*100:.1f}% "
                              f"(using {approach_used})")
        else:
            # Count factors used for general info
            factors_used = sum(1 for v in factor_values if v)
            confidence_text = f"Prediction based on {factors_used} factors (using {approach_used})"
        
        # Create feature importance plot instead of factor distribution
        selected_factors = [v for v in factor_values if v]
        
        if X_pred is not None:
            # Create importance chart
            importance_fig = create_importance_chart(model, X_pred, selected_factors, approach_used)

            if toggle_checkbox:
                conn = sqlite3.connect(DATA_PATH)
                # Query specific columns for the first row and convert to a list
                values_list = df.loc[0, [f"Driver{i}" for i in range(1, 18)]].tolist()
                insert_into_rankings(conn, used_subject_id, predicted_rank_value, values_list, f"{date_picker} {time_input}:00", False, selected_model)
                conn.close()
                df = load_sqlite_to_dataframe(DATA_PATH)
                df['UpdateDT'] = pd.to_datetime(df['UpdateDT'])
                conn.close()
            
            return f"Rank: {predicted_rank_value}", confidence_text, importance_fig, 'Dummy signal for second callback'
        else:
            # Fallback if we don't have prediction data
            fig = px.bar(
                x=selected_factors,
                y=[1] * len(selected_factors),
                labels={"x": "Selected Factors", "y": "Count"},
                title="Feature importance not available"
            )


            X_pred.to_csv("test.csv")

            if toggle_checkbox:
                conn = sqlite3.connect(DATA_PATH)
                # Query specific columns for the first row and convert to a list
                values_list = df.loc[0, [f"Driver{i}" for i in range(1, 18)]].tolist()
                insert_into_rankings(conn, used_subject_id, predicted_rank_value, values_list, f"{date_picker} {time_input}:00", False, selected_model)
                conn.close()
                df = load_sqlite_to_dataframe(DATA_PATH)
                df['UpdateDT'] = pd.to_datetime(df['UpdateDT'])
                conn.close()

            return f"Rank: {predicted_rank_value}", confidence_text, fig, 'Dummy signal for second callback'

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error", "Could not generate prediction. Check logs for details.", {}, 'Dummy signal for second callback'

# Expose the server for Gunicorn to run
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
    # try:
    #     app.run_server(host='0.0.0.0', port=8050, debug=False)
    # except Exception as e:
    #     print(f"Error starting server: {str(e)}")