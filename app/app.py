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

# =============================================================================
# Configuration and Initialization
# =============================================================================

# Define paths according to repository structure
MODEL_PATH = "model/saved_models_features"
MODEL_SUMMARY_PATH = "model/model_summary_enhanced.csv"
DATA_PATH = "data/rankings.db"

# Load model summary for model selection
model_summary = pd.read_csv(MODEL_SUMMARY_PATH)
print(f"Loaded model summary with {len(model_summary)} models")

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

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

# Debug information for data loading
print(f"DATA_PATH: {DATA_PATH}")
print(f"CSV exists: {os.path.exists(DATA_PATH)}")
if not df.empty:
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"First few subject IDs: {df['SubjectID'].head().tolist() if 'SubjectID' in df.columns else 'No SubjectID column'}")

# =============================================================================
# Process Reference Data
# =============================================================================

all_factors = []
if not df.empty:
    # Convert date column if exists
    if 'UpdateDT' in df.columns:
        try:
            df['UpdateDT'] = pd.to_datetime(df['UpdateDT'])
        except Exception as e:
            print(f"Error converting UpdateDT to datetime: {str(e)}")
    
    # Identify driver columns (columns that contain factor information)
    driver_columns = [col for col in df.columns if col.startswith('Driver') and col.replace('Driver', '').isdigit()]
    if not driver_columns and len(df.columns) > 2:
        # Fallback: assume columns 2-19 are driver columns if naming pattern not found
        driver_columns = df.columns[2:19].tolist()
    
    # Extract all unique factors from reference data
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

# =============================================================================
# Helper Functions
# =============================================================================

def get_subject_history(subject_id, reference_df):
    """
    Get previous rank information for a specific subject ID
    
    Args:
        subject_id: The subject ID to look up
        reference_df: The reference dataset
        
    Returns:
        Dict containing previous rank and other historical data
    """    
    # Filter for this subject
    subject_data = reference_df[reference_df['SubjectID'] == subject_id]
    print(subject_data)
    
    # Sort by date to get most recent entry
    if 'UpdateDT' in subject_data.columns:
        try:
            subject_data = subject_data.sort_values('UpdateDT', ascending=False)
        except:
            pass  # If sorting fails, continue without sorting
    
    # Get most recent record
    try:
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
    except:
        # Fallback history data when no records are found
        history = {
            'previous_rank': None,
            'last_update': None,
            'num_records': 0,
            'avg_rank': None,
            'min_rank': None,
            'max_rank': None,
        }

    return history

def engineer_features(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform feature engineering for prediction
    
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
    
    # Add timestamp
    processed_df['UpdateDT'] = datetime.now()
    processed_df['UpdateDT'] = pd.to_datetime(processed_df['UpdateDT'])
    
    # 1. Add time-based features
    processed_df['Year'] = processed_df['UpdateDT'].dt.year
    processed_df['Month'] = processed_df['UpdateDT'].dt.month
    processed_df['DayOfMonth'] = processed_df['UpdateDT'].dt.day
    processed_df['DayOfWeek'] = processed_df['UpdateDT'].dt.dayofweek
    
    # Calculate days since first observation
    min_date = reference_df['UpdateDT'].min() if 'UpdateDT' in reference_df.columns else processed_df['UpdateDT'].min()
    processed_df['DaysSinceFirst'] = (processed_df['UpdateDT'] - pd.to_datetime(min_date)).dt.days
    
    # Set days since previous observation to 0 for new input
    processed_df['DaysSincePrev'] = 0
    
    # 2. Add previous rank features
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
    
    if numeric_non_onehot and 'scaler' in globals():
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

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """
    Plot feature importance for a trained model.

    Args:
        model: Trained model (supports tree-based models).
        feature_names: List of feature names.
        model_name: Name of the model.
        top_n: Number of top features to display.
    """
    print(f"\nPlotting feature importance for {model_name}...")

    importances = {}

    # Retrieve feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print(f"Feature importance not available for {model_name}. Skipping.")
        return
    
    # Store importance for later comparison
    importances[model_name] = importance
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n)

# Helper function to extract feature importances from different model types
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


# Function to create a feature importance visualization chart
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
    factor_importance_df = importance_df[importance_df['Feature'].str.contains('Factor')]
    
    # Get the top 10 most important features
    top_features = factor_importance_df.head(10).copy()
    
    # Create importance plot
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
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


# Define the model selector dropdown UI component
model_selector = html.Div([
    html.Label("Select Prediction Model", className="fw-bold mb-2"),
    dcc.Dropdown(
        id="model-selector",
        options=[{"label": name, "value": name} for name in model_summary["model_name"]],
        value="Random Forest",
        clearable=False,
        className="mb-3"
    ),
    html.Div(id="selected-model-info", className="small text-muted")
])


# Create datetime selector UI component
date_time_ui = html.Div([
    dbc.Row([
        html.Label("Enter Date", style={'width': '70%', 'text-align': 'center'}, className="fw-bold mb-2"),
        html.Label("Enter HH:MM", style={'width': '30%', 'text-align': 'center'}, className="fw-bold mb-2"),
    ]),
    dbc.Row([
        dcc.DatePickerSingle(
            id="date-picker",
            date=datetime.datetime.now().date(),
            display_format="YYYY-MM-DD",
            className="dash-bootstrap",
            style={'width': '70%'}
        ),
        dbc.Input(
            id="time-input",
            type="text",
            value=datetime.datetime.now().strftime("%H:%M"),
            placeholder="HH:MM",
            style={'width': '30%', 'text-align': 'center'}
        )
    ]),
], style={"display": "none"}, id="date-time-container")


# Create subject ID selector UI component
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
    # Check if reference data exists
    if reference_df.empty or 'SubjectID' not in reference_df.columns:
        return go.Figure(data=[go.Scatter(x=[], y=[], mode='lines', name="No history data available")])
    
    # Filter for this subject
    subject_data = reference_df[reference_df['SubjectID'] == subject_id]
    
    # Check if subject has rank data
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
    Returns:
        HTML Div containing rows of factor selector dropdowns
    """
    selectors = []
    
    # Create a dropdown for each driver
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
    
    # Group selectors into rows of 6 columns each for better UI layout
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
                
                # Subject selector with date/time options
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
    
    # Subject history section with tabs for rank history and subject details
    html.Div(
        id="subject-history-container",
        style={"display": "block"},  # Initially visible
        children=[
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="bi bi-clock-history me-2"),
                        "Subject History"
                    ], className="mb-0"),
                ]),
                dbc.CardBody([
                    dbc.Tabs(
                        [
                            # Tab for rank history
                            dbc.Tab(
                                label="Rank History",
                                tab_id="rank-info",
                                children=html.Div(
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
                                    className="mt-4",  # Spacing between tabs and content
                                ),
                            ),
                            # Tab for subject details
                            dbc.Tab(
                                label="Subject Details",
                                tab_id="subject-info",
                                children=html.Div(dcc.Graph(id="subject-chart"), className="mt-4"),
                            ),
                        ],
                        id="subject-history-tabs",
                        active_tab="rank-info",
                    ),
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
            # Factor selection dropdowns
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
    
    # Results section (two-column layout)
    dbc.Row([
        # Left column: Prediction result
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
                        html.Div(
                            children=dbc.Alert("Prediction Results will be shown here once factors are selected.", color="primary"), 
                            id="prediction-output"
                        ),
                        html.Div(id="prediction-confidence", className="text-muted text-center"),
                    ], className="py-4 text-center"),
                ], style={"min-height": "300px"}),
            ], className="shadow-sm h-100"),
        ], md=4),
        
        # Right column: Influential factors
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
                    html.Div(
                        id="factor-distribution", 
                        children=dbc.Alert("Influential Factors will be shown once factors are selected.", color="primary")
                    )
                ]),
            ], className="shadow-sm h-100"),
        ], md=8),
    ]),
], className="py-3")


def get_model_description() -> str:
    """
    Get description of model based on its name.
    
    Returns:
        str: A description of the model's feature engineering approach
    """
    return "This model uses a custom feature engineering approach based on the input factors."


def create_model_info_tab(model_name: str) -> html.Div:
    """
    Create model info tab content based on selected model.
    
    Args:
        model_name (str): The name of the model to display information for
        
    Returns:
        html.Div: A Dash component containing the model information layout
    """
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
                    # Left column: Error and correlation metrics
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
                    
                    # Right column: Accuracy metrics with chart
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
                
                # Model description section
                html.H5("Model Description", className="border-bottom pb-2 mt-4"),
                html.P([
                    f"This {model_name} model was trained on historical ranking data with "
                    f"multiple driver factors. It is designed to predict the rank based on "
                    f"the selected decision factors.",
                    html.Br(),
                    "The model was evaluated using a time-based split of the data, with the most "
                    "recent data used for testing.",
                ]),
                
                # Feature type section
                html.Div([
                    html.H5("Feature Type", className="border-bottom pb-2"),
                    html.P(get_model_description()),
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
            # Application description
            html.P("RankLab is an interactive web application designed for Multi-Criteria Decision Analysis Rank Prediction."),
            html.P("This app allows users to enter decision factors and predict a rank based on historical data."),
            html.P("Developed using Python (Dash, Plotly, Pandas) with machine learning models for predictions."),
            
            html.Hr(),
            
            # Contributors section with profile cards
            html.H5("Contributors", className="mb-3"),
            dbc.Row([
                # Profile card for Mark
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Mark Ralston Daniel", className="card-title text-center"),
                            html.Img(src="/assets/mark.png", height="200px", className="d-block mx-auto")
                        ]),
                    ], className="shadow-sm h-100"),
                ], md=3, className="mb-3"),
                
                # Profile card for Amit
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Amit Peled", className="card-title text-center"),
                            html.Img(src="/assets/amit.png", height="200px", className="d-block mx-auto")
                        ]),
                    ], className="shadow-sm h-100"),
                ], md=3, className="mb-3"),
                
                # Profile card for Jake
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Jake Flynn", className="card-title text-center"),
                            html.Img(src="/assets/jake.png", height="200px", className="d-block mx-auto")
                        ]),
                    ], className="shadow-sm h-100"),
                ], md=3, className="mb-3"),
                
                # Profile card for Dr. Song
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Dr. Song", className="card-title text-center"),
                            html.Img(src="/assets/song.png", height="200px", className="d-block mx-auto")
                        ]),
                    ], className="shadow-sm h-100"),
                ], md=3, className="mb-3"),
            ]),
            
            html.Hr(),
            
            # Technical details section
            html.H5("Technical Details", className="mb-3"),
            html.P([
                html.B("Frontend:"), " Dash, Dash Bootstrap Components, Plotly",
                html.Br(),
                html.B("Backend:"), " Python, scikit-learn",
                html.Br(),
                html.B("Data Processing:"), " Pandas, NumPy",
                html.Br(),
                html.B("Visualization:"), " Plotly",
                html.Br(),
                html.B("Deployment:"), " Docker, Gunicorn"
            ])
        ]),
    ], className="shadow-sm"),
], className="py-4")


# External stylesheet URL for Dash Bootstrap Components
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

# Initialize Dash application with required stylesheets
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        dbc_css,
        "/assets/dbc.css"
    ],
    assets_folder='../assets',
    suppress_callback_exceptions=True
)
app.title = "RankLab: Rank Prediction Tool"

# Main application layout structure
app.layout = dbc.Container([
    # Hidden div for callback chaining
    html.Div(id='dummy-output', style={'display': 'none'}),

    # Header section with logo and application title
    dbc.Row([
        dbc.Col(html.I(className="bi bi-bar-chart-fill text-primary", 
                    style={"fontSize": "3rem"}), width="auto", className="pr-1"),
        dbc.Col([
            html.Div([
                html.Span("RankLab", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
            ]),
            html.P("Multi-Criteria Decision Analysis Rank Prediction", className="text-muted"),
        ], width="auto"),
    ], className="border-bottom pb-3 d-flex align-items-center"),
    
    # Tab navigation container
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
    
    # Footer section
    dbc.Row([
        dbc.Col([
            html.P("© 2025 RankLab - All Rights Reserved", className="text-center text-muted small mt-4"),
        ]),
    ], className="border-top pt-3"),
], fluid=True, className="px-4 py-3 bg-light min-vh-100")

# Callback: Updates heatmap visualization based on selected subject
@app.callback(
    Output("subject-chart", "figure"),
    [Input("subject-id-selector", "value"),
     Input('dummy-output', 'children')
    ]
)
def update_heatmap(selected_subject, dummy_output):
    # Return empty chart if no subject is selected
    if selected_subject is None:
        return px.imshow([[0]], labels={"x": "Driver", "y": "Factor"}, title="No Subject Selected")

    # Filter data for the selected subject
    filtered_df = df[df["SubjectID"] == selected_subject]

    # Get only the Driver columns for analysis
    driver_columns = [col for col in df.columns if col.startswith("Driver")]

    # Reshape data to long format: each row has (Driver, Factor)
    melted_df = filtered_df.melt(var_name="Driver", value_name="Factor", value_vars=driver_columns)

    # Count occurrences of each Factor per Driver
    factor_counts = melted_df.groupby(["Driver", "Factor"]).size().reset_index(name="Count")

    # Pivot data for heatmap visualization
    heatmap_pivot = factor_counts.pivot(index="Factor", columns="Driver", values="Count").fillna(0)

    # Create and configure heatmap visualization
    fig = px.imshow(
        heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        color_continuous_scale="Blues",
        labels={"x": "Driver", "y": "Factor", "color": "Occurrences"},
        title=f"Factor Importance Heatmap for Subject {selected_subject}"
    )

    fig.update_layout(
        xaxis_title="Driver Position",
        yaxis_title="Factors",
        template="plotly_white",
        height=500
    )
    
    return fig

# Callback: Randomizes factor selections for all dropdowns
@app.callback(
    [Output({"type": "factor-dropdown", "index": idx}, "value") for idx in range(len(driver_columns))],
    Input("randomize-btn", "n_clicks"),
    prevent_initial_call=True
)
def randomize_factors(n_clicks):
    # Create list of all possible factors
    factor_options = [f"Factor{i}" for i in range(1, 18)]  # Factor1 to Factor17
    
    # Shuffle factors randomly to ensure unique values across dropdowns
    shuffled_factors = random.sample(factor_options, len(factor_options))
    
    return shuffled_factors

# Callback: Controls visibility of datetime input container
@app.callback(
    Output("date-time-container", "style"),
    [Input("toggle-checkbox", "value")]
)
def toggle_new_date_time_input(selected: bool) -> Dict:
    """Show/hide date time input field based on checkbox selection"""
    if selected:
        return {"display": "block"}
    return {"display": "none"}

# Callback: Updates model information tab content
@app.callback(
    Output("model-info-tab", "children"),
    [Input("model-selector", "value")]
)
def update_model_info_tab(selected_model: str) -> html.Div:
    """Display detailed information for the selected model"""
    if not selected_model:
        return html.Div("No model selected")
    return create_model_info_tab(selected_model)

# Callback: Updates selected model summary information
@app.callback(
    Output("selected-model-info", "children"),
    [Input("model-selector", "value")]
)
def update_selected_model_info(selected_model: str) -> Union[str, List]:
    """Display brief model performance metrics"""
    if not selected_model:
        return "No model selected"
        
    model_row = model_summary[model_summary['model_name'] == selected_model]
    if model_row.empty:
        return f"Using {selected_model}"
    
    # Display key performance metrics for the selected model
    info = model_row.iloc[0]
    return [
        f"MAE: {info['mae']:.2f} | R²: {info['r2']:.3f}", html.Br(),
        f"Accuracy within 10 ranks: {info['accuracy_10']*100:.1f}%"
    ]

# Callback: Updates subject history section
@app.callback(
    [Output("subject-history-container", "style"),
     Output("subject-history-content", "children"),
     Output("subject-history-chart", "figure")],
    [Input("subject-id-selector", "value"),
     State("new-subject-id-input", "value"),
     Input("model-selector", "value"),
     Input('dummy-output', 'children')]
)
def update_subject_history(selected_subject, new_subject, selected_model, dummy_output):
    """Display subject history data and visualization"""
    filtered_df = df

    # Return empty state if no subject is selected
    if not selected_subject:
        empty_fig = px.line(title="No history available")
        empty_fig.update_layout(height=200)
        return {"display": "none"}, "", empty_fig

    # Get history for either existing or new subject
    if selected_subject == "new_subject":
        history = get_subject_history(new_subject, filtered_df)
    else:
        history = get_subject_history(selected_subject, filtered_df)
    
    # Create formatted history display
    content = [
        html.Div([
            # Previous rank information
            html.P([
                html.Strong("Previous Rank: "), 
                html.Span(
                    f"{history['previous_rank']}" if history['previous_rank'] is not None else "Not available",
                    className="fs-4 text-primary"
                )
            ], className="mb-2"),
            
            # Last update date information
            html.P([
                html.Strong("Last Update: "), 
                f"{history['last_update'].strftime('%Y-%m-%d')}" if history['last_update'] is not None else "Not available"
            ], className="mb-2"),
            
            # Number of historical records
            html.P([
                html.Strong("History: "), 
                f"{history['num_records']} records"
            ], className="mb-2"),
            
            # Statistical summary row (average, min, max)
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
            
            # Additional information text
            html.P(
                "This information will be used in the prediction.",
                className="text-muted small mt-2 fst-italic"
            ),
        ], className="h-100 d-flex flex-column justify-content-center")
    ]
    
    # Create history visualization chart
    if selected_subject == "new_subject":
        history_chart = create_subject_history_chart(new_subject, filtered_df)
    else:
        history_chart = create_subject_history_chart(selected_subject, filtered_df)
    
    return {"display": "block"}, content, history_chart

# Callback: Controls visibility of new subject input field
@app.callback(
    Output("new-subject-container", "style"),
    [Input("subject-id-selector", "value")]
)
def toggle_new_subject_input(selected_subject):
    """Show new subject input field only when 'New Subject' is selected"""
    if selected_subject == "new_subject":
        return {"display": "block"}
    return {"display": "none"}

# Callback: Updates factor dropdown options to prevent duplicate selections
@app.callback(
    [Output({"type": "factor-dropdown", "index": ALL}, "options")],
    [Input({"type": "factor-dropdown", "index": ALL}, "value")],
    [State({"type": "factor-dropdown", "index": ALL}, "id")]
)
def update_dropdown_options(selected_values, dropdown_ids):
    """
    Dynamically updates dropdown options to prevent selecting the same factor multiple times.
    Each factor can only be selected in one dropdown at a time.
    """
    # List of all possible factors
    all_factors = [f"Factor{i}" for i in range(1, 18)]
    
    # Get the set of selected values, ignoring None values
    selected_values_set = set(filter(None, selected_values))
    
    # Generate updated options for each dropdown
    updated_options = []
    for dropdown_id, selected_value in zip(dropdown_ids, selected_values):
        # Available options exclude the selected values from other dropdowns
        # Keep the currently selected value in the options list
        available_options = [factor for factor in all_factors 
                           if factor not in selected_values_set or factor == selected_value]
        updated_options.append([{"label": factor, "value": factor} for factor in available_options])
    
    return [updated_options]

def engineer_enhanced_positional_features(df):
    """
    Generates enhanced features for rank prediction based on subject history and factor positions.
    
    Args:
        df: DataFrame containing subject ranking data
    
    Returns:
        DataFrame with engineered features ready for model input
    """
    # Convert date column to datetime format
    df['UpdateDT'] = pd.to_datetime(df['UpdateDT'])
    
    # Sort and deduplicate data to ensure proper time ordering
    df = df.sort_values(['SubjectID', 'UpdateDT'])
    df = df.drop_duplicates()
    
    # Create a copy for processing
    processed_df = df.sort_values(['SubjectID', 'UpdateDT']).copy()
    
    # Add calendar-based features
    processed_df['Year'] = processed_df['UpdateDT'].dt.year
    processed_df['Month'] = processed_df['UpdateDT'].dt.month
    processed_df['DayOfMonth'] = processed_df['UpdateDT'].dt.day
    processed_df['DayOfWeek'] = processed_df['UpdateDT'].dt.dayofweek
    
    # Add time-based features
    min_date = processed_df['UpdateDT'].min()
    processed_df['DaysSinceFirst'] = (processed_df['UpdateDT'] - min_date).dt.days
    
    # Calculate days since previous observation for each subject
    processed_df['DaysSincePrev'] = processed_df.groupby('SubjectID')['UpdateDT'].diff().dt.days
    processed_df['DaysSincePrev'] = processed_df['DaysSincePrev'].fillna(0)
    
    # Add historical ranking features
    processed_df['PrevRank'] = processed_df.groupby('SubjectID')['Rank'].shift(1)
    processed_df['RankChange'] = processed_df['Rank'] - processed_df['PrevRank']
    
    # Set first rank change as NaN so it's ignored in the expanding mean
    processed_df.loc[processed_df.groupby('SubjectID').cumcount() == 0, 'RankChange'] = np.nan
    
    # Calculate average of all previous ranks (excluding current rank)
    processed_df['AvgPreviousRanks'] = processed_df.groupby('SubjectID')['Rank'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # Fill missing values in AvgPreviousRanks with first available rank
    processed_df['AvgPreviousRanks'] = processed_df.groupby('SubjectID')['AvgPreviousRanks'].transform(
        lambda x: x.fillna(method='bfill').fillna(method='ffill')
    )
    
    # Calculate and fill average rank changes
    processed_df['AvgPreviousRankChanges'] = processed_df.groupby('SubjectID')['RankChange'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    processed_df['AvgPreviousRankChanges'] = processed_df['AvgPreviousRankChanges'].fillna(0)
    
    # Transform driver-factor relationships into positional features
    driver_cols = [f'Driver{i}' for i in range(1, 18)]
    
    # Melt DataFrame to long format for factor position analysis
    melted_df = processed_df.melt(id_vars=['SubjectID', 'UpdateDT'], value_vars=driver_cols, 
                                 var_name='DriverCol', value_name='Factor')
    
    # Extract numerical driver position from column name
    melted_df['Driver_Number'] = melted_df['DriverCol'].str.extract('(\d+)').astype(int)
    
    # Pivot back to wide format with factor values as driver numbers
    position_features = melted_df.pivot(index=['SubjectID', 'UpdateDT'], 
                                       columns='Factor', values='Driver_Number').fillna(0)
    
    # Merge positional features back to the main DataFrame
    processed_df = processed_df.merge(position_features, on=['SubjectID', 'UpdateDT'], how='left')
    
    # Remove original driver columns as they're replaced by factor positions
    processed_df.drop(columns=driver_cols, inplace=True)
    
    # Fill any remaining missing values to ensure model compatibility
    processed_df.fillna(0, inplace=True)

    # Define feature columns by excluding non-feature columns
    exclude_cols = ['SubjectID', 'Rank', 'UpdateDT', 'RankChange', 'Year', 'Month', 
                   'DayOfMonth', 'DayOfWeek', 'OriginalDataFlag', 'Model']
    feature_cols = [col for col in processed_df.columns if col not in exclude_cols]

    # Return only the feature columns
    return processed_df[feature_cols]

# Callback for prediction & visualization
@app.callback(
    [
        Output("prediction-output", "children"), 
        Output("prediction-confidence", "children"),
        Output("factor-distribution", "children"),  # Reused for feature importance
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
    """
    Make predictions based on user input and selected model.
    
    Parameters:
    -----------
    n_clicks : int
        Number of times the prediction button has been clicked
    factor_values : list
        Values of the factor dropdowns
    selected_model : str
        Name of the selected model for prediction
    subject_id : str
        Selected subject ID
    new_subject_id : str
        New subject ID if user chooses to create one
    date_picker : str
        Selected date
    time_input : str
        Selected time
    toggle_checkbox : bool
        Whether to save the prediction to the database
        
    Returns:
    --------
    tuple
        (prediction output, confidence metrics, feature importance graph, dummy output)
    """
    global df
    
    # Return early if button not clicked
    if not n_clicks:
        return "No prediction yet", "", {}, 'Dummy signal for second callback'
    
    # Filter dataframe for the selected model
    filtered_df = df[(df['Model'] == selected_model) | (df['Model'].isna())]
    
    # Debug information
    print(f"Prediction requested for model: {selected_model}")
    print(f"Subject ID: {subject_id} (New: {new_subject_id})")
    print(f"Factor values: {factor_values}")
    
    # Determine which subject ID to use
    used_subject_id = new_subject_id if subject_id == "new_subject" and new_subject_id else subject_id
    print(used_subject_id)
    
    # Create input dataframe from factor values
    input_data = {}
    for i, col in enumerate(driver_columns):
        if i < len(factor_values):
            factor_value = factor_values[i] if factor_values[i] else ""
        else:
            factor_value = ""
        input_data[col] = [factor_value]
    
    # Add subject ID and placeholder rank
    input_data["SubjectID"] = [used_subject_id]
    input_data['Rank'] = np.nan
    
    # Convert to DataFrame
    df_new = pd.DataFrame(input_data)
    print(f"Created input DataFrame: {df_new.shape}")
    print(f"Input data: {df_new.to_dict(orient='records')}")
    
    # Check if any factors were selected
    has_factors = any(v for v in factor_values if v)
    
    if not has_factors:
        return "No prediction", "Please select at least one driver factor", {}, 'Dummy signal for second callback'

    # Add timestamp and engineer features
    df_new['UpdateDT'] = pd.to_datetime(f"{date_picker} {time_input}:00")
    new_df = engineer_enhanced_positional_features(df_new)
    new_df = new_df.tail(1)

    # Load the appropriate model
    if selected_model == "XGBoost":
        with open('model/xgboost_enhanced.pkl', 'rb') as file:
            model = pickle.load(file)["model"]
    elif selected_model == "Random Forest":
        with open('model/random_forest_enhanced.pkl', 'rb') as file:
            model = pickle.load(file)["model"]
    elif selected_model == "LightGBM":
        with open('model/lightgbm_enhanced.pkl', 'rb') as file:
            model = pickle.load(file)["model"]
    elif selected_model == "Gradient Boosting":
        with open('model/gradient_boosting_enhanced.pkl', 'rb') as file:
            model = pickle.load(file)["model"]

    # Make prediction
    predicted_rank = model.predict(new_df)
    predicted_rank_value = int(round(predicted_rank[0]))

    # Save to database if checkbox is toggled
    if toggle_checkbox:
        conn = sqlite3.connect(DATA_PATH)
        # Get driver values and insert into database
        values_list = df_new.loc[0, [f"Driver{i}" for i in range(1, 18)]].tolist()
        insert_into_rankings(conn, used_subject_id, predicted_rank_value, values_list, 
                             f"{date_picker} {time_input}:00", False, selected_model)
        conn.close()
        # Reload data from SQLite
        df = load_sqlite_to_dataframe(DATA_PATH)
        df['UpdateDT'] = pd.to_datetime(df['UpdateDT'])

    # Return error if prediction failed
    if predicted_rank_value is None:
        return "Error", "Could not generate a prediction", {}, 'Dummy signal for second callback'
            
    # Generate confidence text based on model metrics
    model_row = model_summary[model_summary['model_name'] == selected_model]
    if not model_row.empty:
        info = model_row.iloc[0]
        accuracy = html.Span([html.B("Expected accuracy:"), f" ±{info['mae']:.1f} ranks"])
        within_ten = html.Span([html.B("Within 10 ranks:"), f" {info['accuracy_10']*100:.1f}%"])

        # Count factors used for general info
        factors_used = sum(1 for v in factor_values if v)
        prediction_based = html.Span([html.B("Prediction:"), f" Based on {factors_used} factors"])

        confidence_text = html.Div([
            accuracy, html.Br(), html.Br(),
            within_ten, html.Br(), html.Br(),
            prediction_based
        ])
    
    # Create feature importance visualization
    selected_factors = [v for v in factor_values if v]
    importance_fig = create_importance_chart(model, new_df, selected_factors, selected_model)

    # Return prediction, confidence metrics, and visualization
    return (
        html.Div(children=f"Rank: {predicted_rank_value}", className="display-4 text-center mb-3"),
        html.Div(children=confidence_text, className="text-muted text-center"),
        dcc.Graph(style={"height": "300px"}, figure=importance_fig),
        f'Dummy signal for second callback {n_clicks}'
    )


# Expose the server for Gunicorn to run
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
