import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import dash_ag_grid as dag
import pandas as pd
import subprocess
import json
import plotly.express as px

# Load dataset
df = pd.read_csv('../data/Rank.csv')
driver_columns = df.columns[2:19].tolist()

# Generate column definitions for AG Grid
column_defs = [
    {"headerName": col, "field": col, "editable": True, "cellEditor": "agSelectCellEditor", 
     "cellEditorParams": {"values": df[col].unique().tolist()}}
    for col in driver_columns
]

# Initial blank row for user input
initial_data = [{col: "" for col in driver_columns}]

# Define the "Rank Prediction" tab content
rank_prediction_tab = dbc.Container([
    html.P("Edit the table below to enter driver factors for prediction."),

    dag.AgGrid(
        id="input-grid",
        columnDefs=column_defs,
        rowData=initial_data,
        defaultColDef={"editable": True, "sortable": True, "filter": True, "resizable": True},
        style={"height": "400px", "width": "100%"},
        columnSize="sizeToFit",
    ),

    dbc.Button("Predict Rank", id="predict-btn", color="primary", className="mt-3"),
    html.H3(id="prediction-output", className="mt-3"),

    # Visualization of factor distribution
    dcc.Graph(id="factor-distribution", style={"marginTop": "20px"}),
], className="mt-4")

# Define the "About Me" tab content
about_me_tab = dbc.Container([
    html.H2("About Me"),
    html.P("This project was created by Mark Rarlston Daniel, Amit Peled, and Jake Flynn."),
    html.P("RankLab is an interactive web application designed for Multi-Criteria Decision Analysis Rank Prediction."),
    html.P("This app allows users to enter decision factors and predict a rank based on historical data."),
    html.P("Developed using Python (Dash, Plotly, Pandas) and R for predictive modeling."),
], className="mt-4")

# App Layout with Title & Tabs
application = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
application.title = "RankLab: Rank Prediction Tool"

application.layout = dbc.Container([
    html.H1("RankLab: Rank Prediction Tool", className="text-center mt-4"),  # Title above tabs
    dbc.Tabs([
        dbc.Tab(rank_prediction_tab, label="Rank Prediction"),
        dbc.Tab(about_me_tab, label="About Me"),
    ])
])

# Callback for prediction & visualization
@application.callback(
    [Output("prediction-output", "children"), Output("factor-distribution", "figure")],
    Input("predict-btn", "n_clicks"),
    State("input-grid", "rowData"),
)
def predict_rank(n_clicks, row_data):
    if not n_clicks:
        return "Edit table and click Predict.", {}

    # Convert input to DataFrame
    df_new = pd.DataFrame(row_data)

    # Save input as CSV for R script
    df_new.to_csv("new_input.csv", index=False)

    # Run R prediction script
    result = subprocess.run(["Rscript", "predict_rank.R"], capture_output=True, text=True)

    try:
        prediction = json.loads(result.stdout.strip())
        predicted_rank = prediction["predicted_rank"]

        # Prepare data for factor distribution plot
        selected_factors = df_new.iloc[0].dropna().tolist()
        factor_counts = df[driver_columns].melt().query("value in @selected_factors")["value"].value_counts().reset_index()
        factor_counts.columns = ["Factor", "Count"]

        # Create a distribution bar chart
        fig = px.bar(
            factor_counts,
            x="Factor",
            y="Count",
            labels={"x": "Factor", "y": "Count"},
            title="Distribution of Selected Factors in Dataset",
            text="Count",
        )
        fig.update_traces(textposition="outside")

        return f"Predicted Rank: {predicted_rank}", fig

    except:
        return "Prediction error. Please check input.", {}

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8080)
