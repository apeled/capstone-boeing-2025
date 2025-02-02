import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import subprocess
import json

# Load dataset
df = pd.read_csv("../data/Rank.csv")
driver_columns = df.columns[2:19].tolist()

# Dash App Setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "RankLab: An Interactive Web Application for Multi-Criteria Decision Analysis Rank Prediction"

app.layout = dbc.Container([
    html.H1("RankLab: Rank Prediction Tool"),
    html.P("Select Driver Factors to Predict Rank"),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id=f"driver_{i}",
                options=[{"label": factor, "value": factor} for factor in df[col].unique()],
                placeholder=col,
            ) for i, col in enumerate(driver_columns)
        ], width=4)
    ]),
    
    dbc.Button("Predict Rank", id="predict-btn", color="primary", className="mt-3"),
    html.H3(id="prediction-output", className="mt-3"),
])

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(f"driver_{i}", "value") for i in range(len(driver_columns))]
)
def predict_rank(n_clicks, *selected_factors):
    if not n_clicks:
        return "Select factors and click Predict."
    
    new_data = {driver_columns[i]: [selected_factors[i] if selected_factors[i] else ""] for i in range(len(driver_columns))}
    df_new = pd.DataFrame(new_data)
    df_new.to_csv("new_input.csv", index=False)
    
    result = subprocess.run(["Rscript", "-e", "source('../models/rank_model.R'); print(decision_rank_predict(read.csv('new_input.csv')))"], capture_output=True, text=True)
    
    try:
        prediction = json.loads(result.stdout.strip())
        print(prediction)
        return f"Predicted Rank: {prediction['predicted_rank']}"
    except:
        return "Prediction error. Please check input."

if __name__ == "__main__":
    app.run_server(debug=True)
