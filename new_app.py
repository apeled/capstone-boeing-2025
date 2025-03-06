'''
Note: To use this app, create an "assets" folder in the same directory as this Python file
and save the RankLab logo SVG file as "ranklab-logo.svg" in that folder.

Dash automatically serves files from the assets folder.
'''
import dash
from dash import dcc, html, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import json
import datetime

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# Define custom colors to match the logo
GRAY_COLOR = "#808285"  # Matches the darker gray in the logo
LIGHT_GRAY = "#B0B2B5"  # Matches the lighter gray in the logo

# Define the mapping from Factor# (cell values) to supply chain risk factors
FACTOR_MAPPING = {
    "Factor1": "supplier_financial_stability",
    "Factor2": "geopolitical_disruptions",
    "Factor3": "natural_disasters",
    "Factor4": "transportation_reliability",
    "Factor5": "cybersecurity_vulnerabilities",
    "Factor6": "regulatory_compliance",
    "Factor7": "quality_control_issues",
    "Factor8": "inventory_management",
    "Factor9": "demand_forecasting_accuracy",
    "Factor10": "single_source_dependencies",
    "Factor11": "sustainability_compliance",
    "Factor12": "labor_disruptions",
    "Factor13": "raw_material_price_volatility",
    "Factor14": "intellectual_property_protection",
    "Factor15": "communication_failures",
    "Factor16": "technology_obsolescence",
    "Factor17": "pandemic_health_crisis"
}

# Mapping for display names (human-readable)
FACTOR_DISPLAY_NAMES = {
    "supplier_financial_stability": "Supplier Financial Stability",
    "geopolitical_disruptions": "Geopolitical Disruptions",
    "natural_disasters": "Natural Disasters",
    "transportation_reliability": "Transportation Reliability",
    "cybersecurity_vulnerabilities": "Cybersecurity Vulnerabilities",
    "regulatory_compliance": "Regulatory Compliance",
    "quality_control_issues": "Quality Control Issues",
    "inventory_management": "Inventory Management",
    "demand_forecasting_accuracy": "Demand Forecasting Accuracy",
    "single_source_dependencies": "Single Source Dependencies",
    "sustainability_compliance": "Sustainability Compliance",
    "labor_disruptions": "Labor Disruptions",
    "raw_material_price_volatility": "Raw Material Price Volatility",
    "intellectual_property_protection": "Intellectual Property Protection",
    "communication_failures": "Communication Failures",
    "technology_obsolescence": "Technology Obsolescence",
    "pandemic_health_crisis": "Pandemic or Health Crisis"
}

# Sample supplier IDs
SUPPLIER_IDS = [
    {"label": "32000 (Acme Corp)", "value": 32000},
    {"label": "32001 (XYZ Manufacturing)", "value": 32001},
    {"label": "32002 (Global Logistics Inc)", "value": 32002},
    {"label": "32003 (Tech Components Ltd)", "value": 32003},
    {"label": "32004 (Prime Materials Co)", "value": 32004},
    {"label": "32005 (Innovative Solutions)", "value": 32005},
    {"label": "32006 (FastTrack Shipping)", "value": 32006},
    {"label": "32007 (Quality Assemblies)", "value": 32007},
    {"label": "32008 (Reliable Electronics)", "value": 32008},
    {"label": "32009 (Summit Industries)", "value": 32009},
]

# Custom CSS for consistent colors
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom styles to match logo colors */
            .nav-tabs .nav-link.active {
                background-color: ''' + GRAY_COLOR + ''';
                color: white;
                border-color: ''' + GRAY_COLOR + ''';
            }
            .nav-tabs .nav-link:hover {
                border-color: ''' + LIGHT_GRAY + ''';
            }
            .btn-primary {
                background-color: ''' + GRAY_COLOR + ''';
                border-color: ''' + GRAY_COLOR + ''';
            }
            .btn-primary:hover {
                background-color: ''' + LIGHT_GRAY + ''';
                border-color: ''' + LIGHT_GRAY + ''';
            }
            .dropdown-menu .selected > .dropdown-item {
                background-color: ''' + GRAY_COLOR + ''';
                color: white;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Create the app layout with tabs
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src="/assets/ranklab-logo.svg", height="75px"),
            html.Span("RankLab", style={"marginLeft": "10px", "fontSize": "24px", "fontWeight": "bold"})
        ], style={"display": "flex", "alignItems": "center", "padding": "15px 20px", "borderBottom": "1px solid #e0e0e0"}),
    html.Div([
        dbc.Tabs([
            dbc.Tab(label="Survey", tab_id="survey-tab", label_style={"color": GRAY_COLOR}, active_label_style={"backgroundColor": GRAY_COLOR, "color": "white"}, children=[
                html.Div([
                        html.Div([
                        html.H3("Example: Supply Chain Risk Analysis - 17-Question Survey", className="text-center my-4"),
                    ], className="text-center"),
                    html.P([
                        "Please prioritize the following supply chain risk factors by selecting one option for each question. ",
                        html.Strong("Each risk factor can only be selected once across all questions."),
                        " The system will prevent duplicate selections."
                    ], className="text-center mb-4"),
                    
                    # Store for keeping track of selected answers
                    dcc.Store(id='selected-answers', data={}),
                    
                    dbc.Form([
                        # Add Supplier ID dropdown
                        dbc.Row([
                            dbc.Col([
                                html.Label("Supplier ID:"),
                                dcc.Dropdown(
                                    id="supplier-id",
                                    options=SUPPLIER_IDS,
                                    placeholder="Select a supplier",
                                    className="mb-4"
                                )
                            ], width={"size": 6, "offset": 3})
                        ]),
                        
                        # Add Timestamp field (editable)
                        dbc.Row([
                            dbc.Col([
                                html.Label("Submission Date/Time:"),
                                dbc.Row([
                                    dbc.Col(
                                        dcc.DatePickerSingle(
                                            id="date-picker",
                                            date=datetime.datetime.now().date(),
                                            display_format="YYYY-MM-DD",
                                            className="mb-2"
                                        ),
                                        width=6
                                    ),
                                    dbc.Col(
                                        dcc.Input(
                                            id="time-input",
                                            type="text",
                                            value=datetime.datetime.now().strftime("%H:%M"),
                                            placeholder="HH:MM",
                                            className="form-control mb-2"
                                        ),
                                        width=6
                                    )
                                ]),
                                # Hidden input to store combined date/time value
                                dcc.Store(id="timestamp", data=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                            ], width={"size": 6, "offset": 3})
                        ]),

                        html.Hr(className="my-4"),
                        
                        # Create 17 questions with dropdowns
                        *[
                            dbc.Row([
                                dbc.Col([
                                    html.Label(f"{i+1}. What is your {['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth'][i]} priority supply chain risk factor to address?"),
                                    dcc.Dropdown(
                                        id=f"q{i+1}",
                                        options=[
                                            {"label": "Supplier Financial Stability", "value": "supplier_financial_stability"},
                                            {"label": "Geopolitical Disruptions", "value": "geopolitical_disruptions"},
                                            {"label": "Natural Disasters", "value": "natural_disasters"},
                                            {"label": "Transportation Reliability", "value": "transportation_reliability"},
                                            {"label": "Cybersecurity Vulnerabilities", "value": "cybersecurity_vulnerabilities"},
                                            {"label": "Regulatory Compliance", "value": "regulatory_compliance"},
                                            {"label": "Quality Control Issues", "value": "quality_control_issues"},
                                            {"label": "Inventory Management", "value": "inventory_management"},
                                            {"label": "Demand Forecasting Accuracy", "value": "demand_forecasting_accuracy"},
                                            {"label": "Single Source Dependencies", "value": "single_source_dependencies"},
                                            {"label": "Sustainability Compliance", "value": "sustainability_compliance"},
                                            {"label": "Labor Disruptions", "value": "labor_disruptions"},
                                            {"label": "Raw Material Price Volatility", "value": "raw_material_price_volatility"},
                                            {"label": "Intellectual Property Protection", "value": "intellectual_property_protection"},
                                            {"label": "Communication Failures", "value": "communication_failures"},
                                            {"label": "Technology Obsolescence", "value": "technology_obsolescence"},
                                            {"label": "Pandemic or Health Crisis", "value": "pandemic_health_crisis"}
                                        ],
                                        placeholder="Select a risk factor",
                                        className="mb-4"
                                    )
                                ])
                            ]) for i in range(17)
                        ],
                        
                        # Submit Button
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Submit Survey",
                                    id="submit-button",
                                    color="light",
                                    className="mt-3",
                                    style={"backgroundColor": GRAY_COLOR, "color": "white"}
                                ),
                                html.Div(id="submission-response", className="mt-3")
                            ], width={"size": 6, "offset": 3}, className="text-center")
                        ])
                    ], className="p-4 border rounded")
                ], className="p-4")
            ]),

            dbc.Tab(label="Survey Results", tab_id="results-tab", label_style={"color": GRAY_COLOR}, active_label_style={"backgroundColor": GRAY_COLOR, "color": "white"}, children=[
                html.Div([
                    html.Div([
                        html.H3("Example: Supply Chain Risk Analysis Survey Results", className="text-center my-4"),
                    ], className="text-center"),
                    html.Div(id="survey-results-content")
                ], className="p-4")
            ]),

            dbc.Tab(label="Historical Results", tab_id="historical-results-tab", label_style={"color": GRAY_COLOR}, active_label_style={"backgroundColor": GRAY_COLOR, "color": "white"}, children=[
                html.Div([
                    html.Div([
                        html.H3("Example: Supply Chain Risk Analysis Historical Results", className="text-center my-4"),
                    ], className="text-center"),
                    html.Div(id="results-content")
                ], className="p-4")
            ]),
            
            dbc.Tab(label="About", tab_id="about-tab", label_style={"color": GRAY_COLOR}, active_label_style={"backgroundColor": GRAY_COLOR, "color": "white"}, children=[
                html.Div([
                    html.Div([
                        html.H3("About This Project", className="text-center my-4"),
                    ], className="text-center"),
                    
                    html.H4("The Hawks - UW Data Science Capstone 2024-2025", className="mt-4 mb-3"),
                    
                    html.P([
                        "This application was developed by ",
                        html.Strong("Jake Flynn, Amit Peled, and Mark R Daniel"),
                        " as part of 'The Hawks' Master of Science in Data Science capstone project advisede by Dr. Rookman Song at the University of Washington."
                    ], className="mb-3"),
                    
                    html.P([
                        "Over the course of Fall Quarter 2024 and Winter Quarter 2025, our team performed comprehensive literature reviews on Multicriteria Decision Analysis and developed a model to forecast supplier rankings based on 17 ordinal factors from a dataset synthetically generated by our capstone sponsor, Boeing."
                    ], className="mb-3"),
                    
                    html.P([
                        "We productionalized our model using AWS EC2 and created this interactive Dash application to demonstrate our findings and provide a practical tool for supply chain risk analysis."
                    ], className="mb-3"),
                    
                    html.H4("Application Features", className="mt-4 mb-3"),
                    
                    html.Ul([
                        html.Li("Submit surveys ranking the top 17 drivers of supplier performance in supply chain contexts"),
                        html.Li("View exploratory data analysis of the provided dataset from Boeing"),
                        html.Li("Visualize how rank can change for a supplier over time based on the selected 17 drivers"),
                        html.Li("Compare your rankings with historical data to identify patterns and trends")
                    ], className="mb-4"),
                    
                    html.H4("Survey Methodology", className="mt-4 mb-3"),
                    
                    html.P([
                        "This survey application requires respondents to prioritize 17 different supply chain risk factors. Each risk factor can only be selected once, forcing users to make distinct choices and create a clear prioritization of risk factors."
                    ], className="mb-3"),
                    
                    html.P([
                        "This approach is useful for:"
                    ], className="mb-2"),
                    
                    html.Ul([
                        html.Li("Forcing explicit risk prioritization decisions"),
                        html.Li("Creating a clear ranking of risk factors by importance"),
                        html.Li("Understanding organizational risk perception and priorities"),
                        html.Li("Developing targeted risk mitigation strategies based on priorities"),
                        html.Li("Aligning teams on what risks matter most")
                    ], className="mb-4"),
                    
                    html.P([
                        "The application uses real historical survey data to show average rankings across multiple respondents. This helps identify which risk factors are commonly prioritized highly across different organizations."
                    ], className="mb-3"),
                    
                    html.H4("Acknowledgements", className="mt-4 mb-3"),
                    
                    html.P([
                        "We would like to thank Boeing for sponsoring this project and providing the synthetic dataset that made this research possible."
                    ], className="mb-4"),
                    
                ], className="p-4")
            ])
        ], id="tabs", active_tab="survey-tab"),
    ], className="container mt-4")
])])

# No need for the timestamp update callback since it's now editable by the user

# Initialize an empty DataFrame to store responses
responses_df = pd.DataFrame()

# Load and process existing CSV data on startup
try:
    # Load the CSV file
    existing_data = pd.read_csv('data/Rank.csv')
    
    # Create a dataframe to hold processed results
    factor_results = []
    
    # For each factor, calculate average position
    for factor_name, risk_value in FACTOR_MAPPING.items():
        # Find all positions for this factor across all Driver columns
        positions = []
        for i in range(1, 18):
            driver_col = f'Driver{i}'
            # Count occurrences where this factor appears in this position
            factor_at_position = existing_data[existing_data[driver_col] == factor_name]
            if not factor_at_position.empty:
                positions.extend([i] * len(factor_at_position))
        
        # Only process if we found this factor in the data
        if positions:
            avg_rank = sum(positions) / len(positions)
            times_ranked_first = len(existing_data[existing_data['Driver1'] == factor_name])
            
            factor_results.append({
                'risk_factor': FACTOR_DISPLAY_NAMES[risk_value],
                'risk_value': risk_value,
                'average_rank': avg_rank,
                'top_rank_count': times_ranked_first,
                'data_count': len(positions)
            })
    
    # Convert to DataFrame
    historical_results_df = pd.DataFrame(factor_results)
    
except Exception as e:
    print(f"Error loading historical data: {e}")
    historical_results_df = pd.DataFrame(columns=['risk_factor', 'risk_value', 'average_rank', 'top_rank_count', 'data_count'])

# Callback to update dropdown options based on previously selected values
@app.callback(
    [Output(f"q{i+1}", "options") for i in range(17)],
    [Input(f"q{i+1}", "value") for i in range(17)],
    [State('selected-answers', 'data')],
)
def update_dropdowns(*args):
    # Extract current values and store data
    current_values = args[:17]
    selected_data = args[17] or {}
    
    # Get the component that triggered the callback
    triggered_id = ctx.triggered_id or ""
    
    # Update selected_data based on the new selection
    if triggered_id:
        question_number = int(triggered_id[1:]) if triggered_id[0] == 'q' else 0
        if question_number > 0:
            # Find the old value that might have been deselected
            old_value = selected_data.get(triggered_id)
            # Update with new value
            if current_values[question_number-1]:
                selected_data[triggered_id] = current_values[question_number-1]
            else:
                if triggered_id in selected_data:
                    del selected_data[triggered_id]
    
    # Collect all currently selected values
    all_selected = [v for v in current_values if v]
    
    # Base options for all dropdowns - 17 supply chain risk factors
    all_options = [
        {"label": "Supplier Financial Stability", "value": "supplier_financial_stability"},
        {"label": "Geopolitical Disruptions", "value": "geopolitical_disruptions"},
        {"label": "Natural Disasters", "value": "natural_disasters"},
        {"label": "Transportation Reliability", "value": "transportation_reliability"},
        {"label": "Cybersecurity Vulnerabilities", "value": "cybersecurity_vulnerabilities"},
        {"label": "Regulatory Compliance", "value": "regulatory_compliance"},
        {"label": "Quality Control Issues", "value": "quality_control_issues"},
        {"label": "Inventory Management", "value": "inventory_management"},
        {"label": "Demand Forecasting Accuracy", "value": "demand_forecasting_accuracy"},
        {"label": "Single Source Dependencies", "value": "single_source_dependencies"},
        {"label": "Sustainability Compliance", "value": "sustainability_compliance"},
        {"label": "Labor Disruptions", "value": "labor_disruptions"},
        {"label": "Raw Material Price Volatility", "value": "raw_material_price_volatility"},
        {"label": "Intellectual Property Protection", "value": "intellectual_property_protection"},
        {"label": "Communication Failures", "value": "communication_failures"},
        {"label": "Technology Obsolescence", "value": "technology_obsolescence"},
        {"label": "Pandemic or Health Crisis", "value": "pandemic_health_crisis"}
    ]
    
    # For each dropdown, disable options that are already selected elsewhere
    result = []
    for i in range(17):
        # Current value for this dropdown
        current_value = current_values[i]
        
        # Create filtered options - disable those selected elsewhere
        options = []
        for option in all_options:
            # If this option is selected in another dropdown, disable it
            if option["value"] in all_selected and option["value"] != current_value:
                options.append({
                    "label": option["label"],
                    "value": option["value"],
                    "disabled": True
                })
            else:
                options.append(option)
        
        result.append(options)
    
    return result

# Store the selected answers
@app.callback(
    Output('selected-answers', 'data'),
    [Input(f"q{i+1}", "value") for i in range(17)],
    [State('selected-answers', 'data')]
)
def store_selected_answers(*args):
    current_values = args[:17]
    selected_data = args[17] or {}
    
    # Update the stored data
    for i, value in enumerate(current_values):
        question_id = f"q{i+1}"
        if value:
            selected_data[question_id] = value
        elif question_id in selected_data and not value:
            del selected_data[question_id]
    
    return selected_data

# Callback to update timestamp when date or time changes
@app.callback(
    Output("timestamp", "data"),
    [Input("date-picker", "date"), 
     Input("time-input", "value")]
)
def update_timestamp(date_value, time_value):
    if not date_value or not time_value:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Validate time format
    try:
        # Try to parse the time to ensure it's in a valid format
        hours, minutes = time_value.split(":")
        hours, minutes = int(hours), int(minutes)
        if hours < 0 or hours > 23 or minutes < 0 or minutes > 59:
            time_value = datetime.datetime.now().strftime("%H:%M")
    except:
        time_value = datetime.datetime.now().strftime("%H:%M")
    
    return f"{date_value} {time_value}"

@app.callback(
    Output("submission-response", "children"),
    Input("submit-button", "n_clicks"),
    [State("supplier-id", "value"), 
     State("timestamp", "data"),
     *[State(f"q{i+1}", "value") for i in range(17)]],
    prevent_initial_call=True
)
def submit_survey(n_clicks, supplier_id, timestamp, *args):
    global responses_df
    
    # Check if supplier ID is selected
    if not supplier_id:
        return dbc.Alert("Please select a Supplier ID before submitting.", color="danger", style={"borderColor": GRAY_COLOR})
    
    # Check if all questions are answered
    if None in args or '' in args:
        return dbc.Alert("Please answer all questions before submitting.", color="danger", style={"borderColor": GRAY_COLOR})
    
    # Ensure no duplicate answers
    if len(set(args)) != 17:
        return dbc.Alert("Please ensure each risk factor is selected only once.", color="danger", style={"borderColor": GRAY_COLOR})
    
    # Create a response dictionary
    response = {
        "SubjectID": supplier_id,
        **{f"Driver{i+1}": value for i, value in enumerate(args)},
        "UpdateDT": timestamp,
    }
    
    # Add ranks for each risk factor
    risk_factors = list(FACTOR_MAPPING.values())
    
    for factor in risk_factors:
        if factor in args:
            response[factor] = args.index(factor) + 1
        else:
            response[factor] = None
    
            # Timestamp is already in the right format
    responses_df = pd.concat([responses_df, pd.DataFrame([response])], ignore_index=True)
    
    return dbc.Alert("Survey submitted successfully! Thank you for your feedback.", color="light", style={"backgroundColor": LIGHT_GRAY, "color": "white"})

# Display results from the historical data
@app.callback(
    Output("results-content", "children"),
    [Input("tabs", "active_tab"), Input("submit-button", "n_clicks")],
)
def update_historical_results(active_tab, n_clicks):
    if active_tab != "historical-results-tab":
        return dash.no_update

    # If historical data is available, use it
    if not historical_results_df.empty:
        # Sort by average rank for display
        sorted_results = historical_results_df.sort_values('average_rank')
        
        results_content = [
            dbc.Row([
                dbc.Col([
                    html.H4("Historical Data: Average Ranking by Supply Chain Risk Factor", className="text-center"),
                    dcc.Graph(
                        figure=px.bar(
                            sorted_results, y='risk_factor', x='average_rank',
                            labels={
                                'risk_factor': 'Risk Factor',
                                'average_rank': 'Average Rank (lower is better)'
                            },
                            title="Risk Factors by Average Ranking from Historical Data",
                            orientation='h'
                        ).update_layout(yaxis={'categoryorder': 'total ascending'})
                    )
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Historical Data: Top Priority Risk Factors", className="text-center"),
                    dcc.Graph(
                        figure=px.bar(
                            sorted_results.sort_values('top_rank_count', ascending=False), 
                            y='risk_factor', 
                            x='top_rank_count',
                            labels={
                                'risk_factor': 'Risk Factor',
                                'top_rank_count': 'Times Ranked #1 Priority'
                            },
                            title="Number of Times Each Risk Factor Was Ranked as #1 Priority",
                            orientation='h'
                        ).update_layout(yaxis={'categoryorder': 'total ascending'})
                    )
                ], width=12)
            ]),
            html.H4("Survey Statistics", className="text-center mt-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Total Historical Responses", className="card-title text-center"),
                            html.P(historical_results_df.iloc[0]['data_count'] if not historical_results_df.empty else "0", 
                                   className="card-text text-center display-4")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Historically Top Priority", className="card-title text-center"),
                            html.P(
                                sorted_results.iloc[0]['risk_factor'] if not sorted_results.empty else "N/A", 
                                className="card-text text-center"
                            )
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Historically Lowest Priority", className="card-title text-center"),
                            html.P(
                                sorted_results.iloc[-1]['risk_factor'] if not sorted_results.empty else "N/A", 
                                className="card-text text-center"
                            )
                        ])
                    ])
                ], width=4)
            ])
        ]
    
    return results_content

# Display results from the historical data
@app.callback(
    Output("survey-results-content", "children"),
    [Input("tabs", "active_tab"), Input("submit-button", "n_clicks")],
)
def update_historical_results(active_tab, n_clicks):
    if active_tab != "historical-results-tab":
        return dash.no_update

        # If there are new responses, add a section for them
        if not responses_df.empty:
            # Create visualization data for new responses
            viz_data = []
            risk_factors = list(FACTOR_MAPPING.values())
            
            for factor in risk_factors:
                factor_data = {
                    'risk_factor': FACTOR_DISPLAY_NAMES[factor],
                    'average_rank': responses_df[factor].mean()
                }
                # Count how many times this was ranked #1
                top_rank_count = sum(responses_df[factor] == 1)
                factor_data['top_rank_count'] = top_rank_count
                viz_data.append(factor_data)
            
            new_viz_df = pd.DataFrame(viz_data)
            new_viz_df = new_viz_df.sort_values('average_rank')
            
            new_results = [
                html.Hr(),
                html.H3("Your Session Results", className="text-center my-4"),
                dbc.Row([
                    dbc.Col([
                        html.H4("Session Data: Average Ranking by Supply Chain Risk Factor", className="text-center"),
                        dcc.Graph(
                            figure=px.bar(
                                new_viz_df, y='risk_factor', x='average_rank',
                                labels={
                                    'risk_factor': 'Risk Factor',
                                    'average_rank': 'Average Rank (lower is better)'
                                },
                                title="Risk Factors by Average Ranking (Session Data)",
                                orientation='h'
                            ).update_layout(yaxis={'categoryorder': 'total ascending'})
                        )
                    ], width=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Session Responses", className="card-title text-center"),
                                html.P(len(responses_df), className="card-text text-center display-4")
                            ])
                        ])
                    ], width=12)
                ])
            ]
            
            results_content.extend(new_results)
    else:
        # If no historical data, just display a message or any session data
        if responses_df.empty:
            results_content = [
                html.P("No survey responses yet. Submit a survey to see results.", className="text-center")
            ]
        else:
            # Create visualization just from session data
            viz_data = []
            risk_factors = list(FACTOR_MAPPING.values())
            
            for factor in risk_factors:
                factor_data = {
                    'risk_factor': FACTOR_DISPLAY_NAMES[factor],
                    'average_rank': responses_df[factor].mean()
                }
                # Count how many times this was ranked #1
                top_rank_count = sum(responses_df[factor] == 1)
                factor_data['top_rank_count'] = top_rank_count
                viz_data.append(factor_data)
            
            viz_df = pd.DataFrame(viz_data)
            viz_df = viz_df.sort_values('average_rank')
            
            results_content = [
                html.H3("Session Results", className="text-center my-4"),
                dbc.Row([
                    dbc.Col([
                        html.H4("Average Ranking by Supply Chain Risk Factor", className="text-center"),
                        dcc.Graph(
                            figure=px.bar(
                                viz_df, y='risk_factor', x='average_rank',
                                labels={
                                    'risk_factor': 'Risk Factor',
                                    'average_rank': 'Average Rank (lower is better)'
                                },
                                title="Risk Factors by Average Ranking",
                                orientation='h'
                            ).update_layout(yaxis={'categoryorder': 'total ascending'})
                        )
                    ], width=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Session Responses", className="card-title text-center"),
                                html.P(len(responses_df), className="card-text text-center display-4")
                            ])
                        ])
                    ], width=12)
                ])
            ]



# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
