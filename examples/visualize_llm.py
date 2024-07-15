import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from dash import dcc, html, Input, Output
import dash
import plotly.express as px
from prompt_engine.llm_interface import LLMInterface

# Initialize the LLM interface (use remote or local model as needed)
llm_interface = LLMInterface(model_name='gpt2', remote=False)

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Create Layout
app.layout = html.Div([
    html.H1("Car Dataset Explorer"),
    dcc.Dropdown(
        id='x-axis-dropdown',
        options=[{'label': col, 'value': col} for col in data.columns],
        value='horsepower'
    ),
    dcc.Dropdown(
        id='y-axis-dropdown',
        options=[{'label': col, 'value': col} for col in data.columns],
        value='mpg'
    ),
    dcc.Graph(id='scatter-plot'),
    html.Div([
        dcc.Textarea(
            id='query-input',
            value='Summarize the dataset',
            style={'width': '100%', 'height': 100}
        ),
        html.Button('Submit', id='submit-button', n_clicks=0),
        html.Div(id='summary-output')
    ])
])

# Create Callbacks for Interactions
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value')
)
def update_scatter_plot(x_axis, y_axis):
    fig = px.scatter(data, x=x_axis, y=y_axis, color='origin')
    return fig

@app.callback(
    Output('summary-output', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('query-input', 'value')
)
def update_summary(n_clicks, query):
    if n_clicks > 0:
        summary = llm_interface.generate_text(query, max_length=50, num_return_sequences=1)
        return summary[0]['generated_text']
    return ""

# Run the App
if __name__ == '__main__':
    app.run_server(debug=True)
