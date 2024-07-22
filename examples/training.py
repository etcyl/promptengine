# Updated Dash Application for Model Training and Deployment with LLM Integration

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from prompt_engine import LLMInterface
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import logging
import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LLMInterface
llm_interface = LLMInterface(model_name='gpt2')

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Model Training and Deployment"

app.layout = html.Div([
    html.H1("Model Training and Deployment Interface", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Enter your prompt for model suggestion:", style={'fontSize': '16px', 'fontWeight': 'bold'}),
        dcc.Textarea(
            id='user-prompt',
            placeholder='Enter your prompt here...',
            style={'width': '100%', 'height': 100, 'marginBottom': '10px'}
        ),
        html.Button('Submit Prompt', id='submit-prompt', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px'}),
        html.Div(id='model-suggestion', style={'whiteSpace': 'pre-line', 'marginTop': '20px', 'fontSize': '16px'})
    ], style={'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'marginBottom': '20px'}),
    html.Div([
        html.Label("Train and Evaluate Model:", style={'fontSize': '16px', 'fontWeight': 'bold'}),
        html.Button('Train Model', id='train-model', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px', 'marginBottom': '10px'}),
        html.Div(id='training-status', style={'marginTop': '10px', 'fontSize': '16px'}),
        dcc.Graph(id='performance-graph', style={'marginTop': '20px'})
    ], style={'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'marginBottom': '20px'}),
    html.Div([
        html.Label("Deploy Model:", style={'fontSize': '16px', 'fontWeight': 'bold'}),
        dcc.Input(id='model-name', type='text', placeholder='Enter model name', style={'marginRight': '10px'}),
        html.Button('Deploy Model', id='deploy-model', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px'}),
        html.Div(id='deployment-status', style={'marginTop': '10px', 'fontSize': '16px'})
    ], style={'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px'}),
    html.Div([
        html.H3("Flow of Information", style={'textAlign': 'center', 'marginTop': '20px'}),
        html.Img(src='/assets/flowchart.png', style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '50%'})
    ], style={'padding': '10px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'marginTop': '20px'})
])

@app.callback(
    Output('model-suggestion', 'children'),
    Input('submit-prompt', 'n_clicks'),
    State('user-prompt', 'value')
)
def suggest_model(n_clicks, user_prompt):
    if n_clicks > 0 and user_prompt:
        suggestion = llm_interface.interactive_prompt(user_prompt)
        return f"Model Suggestion:\n{suggestion}"
    return "Awaiting model suggestion..."

@app.callback(
    Output('training-status', 'children'),
    Output('performance-graph', 'figure'),
    Input('train-model', 'n_clicks')
)
def train_and_evaluate_model(n_clicks):
    if n_clicks > 0:
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        trained_model = llm_interface.train_model(model, X_train, y_train)
        logger.info("Model training completed.")

        # Evaluate model
        evaluation_score = llm_interface.evaluate_model(trained_model, X_test, y_test)
        logger.info(f"Model evaluation score: {evaluation_score}")

        # Create performance graph
        epochs = list(range(1, 11))
        accuracy = [evaluation_score for _ in epochs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=accuracy, mode='lines+markers', name='Accuracy'))
        fig.update_layout(title="Model Performance Over Epochs", xaxis_title="Epoch", yaxis_title="Accuracy")

        return f"Model training completed. Evaluation score: {evaluation_score}", fig
    return "Awaiting model training...", {}

@app.callback(
    Output('deployment-status', 'children'),
    Input('deploy-model', 'n_clicks'),
    State('model-name', 'value')
)
def deploy_model(n_clicks, model_name):
    if n_clicks > 0 and model_name:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        trained_model = llm_interface.train_model(model, X_train, y_train)
        deployment_path = llm_interface.deploy_model(trained_model, model_name)
        return f"Model deployed at: {deployment_path}"
    return "Awaiting model deployment..."

if __name__ == '__main__':
    app.run_server(debug=True)

