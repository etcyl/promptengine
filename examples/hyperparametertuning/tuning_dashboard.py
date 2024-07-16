import pandas as pd
from sklearn.model_selection import train_test_split
from prompt_engine import LLMInterface
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = load_iris()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LLMInterface
llm_interface = LLMInterface(model_name='gpt2')

def objective(trial):
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 10, 200)
    rf_max_depth = trial.suggest_int('rf_max_depth', 1, 20)
    gb_n_estimators = trial.suggest_int('gb_n_estimators', 10, 200)
    gb_learning_rate = trial.suggest_loguniform('gb_learning_rate', 0.01, 0.1)
    
    rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    
    score = cross_val_score(ensemble, X_train, y_train, cv=5, n_jobs=-1).mean()
    return score

# Run Bayesian Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Summarize results
best_params = study.best_params
best_value = study.best_value

logger.info(f"Best hyperparameters: {best_params}")
logger.info(f"Best cross-validation score: {best_value:.2f}")

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Hyperparameter Tuning with LLM Support"),
    html.Div([
        html.Label("Learning Rate:"),
        dcc.Slider(id='learning_rate', min=1e-5, max=1e-1, step=1e-5, value=3e-5,
                   marks={i/100: str(i/100) for i in range(1, 11)}),
        html.Label("Batch Size:"),
        dcc.Slider(id='batch_size', min=8, max=64, step=8, value=32, marks={i: str(i) for i in range(8, 65, 8)}),
    ], style={'padding': '20px'}),
    html.Button('Optimize', id='optimize-button', n_clicks=0, style={'margin': '20px'}),
    dcc.Graph(id='performance_graph'),
    html.Div(id='llm-suggestion', style={'whiteSpace': 'pre-line', 'padding': '20px', 'border': '1px solid #ddd'})
])

@app.callback(
    Output('performance_graph', 'figure'),
    Output('llm-suggestion', 'children'),
    Input('optimize-button', 'n_clicks'),
    Input('learning_rate', 'value'),
    Input('batch_size', 'value')
)
def update_graph(n_clicks, learning_rate, batch_size):
    if n_clicks > 0:
        prompt = f"Optimize a machine learning model with learning_rate={learning_rate} and batch_size={batch_size}."
        response = llm_interface.generate_text(prompt, max_length=150, num_return_sequences=1)

        # Verify response is correctly unpacked
        suggestion = response if isinstance(response, str) else response[0]

        # Dummy performance data for illustration
        epochs = list(range(1, 11))
        accuracy = [0.8 + 0.01*epoch for epoch in epochs]

        # Create and update the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=accuracy, mode='lines+markers', name='Accuracy'))
        fig.update_layout(
            title=f"Performance with learning_rate={learning_rate} and batch_size={batch_size}",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            hovermode="closest"  # Improved interactivity with hover information
        )

        return fig, suggestion
    
    return go.Figure(), "Click the 'Optimize' button to get suggestions from the LLM."

if __name__ == '__main__':
    app.run_server(debug=True)
