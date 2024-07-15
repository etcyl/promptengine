import os
import time
from dash import dcc, html, Input, Output, State
import dash
import dash_daq as daq
import dash_uploader as du
from prompt_engine.llm_interface import LLMInterface

# Temporary override for dash_uploader's deprecated imports
import dash_uploader.configure_upload as configure_upload
configure_upload.html = html

# Define initial model name and set up the LLM interface
model_name = 'gpt2'
llm_interface = LLMInterface(model_name=model_name, remote=False)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])
app.title = "LLM Visualization Dashboard"
du.configure_upload(app, r'./uploads')

# List of available models
available_models = [
    {'label': 'GPT-2', 'value': 'gpt2'},
    {'label': 'GPT-3', 'value': 'gpt3'},
    {'label': 'DistilGPT-2', 'value': 'distilgpt2'},
    {'label': 'Custom Local Model', 'value': 'local'}
]

# Create Layout
app.layout = html.Div([
    html.H1("LLM Visualization Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),
    html.Div(id='current-model', children=f"Current model: {model_name}", style={'textAlign': 'center', 'marginBottom': 20}),
    html.Div([
        html.Div([
            dcc.Textarea(
                id='query-input',
                placeholder='Enter your prompt here...',
                style={'width': '100%', 'height': 200, 'marginBottom': '10px'}
            ),
            html.Div(id='processing-message', style={'marginBottom': '10px'}),
            daq.ColorPicker(
                id='color-picker',
                label='Pick Submit Button Color',
                value=dict(hex='#4CAF50'),
                size=150,
                style={'marginBottom': '10px'}
            ),
            html.Div([
                html.Button('Submit', id='submit-button', n_clicks=0, style={'padding': '5px 10px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px'}),
            ], style={'display': 'flex', 'justifyContent': 'center'}),
        ], className='six columns', style={'padding': '10px', 'borderRight': '1px solid #ccc'}),
        html.Div([
            html.H4("Model Output", style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.Div(id='summary-output', style={'whiteSpace': 'pre-line', 'border': '1px solid #ccc', 'padding': '10px', 'height': '200px', 'overflowY': 'scroll'})
        ], className='six columns', style={'padding': '10px'})
    ], className='row'),
    html.Div([
        html.Label("Select a model from the drop-down list below:", style={'fontSize': '16px', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='model-dropdown',
            options=available_models,
            value='gpt2',
            style={'marginBottom': '10px'}
        ),
        du.Upload(
            id='upload-data',
            text='Drag and Drop or Click to Select a Local Model',
            max_files=1,
            filetypes=['.bin', '.pt'],
            default_style={'marginBottom': '10px'}
        ),
        html.Button('Change Model', id='change-model-button', n_clicks=0, style={'marginTop': '10px'})
    ], style={'textAlign': 'center', 'padding': '10px'})
])

# Callback to change button color
@app.callback(
    Output('submit-button', 'style'),
    Input('color-picker', 'value')
)
def update_button_color(color):
    return {'padding': '5px 10px', 'backgroundColor': color['hex'], 'color': 'white', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px'}

# Callback to process LLM response
@app.callback(
    Output('summary-output', 'children'),
    Output('processing-message', 'children'),
    Input('submit-button', 'n_clicks'),
    State('query-input', 'value')
)
def update_summary(n_clicks, query):
    if n_clicks > 0 and query:
        start_time = time.time()
        processing_message = "Processing request..."
        summary = llm_interface.generate_text(query, max_length=500, num_return_sequences=1)
        response_time = time.time() - start_time
        processing_message = f"Response from LLM took {response_time:.2f} seconds"
        return summary[0]['generated_text'], processing_message
    return "", ""

# Callback to change the model
@app.callback(
    Output('current-model', 'children'),
    Input('change-model-button', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('upload-data', 'isCompleted'),
    State('upload-data', 'fileNames')
)
def change_model(n_clicks, selected_model, is_completed, file_names):
    global llm_interface, model_name
    if n_clicks > 0:
        if selected_model == 'local' and is_completed and file_names:
            # Assume the uploaded model is stored in the ./uploads directory
            model_path = os.path.join('./uploads', file_names[0])
            llm_interface = LLMInterface(model_name=model_path, remote=False)
            model_name = f'Custom Local Model ({file_names[0]})'
        else:
            llm_interface = LLMInterface(model_name=selected_model, remote=False)
            model_name = selected_model
        return f"Current model: {model_name}"
    return f"Current model: {model_name}"

# Run the App
if __name__ == '__main__':
    app.run_server(debug=True)
