# restful_remote_llm.py
from flask import Flask, request, jsonify
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ensure the prompt_engine directory is in the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define a function to download and load the model
def load_model(model_name='gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Initialize Flask app
app = Flask(__name__)

# Load the model
tokenizer, model, device = load_model()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt')
    max_length = data.get('max_length', 50)
    num_return_sequences = data.get('num_return_sequences', 1)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 50)
    top_p = data.get('top_p', 0.9)

    try:
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences, temperature=temperature, top_k=top_k, top_p=top_p)
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return jsonify(generated_texts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/grid_search', methods=['POST'])
def grid_search():
    data = request.get_json()
    model = data.get('model')
    param_grid = data.get('param_grid')
    X = data.get('X')
    y = data.get('y')

    try:
        best_params, best_score = llm_interface.grid_search(model, param_grid, X, y)
        return jsonify({'best_params': best_params, 'best_score': best_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bayesian_optimization', methods=['POST'])
def bayesian_optimization():
    data = request.get_json()
    objective = data.get('objective')
    n_trials = data.get('n_trials', 50)

    try:
        best_params, best_value = llm_interface.bayesian_optimization(objective, n_trials)
        return jsonify({'best_params': best_params, 'best_value': best_value})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json()
    model = data.get('model')
    X_train = data.get('X_train')
    y_train = data.get('y_train')

    try:
        trained_model = llm_interface.train_model(model, X_train, y_train)
        return jsonify({'trained_model': trained_model})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_model', methods=['POST'])
def evaluate_model():
    data = request.get_json()
    model = data.get('model')
    X_test = data.get('X_test')
    y_test = data.get('y_test')

    try:
        score = llm_interface.evaluate_model(model, X_test, y_test)
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
