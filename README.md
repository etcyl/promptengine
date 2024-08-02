# Description

A Python package for interacting with Large Language Models (LLMs) both locally and remotely. It's designed to be easy to use for data scientists and AI researchers who need a generic interface for generating text and other NLP tasks.

## Features

- **Local and Remote Operation**: Run models locally or via a RESTful API.
- **Flexible and Generic**: Customize operations with a wide range of parameters.
- **Easy Integration**: Designed to fit seamlessly into existing Python environments.

## Installation

Clone this repository and install:

```git clone <link> && cd promptengine```

Create a Python virtual environment:

```python -m venv promptengine```

Activate the virtual environment:
- On Windows:

  ```source promptengine/Scripts/activate```
  
- On MacOS/Linux:
  
  ```source promptengine/bin/activate```

Using Conda:

```conda create --name promptengine python=3.9```

```conda activate promptengine```

Install the package:

```pip install -r ../requirements.txt```

## Usage

### Local Usage

```python
from prompt_engine import LLMInterface

engine = LLMInterface(model_name='gpt2')
print(engine.generate_text("Hello, world!"))
```

### Remote Usage
```python
from prompt_engine import LLMInterface

engine = LLMInterface(remote=True, api_url="http://api.yourserver.com")
print(engine.generate_text("Hello, world!"))
```

### Examples
Using JAX for Machine Learning: Example Walkthrough

This section of the README guides you through running the jax_example.py script, which demonstrates the integration of JAX in machine learning applications using our system. JAX is a high-performance library designed for high-speed numerical computing and machine learning. It extends NumPy and automatic differentiation capabilities with GPU and TPU acceleration, making it an excellent tool for applications that require efficient mathematical transformations.

1. Ensure that the `OPENAI_API_KEY` is set in your environment:
    ```export OPENAI_API_KEY='your_openai_api_key_here'```

2. Run the script: ```python jax_example.py```

Expected Output

The script will output the prompt provided, the model used for generating responses, and the model's output, formatted for readability. Here's what you should expect:

```
PROMPT: Explain the benefits of using JAX for machine learning.
MODEL: gpt-4
OUTPUT:
JAX IS A HIGH-PERFORMANCE MACHINE LEARNING LIBRARY THAT COMBINES THE POWER OF NUMPY,
AUTOMATIC DIFFERENTIATION, AND HARDWARE ACCELERATION ON GPUS AND TPUS. THIS ENABLES IT
TO PERFORM LARGE-SCALE AND HIGH-SPEED COMPUTATIONS THAT ARE ESSENTIAL FOR TRAINING
MODERN MACHINE LEARNING MODELS.
```

Why Use JAX?

JAX is particularly useful for projects that require:

    Speed and Efficiency: JAX can execute large-scale numerical computations much faster than traditional CPU-bound libraries. This is crucial for training deep learning models where matrix calculations are extensive and frequent.
    Automatic Differentiation: JAX facilitates the easy calculation of derivatives, enabling automatic gradient calculations for optimization algorithms commonly used in machine learning.
    GPU/TPU Acceleration: Leveraging hardware acceleration, JAX can significantly reduce the time required for model training and inference, which is beneficial for projects with heavy computational loads.

By integrating JAX, this example aims to showcase how you can harness these capabilities for enhanced performance in machine learning tasks. The script is designed to be a simple demonstration, emphasizing JAX's application in generating and post-processing machine learning outputs.

RESTful API

This example demonstrates how to set up a RESTful API using Flask to interact with a locally downloaded language model (LLM). T
he example shows how to download a model using the transformers library and exposes endpoints to generate text using the model.

To run this example:
1. ```cd examples && python examples/restful_remote_llm.py```

This script sets up a Flask server with endpoints to interact with a language model. It includes the following functionalities:

    Model Download and Loading:
    The script downloads the gpt2 model from Hugging Face and loads it into memory for inference.

    Flask API Endpoints:
        /generate: Generates text based on the provided prompt and parameters.
        /grid_search: Placeholder for grid search functionality (to be implemented).
        /bayesian_optimization: Placeholder for Bayesian optimization functionality (to be implemented).
        /train_model: Placeholder for model training functionality (to be implemented).
        /evaluate_model: Placeholder for model evaluation functionality (to be implemented).

Use Postman or ```curl``` to interact with the API:
```
curl -X POST "http://localhost:5000/generate" -H "Content-Type: application/json" -d '{
  "prompt": "Once upon a time",
  "max_length": 50,
  "num_return_sequences": 1,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9
}'
```

Expected output:
When you send a request to the /generate endpoint, you should receive a JSON response containing the generated text. The response will vary based on the input prompt and parameters.

Example output:
For the given request with the prompt "Once upon a time", the response might look like:
```["Once upon a time, in a land far away, there lived a princess who..."]```

### Running with Docker Compose
1. Build the Docker Image:
```docker-compose build```

2. Run the Docker Container:
```docker-compose up```

3. Kill the last Docker Container:
```docker stop $(docker ps -q | head -n 1) && docker ps```

Generating Fuzz Tests with Atheris

The example script examples/generate_fuzz.py will download two GPT models for writing Atheris fuzz tests in Python.
Run this example:
```cd examples && python generate_fuzz.py```

### Running with Docker Compose
1. Build the Docker Image:
```docker-compose build```

2. Run the Docker Container:
```docker-compose up```

3. Kill the last Docker Container:
```docker stop $(docker ps -q | head -n 1) && docker ps```

LLM Visualization Dashboard

This example demonstrates a dashboard for visualizing and interacting with Large Language Models (LLMs) using Dash. Users can input prompts, select different models, and view the generated text output. The dashboard supports both local and remote LLMs, including OpenAI's GPT-3.

- Model Selection: switch between GPT-2, GPT-3, DistilGPT-2, or a custom local model you select. 
- Interactive Input: enter prompts and customize the submit button color.
- Output Display: view the generated text and processing time. 

1. ```python visualize_llm_example2.py```

2. Access the Dashboard: open your browser and go to http://127.0.0.1:8050/.

3. Expected Results:
- Model Selection: Drop-down menu to choose the LLM.
- Prompt Input: Text area for entering prompts.
- Color Picker: Tool to customize the submit button color.
- Generated Output: Display area showing the text generated by the LLM and the time taken for processing.
- The flow diagram visualizes the data flow between these components, highlighting how user inputs are processed and results are displayed.