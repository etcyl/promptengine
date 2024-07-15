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

### Example
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
