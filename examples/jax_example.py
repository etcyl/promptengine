import os
from prompt_engine import LLMInterface
import textwrap  # To format output more prettily

# Fetch OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize the interface with JAX support and OpenAI
engine = LLMInterface(use_openai=True, openai_api_key=openai_api_key, use_jax=True)

# Example prompt
prompt = "Explain the benefits of using JAX for machine learning."

# Generate text with JAX post-processing
response = engine.generate_text(prompt, max_length=50, num_return_sequences=1, temperature=0.7)

# Model used
model_used = "gpt-4"  # This should ideally come from the engine if it's dynamic

# Format and print the output more prettily
formatted_response = textwrap.fill(response[0], width=80)  # Wrap text at 80 characters
print(f"PROMPT: {prompt}")
print(f"MODEL: {model_used}")
print("OUTPUT:")
print(formatted_response)
