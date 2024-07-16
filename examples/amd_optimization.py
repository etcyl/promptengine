from llm_interface import LLMInterface, objective

# Initialize the interface with AMD optimization if available
engine = LLMInterface(model_name='gpt3', use_amd_optimization=True)

# Example prompt
prompt = "Discuss the impact of AI on modern computing."
response = engine.generate_text(prompt)
print("Generated Text:", response)
