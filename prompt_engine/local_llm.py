from transformers import pipeline, set_seed

class LocalLLM:
    def __init__(self, model_name):
        self.generator = pipeline('text-generation', model=model_name)
        set_seed(42)

    def generate_text(self, prompt, max_length, num_return_sequences):
        return self.generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
