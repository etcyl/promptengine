import openai

class OpenAILLM:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_text(self, prompt, max_length, num_return_sequences):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_length,
            n=num_return_sequences
        )
        return [choice['text'].strip() for choice in response.choices]
