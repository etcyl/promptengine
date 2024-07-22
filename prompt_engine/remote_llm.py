import requests

class RemoteLLM:
    def __init__(self, api_url):
        self.api_url = api_url

    def generate_text(self, prompt, max_length, num_return_sequences, temperature=0.7, top_k=50, top_p=0.9):
        data = {
            'prompt': prompt,
            'max_length': max_length,
            'num_return_sequences': num_return_sequences,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        }
        response = requests.post(f"{self.api_url}/generate", json=data)
        return response.json()
