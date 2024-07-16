import requests

class RemoteLLM:
    def __init__(self, api_url):
        self.api_url = api_url

    def generate_text(self, prompt, max_length, num_return_sequences):
        data = {'prompt': prompt, 'max_length': max_length, 'num_return_sequences': num_return_sequences}
        response = requests.post(f"{self.api_url}/generate", json=data)
        return response.json()
