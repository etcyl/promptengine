from prompt_engine.local_llm import LocalLLM
from prompt_engine.remote_llm import RemoteLLM
from prompt_engine.openai_llm import OpenAILLM

class LLMInterface:
    def __init__(self, model_name='gpt2', remote=False, api_url=None, use_openai=False, openai_api_key=None):
        self.remote = remote
        if use_openai and openai_api_key:
            self.handler = OpenAILLM(openai_api_key)
        elif self.remote and api_url:
            self.handler = RemoteLLM(api_url)
        else:
            self.handler = LocalLLM(model_name)

    def generate_text(self, prompt, max_length=50, num_return_sequences=1):
        return self.handler.generate_text(prompt, max_length, num_return_sequences)
