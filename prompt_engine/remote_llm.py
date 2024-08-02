# remote_llm.py
import requests
import jax
import jax.numpy as jnp

class RemoteLLM:
    def __init__(self, api_url, use_jax=False):
        self.api_url = api_url
        self.use_jax = use_jax

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
        texts = response.json()
        if self.use_jax:
            texts = self._process_with_jax(texts)
        return texts

    def _process_with_jax(self, texts):
        # Example post-processing: convert all text to uppercase using JAX
        def to_upper(text):
            return ''.join([chr(ord(c) - 32) if 'a' <= c <= 'z' else c for c in text])

        v_to_upper = jax.vmap(to_upper, in_axes=(0,))
        texts_jnp = jnp.array(texts)
        upper_texts = v_to_upper(texts_jnp)
        return list(upper_texts)
