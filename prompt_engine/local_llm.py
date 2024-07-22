from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalLLM:
    def __init__(self, model_name='gpt2', use_amd_optimization=False, use_intel_optimization=False, seed=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if seed:
            torch.manual_seed(seed)

    def generate_text(self, prompt, max_length=50, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences, temperature=temperature, top_k=top_k, top_p=top_p)
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts
