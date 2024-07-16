import random
import numpy as np
from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2LMHeadModel
import platform

try:
    from openvino.inference_engine import IECore
    openvino_available = True
except ImportError:
    openvino_available = False

class LocalLLM:
    def __init__(self, model_name, use_amd_optimization=False, use_intel_optimization=False, seed=None):
        self.use_intel_optimization = use_intel_optimization and openvino_available
        self.model_name = model_name
        if seed is None:
            seed = random.randint(0, 10000)  # Use a random seed if none is provided
        set_seed(seed)
        self.seed = seed

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        if self.use_intel_optimization:
            self.generator = self._load_openvino_model(model_name)
        else:
            self.generator = pipeline('text-generation', model=model_name)

    def _load_openvino_model(self, model_name):
        ie = IECore()
        model_xml = f"{model_name}.xml"
        model_bin = f"{model_name}.bin"
        net = ie.read_network(model=model_xml, weights=model_bin)
        exec_net = ie.load_network(network=net, device_name="CPU")
        return exec_net

    def generate_text(self, prompt, max_length, num_return_sequences, temperature=0.7, top_k=50, top_p=0.9):
        if self.use_intel_optimization:
            # Perform inference using OpenVINO
            input_data = self._prepare_input(prompt)
            output = self.generator.infer(inputs=input_data)
            return self._process_output(output)
        else:
            return self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

    def _prepare_input(self, prompt):
        # Convert the input prompt into the format required by the OpenVINO model
        input_ids = self.tokenizer.encode(prompt, return_tensors='np')
        return {'input_ids': input_ids}

    def _process_output(self, output):
        # Convert the output of the OpenVINO model back into text
        output_ids = output['logits']
        text = self.tokenizer.decode(np.argmax(output_ids, axis=-1)[0])
        return [text]
