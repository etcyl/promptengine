import torch
import time
import json
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def generate_output(tokenizer, model, device, prompt, max_new_tokens=300):
    inputs = tokenizer.encode(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    attention_mask = inputs.ne(tokenizer.pad_token_id).to(device)
    outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def format_code_block(text):
    return f"<pre><code>{text}</code></pre>"

def main():
    models = [
        'gpt2',
        'EleutherAI/gpt-neo-2.7B',
        # Add more models as needed
    ]
    
    prompts = [
        (
            "Atheris is a coverage-guided Python fuzzing engine. It supports fuzzing of Python code and native extensions. "
            "Atheris uses Python code coverage information to explore all possible input branches and identify edge cases. "
            "Below is a Python function. Please generate a diverse set of fuzz tests using Atheris to test this function extensively. "
            "The fuzz tests should explore all possible input branches, identify edge cases, and focus on error handling and input validation. "
            "Make sure the tests cover various types of inputs including edge cases such as empty lists, very large lists, and lists with negative numbers.\n\n"
            "Example Atheris fuzz test:\n"
            "import atheris\n"
            "import sys\n"
            "\n"
            "@atheris.instrument_func\n"
            "def TestOneInput(data):\n"
            "    fdp = atheris.FuzzedDataProvider(data)\n"
            "    sample_numbers = [fdp.ConsumeInt(4) for _ in range(fdp.ConsumeIntInRange(1, 10))]\n"
            "    process_numbers(sample_numbers)\n"
            "\n"
            "atheris.Setup(sys.argv, TestOneInput)\n"
            "atheris.Fuzz()\n\n"
            "# Additional examples of diverse fuzz tests:\n"
            "@atheris.instrument_func\n"
            "def TestEdgeCases(data):\n"
            "    fdp = atheris.FuzzedDataProvider(data)\n"
            "    # Test with empty list\n"
            "    process_numbers([])\n"
            "    # Test with large numbers\n"
            "    process_numbers([fdp.ConsumeInt(4) for _ in range(1000)])\n"
            "    # Test with negative numbers\n"
            "    process_numbers([fdp.ConsumeInt(4) * -1 for _ in range(10)])\n"
            "    # Test with mixed positive and negative numbers\n"
            "    process_numbers([fdp.ConsumeInt(4) * (1 if fdp.ConsumeBool() else -1) for _ in range(10)])\n"
            "    # Test with zeros\n"
            "    process_numbers([0 for _ in range(fdp.ConsumeIntInRange(1, 10))])\n"
            "\n"
            "atheris.Setup(sys.argv, TestEdgeCases)\n"
            "atheris.Fuzz()\n\n"
        ),
        # Add more prompts here
        "def add(a, b): return a + b\n\n# Please generate a set of fuzz tests using Atheris for this function.\n",
        "def subtract(a, b): return a - b\n\n# Please generate a set of fuzz tests using Atheris for this function.\n",
        "def multiply(a, b): return a * b\n\n# Please generate a set of fuzz tests using Atheris for this function.\n",
        "def divide(a, b): return a / b\n\n# Please generate a set of fuzz tests using Atheris for this function, considering edge cases like division by zero.\n"
    ]
    
    results = []
    start_time = time.time()
    for model_name in models:
        tokenizer, model, device = load_model(model_name)
        for prompt in prompts:
            print(f"Generating output for model: {model_name} with prompt:\n{prompt[:60]}...")
            
            prompt_start_time = time.time()
            output = generate_output(tokenizer, model, device, prompt)
            prompt_end_time = time.time()
            
            total_prompt_time = prompt_end_time - prompt_start_time
            time_taken = f"{total_prompt_time:.2f} seconds" if total_prompt_time < 60 else f"{total_prompt_time // 60:.0f} minutes and {total_prompt_time % 60:.2f} seconds"
            
            results.append({
                "Model": model_name,
                "Prompt": prompt,
                "Output": output,
                "Time Taken": time_taken
            })
    end_time = time.time()
    
    total_time = end_time - start_time
    total_time_formatted = f"{total_time:.2f} seconds" if total_time < 60 else f"{total_time // 60:.0f} minutes and {total_time % 60:.2f} seconds"
    print(f"Total time taken for all executions: {total_time_formatted}")
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"llm_comparison_{timestamp}.json"
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    # Save results to HTML
    df = pd.DataFrame(results)
    for col in ['Prompt', 'Output']:
        df[col] = df[col].apply(format_code_block)
    html_filename = f"llm_comparison_{timestamp}.html"
    df.to_html(html_filename, index=False, escape=False)
    
    # Save detailed JSON for each prompt and model
    detailed_results = {}
    for result in results:
        prompt = result["Prompt"]
        if prompt not in detailed_results:
            detailed_results[prompt] = []
        detailed_results[prompt].append({
            "Model": result["Model"],
            "Output": result["Output"],
            "Time Taken": result["Time Taken"]
        })
    detailed_json_filename = f"llm_detailed_comparison_{timestamp}.json"
    with open(detailed_json_filename, 'w') as detailed_json_file:
        json.dump(detailed_results, detailed_json_file, indent=4)
    
    print(f"Results saved to {json_filename}, {html_filename}, and {detailed_json_filename}")

if __name__ == "__main__":
    main()
