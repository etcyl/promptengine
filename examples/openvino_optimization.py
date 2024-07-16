from prompt_engine import LLMInterface, objective
import logging
import argparse

# Setup logging to reduce noise
logging.basicConfig(level=logging.INFO)

def main(api_url=None, use_local=True, use_intel_optimization=False, seed=None):
    # Initialize the interface
    if use_local:
        engine = LLMInterface(model_name='gpt2', use_intel_optimization=use_intel_optimization, seed=seed)
        if use_intel_optimization:
            print("Intel optimization is enabled using OpenVINO.")
    else:
        if not api_url:
            raise ValueError("API URL must be provided for remote model usage.")
        engine = LLMInterface(remote=True, api_url=api_url, seed=seed)
        print("Using remote model.")

    # Example prompt
    prompt = "Discuss the impact of AI on modern computing."
    response = engine.generate_text(prompt, max_length=100, num_return_sequences=1, temperature=0.9, top_k=50, top_p=0.95)
    print("\nGenerated Text:")
    for i, text in enumerate(response):
        print(f"Output {i+1}: {text}")

    # Perform hyperparameter tuning
    best_params, best_score = engine.bayesian_optimization(objective)
    print("\nBest Hyperparameters and Score from Bayesian Optimization:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model optimization demo with optional Intel optimization and seed.")
    parser.add_argument('--api_url', type=str, help='URL of the remote model API')
    parser.add_argument('--use_local', action='store_true', help='Flag to use local model')
    parser.add_argument('--use_intel_optimization', action='store_true', help='Flag to use Intel optimization (OpenVINO)')
    parser.add_argument('--seed', type=int, help='Random seed for text generation')
    args = parser.parse_args()

    main(api_url=args.api_url, use_local=args.use_local, use_intel_optimization=args.use_intel_optimization, seed=args.seed)
