from prompt_engine import LLMInterface, objective
import logging
import argparse

# Setup logging to reduce noise
logging.basicConfig(level=logging.INFO)

def main(seed=None):
    # Example: Initialize the interface with Intel optimization
    try:
        engine = LLMInterface(model_name='gpt2', use_intel_optimization=True, seed=seed)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Check if Intel optimization is enabled
    if engine.use_intel_optimization:
        print("Intel optimization is enabled using OpenVINO.")
    else:
        print("Intel optimization is not enabled. Proceeding without Intel-specific optimizations.")

    # Example prompt
    prompt = "Discuss the impact of AI on modern computing."
    response = engine.generate_text(prompt, max_length=100, num_return_sequences=1, temperature=0.9, top_k=50, top_p=0.95)
    print("\nGenerated Text:")
    for i, text in enumerate(response):
        print(f"Output {i+1}: {text}")

    # Perform hyperparameter tuning
    #best_params, best_score = engine.bayesian_optimization(objective)
    #print("\nBest Hyperparameters and Score from Bayesian Optimization:")
    #print(f"Best Parameters: {best_params}")
    #print(f"Best Score: {best_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Intel optimization example with optional seed.")
    parser.add_argument('--seed', type=int, help='Random seed for text generation')
    args = parser.parse_args()

    main(seed=args.seed)
