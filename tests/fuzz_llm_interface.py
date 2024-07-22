import atheris
import sys
import fuzz_helpers
import optuna
from sklearn.svm import SVC
import numpy as np

with atheris.instrument_imports():
    from llm_interface import LLMInterface

def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)

    model_name = fdp.ConsumeUnicodeNoSurrogates(10)
    remote = fdp.ConsumeBool()
    api_url = fdp.ConsumeUnicodeNoSurrogates(50)
    use_openai = fdp.ConsumeBool()
    openai_api_key = fdp.ConsumeUnicodeNoSurrogates(50)
    use_amd_optimization = fdp.ConsumeBool()
    use_intel_optimization = fdp.ConsumeBool()
    seed = fdp.ConsumeInt(32)

    try:
        llm = LLMInterface(
            model_name=model_name,
            remote=remote,
            api_url=api_url,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            use_amd_optimization=use_amd_optimization,
            use_intel_optimization=use_intel_optimization,
            seed=seed
        )
    except Exception as e:
        return

    prompt = fdp.ConsumeUnicodeNoSurrogates(100)
    max_length = fdp.ConsumeIntInRange(1, 100)
    num_return_sequences = fdp.ConsumeIntInRange(1, 5)
    temperature = fdp.ConsumeFloat()
    top_k = fdp.ConsumeIntInRange(1, 100)
    top_p = fdp.ConsumeFloat()

    try:
        llm.generate_text(prompt, max_length, num_return_sequences, temperature, top_k, top_p)
    except Exception as e:
        pass

    # Fuzzing grid_search function
    try:
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        model = SVC()
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        llm.grid_search(model, param_grid, X, y)
    except Exception as e:
        pass

    # Fuzzing bayesian_optimization function
    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return x * x

    try:
        llm.bayesian_optimization(objective, n_trials=5)
    except Exception as e:
        pass

def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()

if __name__ == "__main__":
    main()

