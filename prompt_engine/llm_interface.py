# llm_interface.py
from prompt_engine.local_llm import LocalLLM
from prompt_engine.remote_llm import RemoteLLM
from prompt_engine.openai_llm import OpenAILLM
import platform
import optuna
from sklearn.model_selection import GridSearchCV

class LLMInterface:
    def __init__(self, model_name='gpt2', remote=False, api_url=None, use_openai=False, openai_api_key=None, use_amd_optimization=False, use_intel_optimization=False, use_jax=False, seed=None):
        self.remote = remote
        self.use_amd_optimization = use_amd_optimization and self._detect_amd_hardware()
        self.use_intel_optimization = use_intel_optimization and self._detect_intel_hardware()
        self.use_jax = use_jax

        if use_openai and openai_api_key:
            self.handler = OpenAILLM(openai_api_key, use_jax=self.use_jax)
        elif self.remote and api_url:
            self.handler = RemoteLLM(api_url, use_jax=self.use_jax)
        else:
            self.handler = LocalLLM(model_name, use_amd_optimization=self.use_amd_optimization, use_intel_optimization=self.use_intel_optimization, use_jax=self.use_jax, seed=seed)

    def generate_text(self, prompt, max_length=50, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9):
        if isinstance(self.handler, OpenAILLM):
            return self.handler.generate_text(prompt, max_length, num_return_sequences, temperature)
        return self.handler.generate_text(prompt, max_length, num_return_sequences, temperature, top_k, top_p)

    def grid_search(self, model, param_grid, X, y):
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X, y)
        return grid_search.best_params_, grid_search.best_score_

    def bayesian_optimization(self, objective, n_trials=50):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params, study.best_value

    def train_model(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        score = model.score(X_test, y_test)
        return score

    def _detect_amd_hardware(self):
        """ Detects if the system uses an AMD processor with AI capabilities. """
        return 'AMD' in platform.processor()

    def _detect_intel_hardware(self):
        """ Detects if the system uses an Intel processor with AI capabilities. """
        return 'Intel' in platform.processor()
