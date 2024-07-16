from prompt_engine.local_llm import LocalLLM
from prompt_engine.remote_llm import RemoteLLM
from prompt_engine.openai_llm import OpenAILLM
from sklearn.model_selection import GridSearchCV
import optuna

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
    
    def grid_search(self, model, param_grid, X, y):
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X, y)
        return grid_search.best_params_, grid_search.best_score_

    def bayesian_optimization(self, objective, n_trials=50):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params, study.best_value

# Objective Functions for Hyperparameter Tuning
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Load data
data = load_iris()
X, y = data.data, data.target

# Define parameter grid for Grid Search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# Create an instance of LLMInterface for Grid Search
llm_interface = LLMInterface()
best_params, best_score = llm_interface.grid_search(SVC(), param_grid, X, y)
print("Best parameters found (Grid Search): ", best_params)
print("Best cross-validation score (Grid Search): {:.2f}".format(best_score))

# Define the objective function for Bayesian Optimization
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e2)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    model = SVC(C=C, gamma=gamma, kernel=kernel)
    score = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    return score.mean()

# Run Bayesian Optimization
best_params_bo, best_value_bo = llm_interface.bayesian_optimization(objective, n_trials=50)
print("Best parameters found (Bayesian Optimization): ", best_params_bo)
print("Best cross-validation score (Bayesian Optimization): {:.2f}".format(best_value_bo))
