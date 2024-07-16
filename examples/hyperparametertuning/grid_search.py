from sklearn.datasets import load_iris
from sklearn.svm import SVC
from prompt_engine import LLMInterface

# Load data
data = load_iris()
X, y = data.data, data.target

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# Initialize and run grid search
llm_interface = LLMInterface()
best_params, best_score = llm_interface.grid_search(SVC(), param_grid, X, y)
print("Best parameters found (Grid Search): ", best_params)
print("Best cross-validation score (Grid Search): {:.2f}".format(best_score))
