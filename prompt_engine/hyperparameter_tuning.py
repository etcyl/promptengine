from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Load data
data = load_iris()
X, y = data.data, data.target

# Define the objective function for Bayesian Optimization
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e2)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    model = SVC(C=C, gamma=gamma, kernel=kernel)
    score = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    return score.mean()
