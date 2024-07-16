import pandas as pd
from sklearn.model_selection import train_test_split
from prompt_engine import LLMInterface
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = load_iris()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LLMInterface
llm_interface = LLMInterface(model_name='gpt2')

def objective(trial):
    rf_n_estimators = trial.suggest_int('rf_n_estimators', 10, 200)
    rf_max_depth = trial.suggest_int('rf_max_depth', 1, 20)
    gb_n_estimators = trial.suggest_int('gb_n_estimators', 10, 200)
    gb_learning_rate = trial.suggest_loguniform('gb_learning_rate', 0.01, 0.1)
    
    rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    
    score = cross_val_score(ensemble, X_train, y_train, cv=5, n_jobs=-1).mean()
    return score

# Run Bayesian Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Summarize results
best_params = study.best_params
best_value = study.best_value

logger.info(f"Best hyperparameters: {best_params}")
logger.info(f"Best cross-validation score: {best_value:.2f}")
