import argparse
import logging
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from prompt_engine import LLMInterface
import optuna
from tabulate import tabulate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = load_iris()
X, y = data.data, data.target

# Initialize LLMInterface
llm_interface = LLMInterface(model_name='gpt2')

def prompt_llm(task_description, max_length=150):
    response = llm_interface.generate_text(task_description, max_length=max_length, num_return_sequences=1)
    return response[0]['generated_text']

def perform_grid_search(classifier, param_grid):
    grid_search = GridSearchCV(classifier(), param_grid, cv=5)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    return best_params, best_score

def perform_bayesian_optimization():
    def objective(trial):
        C = trial.suggest_loguniform('C', 1e-4, 1e2)
        gamma = trial.suggest_loguniform('gamma', 1e-4, 1e2)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        model = SVC(C=C, gamma=gamma, kernel=kernel)
        score = cross_val_score(model, X, y, cv=5, n_jobs=-1)
        return score.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    best_value = study.best_value
    return best_params, best_value

def main(output_summary):
    try:
        task_description = "Generate Python code for hyperparameter tuning using Grid Search and Bayesian Optimization for an SVM model."
        generated_code = prompt_llm(task_description, max_length=300)
        logger.info("Generated Code by LLM:\n%s", generated_code)

        # Execute the generated code (for demonstration purposes, using predefined functions)
        # In a real scenario, you'd evaluate and possibly modify the generated code before execution
        exec(generated_code)

        best_params_grid, best_score_grid = perform_grid_search(SVC, {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': [0.001, 0.01, 0.1, 1]
        })
        best_params_bo, best_value_bo = perform_bayesian_optimization()

        if output_summary:
            summary_table = [
                ["Method", "Best Parameters", "Best Cross-Validation Score"],
                ["Grid Search", best_params_grid, f"{best_score_grid:.2f}"],
                ["Bayesian Optimization", best_params_bo, f"{best_value_bo:.2f}"]
            ]

            comparison = (
                "Bayesian Optimization found better hyperparameters"
                if best_value_bo > best_score_grid else
                "Grid Search found better hyperparameters"
                if best_value_bo < best_score_grid else
                "Both methods found equally good hyperparameters"
            )

            logger.info("\n=== Hyperparameter Tuning Summary ===")
            logger.info("\nModel Used: SVM\n")
            logger.info(tabulate(summary_table, headers="firstrow", tablefmt="pretty"))
            logger.info(f"\n=== Comparison of Tuning Methods ===\n{comparison} with a score of {max(best_score_grid, best_value_bo):.2f}")

    except Exception as e:
        logger.error("An error occurred: %s", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning using Grid Search and Bayesian Optimization")
    parser.add_argument('--no-summary', action='store_false', help="Disable output summary", dest='output_summary')
    args = parser.parse_args()
    main(args.output_summary)
