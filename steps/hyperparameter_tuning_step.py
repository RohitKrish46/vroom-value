from zenml import step
import mlflow
import pandas as pd
from utils.hyperparameter_loader import load_hyperparameters
from sklearn.model_selection import GridSearchCV
from typing import List

@step
def tune_models_step(top_run_ids: List[str], dataset: tuple) -> str:
    """Hyperparameter tune the top 3 models dynamically based on logged model name."""
    X_train, X_test, y_train, y_test = dataset
    hyperparams_config = load_hyperparameters()

    best_run_id = None
    best_score = -float('inf')

    for run_id in top_run_ids:
        run = mlflow.get_run(run_id)
        model_name = run.data.params.get("model_name")

        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        param_grid = hyperparams_config.get(model_name, {})
        
        if not param_grid:
            print(f"No hyperparameters defined for {model_name}, skipping tuning.")
            continue
        
        grid = GridSearchCV(model, param_grid, scoring="r2", cv=3)
        grid.fit(X_train, y_train)
        score = grid.best_score_

        if score > best_score:
            best_score = score
            best_run_id = run_id
            best_estimator = grid.best_estimator_
            best_params = grid.best_params_

    # Log best model
    with mlflow.start_run(run_name="best_model_hyper_tuning") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("best_r2", best_score)
        mlflow.sklearn.log_model(best_estimator, "model")

    return best_run_id
