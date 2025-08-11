from zenml import step
import mlflow
import pandas as pd
from utils.hyperparameter_loader import load_hyperparameters
from sklearn.model_selection import GridSearchCV
from typing import List, Tuple, Annotated

@step
def tune_models_step(
    top_run_ids: List[str], 
    X_train: pd.DataFrame, 
    y_train: pd.Series,) -> Tuple[
    Annotated[str, "best_run_id"],
    Annotated[dict, "best_params"],
    ]:
    """Hyperparameter tune the top 3 models dynamically based on logged model name."""
    hyperparams_config = load_hyperparameters()

    best_run_id = None
    best_score = -float('inf')
    best_estimator = None
    best_params = None

    for run_id in top_run_ids:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        model_name = model.named_steps['model'].__class__.__name__
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
            best_model = model_name

    # Check if a valid model was found
    if best_estimator is None or best_params is None:
        print("No models were successfully tuned. Returning original best run ID.")
        return top_run_ids[0] if top_run_ids else None

    # Log best model
    with mlflow.start_run(run_name=f"best_hyper_param_tuning_for_{best_model}") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("best_r2", best_score)
        mlflow.sklearn.log_model(best_estimator, "model")

    return best_run_id, best_params