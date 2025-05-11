from zenml import pipeline
from typing import List, Tuple
from steps.model_evaluator_step import model_evaluator_step
from steps.hyperparameter_tuning_step import tune_models_step
from steps.retrain_best_model_step import retrain_best_model_step
from steps.data_loader_step import load_data  

@pipeline(
    enable_cache=False,
    name="prices_predictor")
def hyperparameter_tuning_pipeline(top_run_ids: List[str], data_engineering_run_id: str) -> Tuple[str, dict]:
    """Pipeline to select top models and tune hyperparameters."""
    
    # Step 1: load data from the artifact store
    X_train, X_test, y_train, y_test = load_data(data_engineering_run_id)
    
    # Step 1: Hyperparameter tuning
    best_run_id, best_params = tune_models_step(
        top_run_ids=top_run_ids, 
        X_train=X_train, 
        y_train=y_train, 
        )
    # Step 2: Retrain the best model
    trained_model, cur_run_id = retrain_best_model_step(
        best_run_id=best_run_id, 
        best_params=best_params, 
        X_train=X_train, 
        y_train=y_train
    )
    # Step 3: Evaluate the retrained model
    r2, mse = model_evaluator_step(
        trained_model=trained_model,
        X_test=X_test,
        y_test=y_test
    )
    
    return best_run_id, best_params, cur_run_id