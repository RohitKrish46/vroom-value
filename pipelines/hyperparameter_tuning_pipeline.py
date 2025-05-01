from zenml import pipeline
from typing import List, Tuple
from steps.model_selection_step import select_top_models_step
from steps.hyperparameter_tuning_step import tune_models_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.retrain_best_model_step import retrain_best_model_step
from steps.data_ingestion_step import data_ingestion_step   # Assuming you already have it

@pipeline(
    enable_cache=False,
    name="prices_predictor")
def hyperparameter_tuning_pipeline(top_run_ids: List[str]) -> Tuple[str, dict]:
    """Pipeline to select top models and tune hyperparameters."""

    # Step 1: Data ingestion
    raw_data = data_ingestion_step(
        file_path="D:/Repositories/vroom-value-with-pipelines/data/Archive.zip"
    )

    # Step 2: Handle missing values
    cleaned_data = handle_missing_values_step(raw_data)

    # Step 3: Feature engineering
    engineered_data = feature_engineering_step(
        cleaned_data, strategy="log", features=["selling_price", "km_driven"]
    )

    # Step 4: Outlier detection
    clean_data = outlier_detection_step(engineered_data)

    # Step 5: Data splitting
    X_train, X_test, y_train, y_test = data_splitter_step(
        clean_data, target_column="selling_price"
    )
    # Step 6: Hyperparameter tuning
    best_run_id, best_params = tune_models_step(
        top_run_ids=top_run_ids, 
        X_train=X_train, 
        X_test=X_test, 
        y_train=y_train, 
        y_test=y_test
        )
    # Step 7: Retrain the best model
    retrain_best_model_step(
        best_run_id=best_run_id, 
        best_params=best_params, 
        X_train=X_train, 
        y_train=y_train
    )
    
    return best_run_id, best_params