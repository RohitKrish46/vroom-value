from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.data_loader_step import load_data
from zenml import pipeline

@pipeline(
    enable_cache=False,
    name="prices_predictor",
)
def ml_pipeline(model_name, data_engineering_run_id) -> None:
    """Pipeline for training any given model."""
    
    # Step 1: load data from the artifact store
    X_train, X_test, y_train, y_test = load_data(data_engineering_run_id)

    # Step 1: Model building step
    trained_model = model_building_step(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name  # Pass different models
    )

    # Step 2: Model evaluation step
    r2, mse = model_evaluator_step(
        trained_model=trained_model,
        X_test=X_test,
        y_test=y_test
    )

    return trained_model
if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()