from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from zenml import pipeline

@pipeline(
    enable_cache=False,
    name="prices_predictor",
)
def model_selection_pipeline(model_name) -> None:
    """Train and evaluate multiple models to select the best one."""
    
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

    # Step 6: Model building step
    trained_model = model_building_step(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name  # Pass different models
    )

    # Step 7: Model evaluation step
    r2, mse = model_evaluator_step(
        trained_model=trained_model,
        X_test=X_test,
        y_test=y_test
    )

    return trained_model

if __name__ == "__main__":
    run = model_selection_pipeline()
