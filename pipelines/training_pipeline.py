from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from zenml import Model, pipeline, step

@pipeline(
        enable_cache=False,
        # The name uniquely identifies this model
        name="prices_predictor",
        )
def ml_pipeline():
    """Defining an end to end machine learning pipeline."""

    # data ingestion step
    raw_data = data_ingestion_step(
        file_path = "D:/Repositories/vroom-value-with-pipelines/data/Archive.zip"
        )

    # handle missing values step
    cleaned_data = handle_missing_values_step(raw_data)

    # feature engineering step
    engineered_data = feature_engineering_step(cleaned_data, strategy="log", features=["selling_price", "km_driven"])

    # outlier detection step
    clean_data = outlier_detection_step(engineered_data)

    # data splitting step
    X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="selling_price")

    # model building step
    model = model_building_step(X_train, y_train)

    # Model Evaluation Step
    r2, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model

if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()