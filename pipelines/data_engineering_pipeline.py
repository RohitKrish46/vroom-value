from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from zenml import pipeline

@pipeline(
    enable_cache=False,
    name="prices_predictor",
)
def data_engineering_pipeline(data_path: str) -> None:
    """Pipeline for data preprocessing."""
    
    # Step 1: Data ingestion
    raw_data = data_ingestion_step(
        file_path=data_path
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
    return X_train, X_test, y_train, y_test