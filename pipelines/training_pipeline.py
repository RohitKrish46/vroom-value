from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from zenml import Model, pipeline, step

@pipeline()
def ml_pipeline():
    """Defining an end to end machine learning pipeline."""

    # data ingestion step
    raw_data = data_ingestion_step(
        file_path = "D:/Repositories/vroom-value-with-pipelines/data/Archive.zip"
        )

    # handle missing values step
    cleaned_data = handle_missing_values_step(raw_data)

    pass
