import pandas as pd
from src.ingest_data import DataIngestionFactory
from zenml import step

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """ Ingests data from a .zip file and returns a pandas DataFrame."""
    # Determine the file extension
    file_extension = ".zip"  # Since we're dealing with ZIP files, this is hardcoded

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestionFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest_data(file_path)
    return df