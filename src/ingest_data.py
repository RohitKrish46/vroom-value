import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd

# abstract class for data ingestion
class DataIngestion(ABC):
    @abstractmethod
    def ingest_data(self):
        """ Abstract method for ingesting data """
        pass

# concrete class for ingesting data from a ZIP file
class ZipDataIngestion(DataIngestion):
    def ingest_data(self, file_path: str) -> pd.DataFrame:
        """ Extracts data from a .zip file and returns a pandas as DataFrame """
        # check if file is a zip file
        if not file_path.endswith('.zip'):
            raise ValueError("The provided file is not a .zip file.")
        
        # extract the zip file
        with zipfile.ZipFile(file_path, "r") as f:
            f.extractall("extracted_data")

        # find the extracted file
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith('.csv')]
        
        if len(csv_files) == 0:
            raise FileNotFoundError("No .csv file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("More than one .csv file found in the extracted data.")

        # read the csv file into a pandas DataFrame
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)
        return df
    
# factory class to create data ingestors
class DataIngestionFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestion:
        """ Returns appropriate data ingestor based on file type """
        if file_extension.endswith('.zip'):
            return ZipDataIngestion()
        else:
            raise ValueError(f"Unsupported file type. No ingestor available for{file_extension}")
