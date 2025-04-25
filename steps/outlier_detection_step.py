import pandas as pd
import logging
from src.outlier_detection import OutlierDetector, ZScoreDetection, IQROutlierDetection
from zenml import step

@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Performs outlier detection using OutlierDetector and the specified strategy."""
    logging.info(f"Starting outlier detection step with dataframe shape: {df.shape} and column name: {column_name}")

    if df is None:
        logging.warning("No dataframe provided for outlier detection.")
        raise ValueError("Input must be a non-null pandas dataframe.")
    
    if not isinstance(df, pd.DataFrame):
        logging.warning(f"Input is not a pandas dataframe. Got {type(df)}")
        raise ValueError("Input must be a pandas dataframe.")
    
    if column_name not in df.columns:
        logging.warning(f"Column '{column_name}' not found in the dataframe.")
        raise ValueError(f"Column '{column_name}' not found in the dataframe.")
    # ensure only numeric columns are used for outlier detection
    df_numeric = df.select_dtypes(include=[int, float])

    outlier_detector = OutlierDetector(ZScoreDetection())
    outliers = outlier_detector.detect_outliers(df_numeric)
    #df_cleaned = outlier_detector.handle_outliers(df_numeric, method="cap")
    return outliers