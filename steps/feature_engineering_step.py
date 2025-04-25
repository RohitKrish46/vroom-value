import pandas as pd
import logging
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    RobustScaling,
    StandardScaling,
    MinMaxScaling,
    OneHotEncoding,
    BinaryEncoding,
)
from zenml import step

@step
def feature_engineering_step(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and the specified strategy."""
    logging.info(f"Starting feature engineering step with dataframe shape: {df.shape} and strategy: {strategy}")
    
    # ensure features are provided
    if features is None:
        features = []
        logging.warning("No features provided for feature engineering.")
        raise ValueError("No features provided for feature engineering.")
    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "Robust_scaling":
        engineer = FeatureEngineer(RobustScaling(features))
    elif strategy == "Standard_scaling":
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "MinMax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "OneHot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    elif strategy == "Binary_encoding":
        engineer = FeatureEngineer(BinaryEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")
    
    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df 