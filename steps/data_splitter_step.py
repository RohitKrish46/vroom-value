import pandas as pd
from typing import Annotated
from src.data_splitter import DataSplitter, SimpleSplittingStrategy
from zenml import step

@step
def data_splitter_step(
    df: pd.DataFrame, target_column: str
    ) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
    ]:
    """Splits the data into train and test sets using the DataSplitter."""
    data_splitter = DataSplitter(SimpleSplittingStrategy())
    X_train, X_test, y_train, y_test = data_splitter.split(df, target_column)
    return X_train, X_test, y_train, y_test