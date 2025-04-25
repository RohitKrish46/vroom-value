import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

# setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# abstract base class for data splitting strategy
# -----------------------------------------------
# this class defines a common interface for data splitting strategies.
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> tuple:
        """
        Abstract method to split the data into train and test sets.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        tuple: A tuple containing the train and test sets.
        """
        pass

# concrete strategy for simple train-test splitting
# --------------------------------------
# this strategy splits the data into train and test sets.
class SimpleSplittingStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=21):
        """
        Initializes the SimpleSplittingStrategy with specific parameters.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str) -> tuple:
        """
        Splits the data into train and test sets using the specified parameters.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        tuple: A tuple containing the train and test sets.
        """
        logging.info("Performing simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logging.info("Simple train-test split completed.")
        return X_train, X_test, y_train, y_test
    
# context class for data splitting
# ---------------------------------
# this class allows you to switch between different data splitting strategies.
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific strategy.

        Parameters:
        strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        Parameters:
        strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logging.info("Switching data splitting strategy.")
        self.strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str) -> tuple:
        """
        Executes the data splitting using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        tuple: A tuple containing the train and test sets.
        """
        logging.info("Splitting data using the selected strategy.")
        return self.strategy.split_data(df, target_column)

