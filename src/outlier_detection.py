import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# abstract base class for outlier detection strategy
#---------------------------------------------------
# this class defines a common interface for outlier detection strategies.

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to detect outliers in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with detected outliers.
        """
        pass

# concrete strategy for z-score outlier detection
# ----------------------------------------------
# this strategy detects outliers using the z-score method.

class ZScoreDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers in the DataFrame using the z-score method.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with detected outliers.
        """
        logging.info(f"Detecting outliers using z-score method with threshold: {self.threshold}")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info("Outlier detection completed.")
        return outliers
    
# concrete strategy for IQR outlier detection
# ------------------------------------------
# this strategy detects outliers using the interquartile range (IQR) method.

class IQROutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=1.5):
        self.threshold = threshold
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers in the DataFrame using the IQR method.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with detected outliers.
        """
        logging.info(f"Detecting outliers using IQR method with threshold: {self.threshold}")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - self.threshold * IQR)) | (df > (Q3 + self.threshold * IQR))
        logging.info("Outlier detection completed.")
        return outliers
    
# context class for outlier detection
# -----------------------------------
# this class allows you to switch between different outlier detection strategies.

class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        """
        Sets a new strategy for the OutlierDetector.

        Parameters:
        strategy (OutlierDetectionStrategy): The new strategy to be used for outlier detection.
        """
        logging.info("Switching outlier detection strategy.")
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the outlier detection using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with detected outliers.
        """
        logging.info("Executing outlier detection strategy.")
        return self.strategy.detect_outliers(df)
    def handle_outliers(self, df: pd.DataFrame, method: str = "cap", **kwargs) -> pd.DataFrame:
        """
        Handles outliers in the DataFrame using the specified method.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        method (str): The method to handle outliers ('cap' or 'remove').

        Returns:
        pd.DataFrame: The DataFrame with outliers handled.
        """
        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.warning(f"Unknown method '{method}'. No outliers handled.")
            df_cleaned = df.copy()
        logging.info("Outlier handling completed.")
        return df_cleaned
    
    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")
    

# Example usage
if __name__ == "__main__":
    # Example dataframe
    df = pd.read_csv("D:/Repositories/vroom-value-with-pipelines/data/Archive.zip", index_col=[0])
    df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
    outlier_detector = OutlierDetector(IQROutlierDetection())

    # Detect and handle outliers
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
    print(outliers.shape)
    print(df_cleaned.shape)
    # Visualize outliers in specific features
    #outlier_detector.visualize_outliers(df_cleaned, features=["selling_price", "km_driven"])
    pass
