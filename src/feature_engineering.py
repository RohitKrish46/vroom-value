import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.binary import BinaryEncoder

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# abstract base class for feature engineering strategy
# ----------------------------------------------------
# This class defines a common interface to apply different feature engineering strategies.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply a feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with feature engineering applied.
        """
        pass

# concrete strategy for log transformation
# ----------------------------------------
# this startegy applies log transformation to skewed feature to normalize distribution
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation strategy with specific features.

        Parameters:
        features (list): The list of features to apply log transformation.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log trandformation to the specified features in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with log transformation applied.
        """
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("Log transformation completed.")
        return df_transformed 
    
# concrete strategy for robust scaling
# ------------------------------------
# this strategy applies robust scaling to features to handle outliers
class RobustScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the RobustScaling strategy with specific features.

        Parameters:
        features (list): The list of features to apply robust scaling.
        """
        self.features = features
        self.scaler = RobustScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Applies robust scaling to the specified features in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with robust scaling applied.
        """
        logging.info(f"Applying robust scaling to featrues: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Robust scaling completed.")
        return df_transformed
    

# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed

# concrete strategy for one-hot encoding
# -------------------------------------
# this strategy applies one-hot encoding to categorical features to convert them into a format that can be provided to ML algorithms
class OneHotEncoding(FeatureEngineeringStrategy):
    def _init__(self, features):
        """
        Initializes the OneHotEncoding strategy with specific features.

        Parameters:
        features (list): The list of features to apply one-hot encoding.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# concrete strategy for binary encoding
# -------------------------------------
# this strategy applies binary encoding to categorical features to convert them into a format that can be provided to ML algorithms
class BinaryEncoding(FeatureEngineeringStrategy):
    def _init__(self, features):
        """
        Initializes the binary encoding strategy with specific features.

        Parameters:
        features (list): The list of features to apply binary encoding.
        """
        self.features = features
        self.encoder = BinaryEncoder(sparse=False)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies binary encoding to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with binary encoded features.
        """
        logging.info(f"Applying binary encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("Binary encoding completed.")
        return df_transformed
    
# context class for feature engineering
# -------------------------------------
# this class allows you to switch between different feature engineering strategies
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self.strategy = strategy
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with transformed features.
        """
        logging.info("Applying feature engineering strategy.")
        return self.strategy.apply_transformation(df)
