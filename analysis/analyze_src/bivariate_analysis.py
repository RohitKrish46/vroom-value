from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Abstract Base Class for Univariate Analysis
# -------------------------------------------
# Subclasses must implement the analyze method.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform a specific type of bivariate analysis on a given column.

        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass

# Strategy for Numerical Features
# --------------------------------
# This strategy analyzes numerical features by plotting their distribution
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using a scatter plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature/column to be analyzed.
        feature2 (str): The name of the second numerical feature/column to be analyzed.

        Returns:
        None: Displays a scatter plot showing the relationship between the two features.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Strategy for Categorical Features
# --------------------------------
# This strategy analyzes categorical features by plotting their frequencydistribution
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature using a box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature/column to be analyzed.
        feature2 (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a box plot showing the relationship between the categorical and numerical features.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


# Context class that helps you choose a Univariate Analysis Strategy
# ------------------------------------------------------
# This class allows you to switch between different univariate analysis strategies.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self.strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self.strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The DataFrame to be analyzed.
        feature (str): The name of the column to be analyzed.

        Returns:
        None: Executes the strategy's analyze method.
        """
        self.strategy.analyze(df, feature1, feature2)

if __name__ == "__main__":
    pass