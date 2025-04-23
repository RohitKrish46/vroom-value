from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Abstract Base Class for Univariate Analysis
# -------------------------------------------
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform a specific type of univariate analysis on a given column.

        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        feature (str): The name of the column to be analyzed.

        Returns:
        None: This method should print the results of the analysis.
        """
        pass

# Strategy for Numerical Features
# --------------------------------
# This strategy analyzes numerical features by plotting their distribution
class NumericalFeatureAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE.

        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        feature (str): The name of the numerical column to be analyzed.

        Returns:
        None: Displays a histogram and KDE plot of the numerical feature.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f"Distribution of {feature}", weight="bold", fontsize=20)
        plt.xlabel(feature)
        if feature == "selling_price":
            plt.xlim(0, 3000000)
        plt.ylabel("Frequency")        
        plt.show()

# Strategy for Categorical Features
# --------------------------------
# This strategy analyzes categorical features by plotting their frequencydistribution
class CategoricalFeatureAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        feature (str): The name of the categorical column to be analyzed.

        Returns:
        None: Displays a barplot of the categorical feature.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.show()


# Context class that helps you choose a Univariate Analysis Strategy
# ------------------------------------------------------
# This class allows you to switch between different univariate analysis strategies.
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self.strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self.strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The DataFrame to be analyzed.
        feature (str): The name of the column to be analyzed.

        Returns:
        None: Executes the strategy's analyze method.
        """
        self.strategy.analyze(df, feature)

if __name__ == "__main__":
    pass