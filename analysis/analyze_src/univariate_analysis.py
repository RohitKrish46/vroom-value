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
    def analyze(self, df: pd.DataFrame, features: list):
        """
        Plots the distribution of a numerical feature using a histogram and KDE.

        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        feature (str): The name of the numerical column to be analyzed.

        Returns:
        None: Displays a histogram and KDE plot of the numerical feature.
        """
        plt.figure(figsize = (15, 15))
        plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1)
        for i in range(len(features)):
            plt.subplot(5, 3, i+1)
            sns.kdeplot(x=df[features[i]], shade=True, color='b')
            plt.xlabel(features[i])
            plt.tight_layout()

# Strategy for Categorical Features
# --------------------------------
# This strategy analyzes categorical features by plotting their frequencydistribution
class CategoricalFeatureAnalysisStrategy(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, features: list):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        feature (str): The name of the categorical column to be analyzed.

        Returns:
        None: Displays a barplot of the categorical feature.
        """
        plt.figure(figsize=(20, 15))
        plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
        # not visualizing categories with too many categories
        for i in range(0, len(features)):
            plt.subplot(2, 2, i+1)
            sns.countplot(x=df[features[i]])
            plt.xlabel(features[i])
            plt.xticks(rotation=45)
            plt.tight_layout()


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