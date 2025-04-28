import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import RegressorMixin

# setting up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# abstract base class for model evaluation strategy
# -------------------------------------------------
# this class defines a common interface for model evaluation strategies.
class ModelEvaluationStartegy(ABC):
    @abstractmethod
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
        """
        Abstract method to evaluate the model.

        Parameters:
        model (RegressorMixin): The trained model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The true labels.

        returns:
        tuple: A tuple containing the evaluation metrics.
        """
        pass

# concrete strategy for regression model evaluation
# -----------------------------------------------
# this strategy evaluates regression models using MSE and R2 score.
class RegressionModelEvaluation(ModelEvaluationStartegy):
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float]:
        """
        Evaluates the regression model using MSE and R2 score.

        Parameters:
        model (RegressorMixin): The trained model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The true labels.

        returns:
        Tuple[float, float]: Tuple containing the MSE and R2 score.
        """
        logging.info("Predicting using the trained model")
        y_pred = model.predict(X_test)

        logging.info("Calculating RMSE and R2 score")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {"mse": mse, "r2": r2}
        logging.info(f"Model evaluation metrics: {metrics}")

        return (mse, r2)
    
# context class that helps you choose a model evaluation strategy
# --------------------------------------------------------------
# this class allows you to switch between different model evaluation strategies.
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStartegy):
        self.strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStartegy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStartegy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self.strategy = strategy

    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (RegressorMixin): The trained model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The true labels.

        returns:
        dict: A dictionary containing the evaluation metrics.
        """
        logging.info("Executing model evaluation strategy.")
        return self.strategy.evaluate_model(model, X_test, y_test)
    
