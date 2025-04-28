import logging
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import step
from zenml.client import Client

from src.model_evaluator import ModelEvaluator, RegressionModelEvaluation

# Set up experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_evaluator_step(
    trained_model: Pipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[float, float]:
    """
    Performs model evaluation using ModelEvaluator and the specified strategy.
    
    Parameters:
        trained_model (Pipeline): The trained model.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The true labels.

    Returns:
        Tuple[float, float]: Returns the MSE score and R2 score.
    """

    # Validate input types
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise ValueError("y_test must be a pandas Series.")

    logging.info("Applying same preprocessing steps to testing data as training data.")

    # Apply preprocessing to test data
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    # Initialize evaluator with regression strategy
    evaluator = ModelEvaluator(RegressionModelEvaluation())

    # Evaluate the model
    metrics = evaluator.evaluate_model(
        trained_model.named_steps["model"],
        X_test_processed,
        y_test
    )

    # Validate metrics output
    if not isinstance(metrics, Tuple):
        raise ValueError("Model evaluation metrics must be a tuple.")

    # Log metrics to MLflow
    mse = metrics[0]
    r2 = metrics[1]
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    return (mse, r2)
