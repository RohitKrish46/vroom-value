from typing import List
import mlflow
from zenml import step
from zenml.client import Client

@step
def select_top_models_step(experiment_name: str, top_k: int = 3) -> List[str]:
    """
    A ZenML step to select top-k models based on a metric from an MLflow experiment.

    Args:
        experiment_name: The name of the MLflow experiment to query.
        top_k: Number of top models to select.

    Returns:
        A list of model names (or any identifier) for the top-k models.
    """
    # Connect to ZenML stack
    client = Client()
    experiment_tracker = client.active_stack.experiment_tracker

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(experiment_tracker.get_tracking_uri())

    # Set the experiment
    mlflow.set_experiment(experiment_name)

    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    # Get all runs for the experiment
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Sort and pick top-k runs based on a metric, e.g., 'r2'
    top_runs = runs_df.sort_values(by="metrics.r2", ascending=False).head(top_k)

    # Print summary
    print(f"Top {top_k} Models:\n")
    print(top_runs[["run_id", "metrics.mse", "metrics.r2", "params.model"]])

    # Return model names (or any other info you prefer)
    top_run_ids = top_runs["run_id"].tolist()

    return top_run_ids
