import click
import mlflow
from pipelines.training_pipeline import ml_pipeline
from pipelines.model_selection_pipeline import model_selection_pipeline
from pipelines.hyperparameter_tuning_pipeline import hyperparameter_tuning_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.client import Client
import warnings
warnings.filterwarnings("ignore")

@click.command()
def main():
    """
    Run the ML pipeline, retrieve top run IDs, and start the MLflow UI for experiment tracking.
    """
    # Run model training pipeline
    # Define models to train
    # models = ["random_forest", "linear_regression", "svr", "lasso", 
    #           "ridge", "adaboost", "gradient_boosting", "knn", "decision_tree"]
    # for model in models:
    #     ml_pipeline(model)

    # Run model selection pipeline
    model_selection_run = model_selection_pipeline()
    
    # Retrieve the top_run_ids artifact from the model selection pipeline
    client = Client()
    pipeline_run = client.get_pipeline_run(model_selection_run.id)
    top_run_ids = pipeline_run.steps["select_top_models_step"].output.load()
    print(f"Top Run IDs: {top_run_ids}")

    # Pass top_run_ids to the hyperparameter tuning pipeline
    hyper_parameter_tuning_run = hyperparameter_tuning_pipeline(top_run_ids=top_run_ids)
    pipeline_run = client.get_pipeline_run(hyper_parameter_tuning_run.id)
    step_outputs = pipeline_run.steps["tune_models_step"].outputs
    best_run_id = step_outputs["output_0"][0].load()
    best_params = step_outputs["output_1"][0].load()
    #best_run_id, best_params = step_outputs["return_value"].load()
    mlflow.set_tracking_uri(get_tracking_uri())
    run = mlflow.get_run(best_run_id)
    best_model_uri = f"runs:/{best_run_id}/model"
    original_model = mlflow.sklearn.load_model(best_model_uri)
    model_name = original_model.named_steps['model'].__class__.__name__
    print(f"Best Run ID: {best_run_id}")
    print(f"Best Model: {model_name}")
    print(f"Best Parameters: {best_params}")

    # print(
    #     "Now run \n "
    #     f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
    #     "To inspect your experiment runs within the mlflow UI.\n"
    #     "You can find your runs tracked within the experiment."
    # )

if __name__ == "__main__":
    main()