import click
from pipelines.training_pipeline import ml_pipeline
from pipelines.model_selection_pipeline import model_selection_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import warnings
warnings.filterwarnings("ignore")

@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Run model training pipeline
    # Define models to train
    models = ["random_forest", "linear_regression", "svr", "lasso", 
              "ridge", "adaboost", "gradient_boosting", "knn", "decision_tree"]
    for model in models:
        ml_pipeline(model)

    # model selection pipeline
    # this will select the top models based on the logged metrics
    top_run_ids = model_selection_pipeline()

    

    # 

    # Retrieve the trained model from the pipeline run
    # You can uncomment and customize the following lines if you want to retrieve and inspect the trained model:
    # trained_model = run["model_building_step"]  # Replace with actual step name if different
    # print(f"Trained Model Type: {type(trained_model)}")

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )


if __name__ == "__main__":
    main()



