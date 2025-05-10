import click
import mlflow
import logging
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from pipelines.data_engineering_pipeline import data_engineering_pipeline
from pipelines.training_pipeline import ml_pipeline
from pipelines.model_selection_pipeline import model_selection_pipeline
from pipelines.hyperparameter_tuning_pipeline import hyperparameter_tuning_pipeline
from utils.deploy import serve_model
from utils.model_loader import load_model_names
from utils.dataset_loader import load_data_path
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_data_engineering_pipeline(data_path: str = None):
    """Run the data engineering pipeline and return its run ID."""
    logger.info("Running Data Engineering pipeline...")
    try:
        data_engineering_run = data_engineering_pipeline(data_path)
        logger.info("Data Engineering pipeline completed.")
        return str(data_engineering_run.id)
    except Exception as e:
        logger.error(f"Error running data engineering pipeline: {e}")
        raise

def run_training_pipelines(models, data_engineering_run_id):
    """Run the training pipeline for each model using the data engineering run ID."""
    for model in models:
        logger.info(f"Running training pipeline for model: {model}")
        try:
            ml_pipeline(model, data_engineering_run_id)
            logger.info(f"Training pipeline for model {model} completed.")
        except Exception as e:
            logger.error(f"Error running training pipeline for model {model}: {e}")
            raise

def run_model_selection_pipeline():
    """Run the model selection pipeline and return the top run IDs."""
    logger.info("Running Model Selection pipeline...")
    try:
        model_selection_run = model_selection_pipeline()
        client = Client()
        pipeline_run = client.get_pipeline_run(model_selection_run.id)
        top_run_ids = pipeline_run.steps["select_top_models_step"].output.load()
        logger.info("Model selection pipeline completed.")
        return top_run_ids
    except Exception as e:
        logger.error(f"Error running model selection pipeline: {e}")
        raise

def run_hyperparameter_tuning_pipeline(top_run_ids, data_engineering_run_id):
    """Run the hyperparameter tuning pipeline and return the outputs."""
    logger.info("Running Hyperparameter Tuning pipeline...")
    try:
        hyper_parameter_tuning_run = hyperparameter_tuning_pipeline(top_run_ids, data_engineering_run_id)
        client = Client()
        pipeline_run = client.get_pipeline_run(hyper_parameter_tuning_run.id)
        model_tune_step_outputs = pipeline_run.steps["tune_models_step"].outputs
        retrained_model_id = pipeline_run.steps["retrain_best_model_step"].outputs
        logger.info("Hyperparameter Tuning pipeline completed.")
        return model_tune_step_outputs, retrained_model_id
    except Exception as e:
        logger.error(f"Error running hyperparameter tuning pipeline: {e}")
        raise

def retrieve_and_print_results(model_tune_step_outputs, retrained_model_id):
    """Retrieve and print the results of the hyperparameter tuning pipeline."""
    retrained_model_run_id = retrained_model_id["retrained_model_run_id"][0].load()
    best_run_id = model_tune_step_outputs["best_run_id"][0].load()
    best_params = model_tune_step_outputs["best_params"][0].load()
    
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.get_run(best_run_id)
    best_model_uri = f"runs:/{best_run_id}/model"
    original_model = mlflow.sklearn.load_model(best_model_uri)
    model_name = original_model.named_steps['model'].__class__.__name__
    
    print(f"Best Run ID: {best_run_id}")
    print(f"Best Model: {model_name}")
    print(f"Best Parameters: {best_params}")
    print(f"Retrained Model Run ID: {retrained_model_run_id}")
    
    return retrained_model_run_id

def serve_retrained_model(retrained_model_run_id):
    """Serve the retrained model using the provided run ID."""
    ml_runs_path = get_tracking_uri()
    run = mlflow.get_run(retrained_model_run_id)
    experiment_id = run.info.experiment_id
    retrained_model_path = f"{ml_runs_path}/{experiment_id}/{retrained_model_run_id}/artifacts/model"
    serve_model(retrained_model_path, port=5001)

@click.command()
def main():
    """
    Run the ML pipeline, retrieve top run IDs, and start the MLflow UI for experiment tracking.
    """
    warnings.filterwarnings("ignore")
    
    try:
        # Run all pipelines first
        data_path = load_data_path()
        data_engineering_run_id = run_data_engineering_pipeline(data_path)
        
        models = load_model_names()
        run_training_pipelines(models, data_engineering_run_id)
        
        top_run_ids = run_model_selection_pipeline()
        print(f"Top Run IDs: {top_run_ids}")
        
        model_tune_step_outputs, retrained_model_id = run_hyperparameter_tuning_pipeline(top_run_ids, data_engineering_run_id)
        
        retrained_model_run_id = retrieve_and_print_results(model_tune_step_outputs, retrained_model_id)
        
        # Only after all pipelines have completed, serve the model
        logger.info("All pipelines completed successfully. Starting model serving...")
        serve_retrained_model(retrained_model_run_id)
    
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()