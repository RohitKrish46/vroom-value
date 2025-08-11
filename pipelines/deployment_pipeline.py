import os
from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer_step import dynamic_importer
from steps.prediction_service_loader_step import prediction_service_loader
from steps.predictor_step import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

@pipeline
def continuous_deployment_pipeline(model_name: str):
    """Run a training job and deploy an MLflow model"""
    # run the training pipeline
    trained_model = ml_pipeline(model_name=model_name)
    
    # deploy the trained model
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)

@pipeline(enable_cache=False)
def inference_pipeline():
    """Run a batch inference job with data loaded from and API."""
    # load batch data for inference
    batch_data = dynamic_importer()

    # load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    # run the predictions on the batch data
    predictor(service=model_deployment_service, input_data=batch_data) 
