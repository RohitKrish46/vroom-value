import logging
import warnings
warnings.filterwarnings("ignore")
from typing import Annotated, Dict, Optional
from category_encoders.binary import BinaryEncoder
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step, Model
from zenml.client import Client
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

# Define the ZenML model
model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for resold cars.",
)

def get_model(model_name: str):
    """Return a model instance based on the model name."""
    if model_name == "LinearRegression":
        return LinearRegression()
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor()
    elif model_name == "SVR":
        return SVR()
    elif model_name == "Lasso":
        return Lasso()
    elif model_name == "Ridge":
        return Ridge()
    elif model_name == "AdaBoostRegressor":
        return AdaBoostRegressor()
    elif model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressor()
    elif model_name == "KNeighborsRegressor":
        return KNeighborsRegressor()
    elif model_name == "DecisionTreeRegressor":
        return DecisionTreeRegressor()
    else:
        raise ValueError(f"Model {model_name} not supported.")

@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
    model=model
)
def retrain_best_model_step(
    best_run_id: str,
    best_params: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None
) -> tuple[Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)], 
           Annotated[str, "retrained_model_run_id"]
]:
    """
    Retrain the best model using tuned hyperparameters.
    
    Args:
        best_run_id: MLflow run ID of the original best model.
        best_params: Dictionary of tuned hyperparameters.
        X_train: Training features.
        y_train: Training target.
        X_test: Optional test features for evaluation.
        y_test: Optional test target for evaluation.
    
    Returns:
        Pipeline: The retrained scikit-learn pipeline.
    """
    # Validate input types
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")
    if not isinstance(best_params, dict):
        raise TypeError("best_params must be a dictionary.")

    # Load the original model to get its configuration
    model_uri = f"runs:/{best_run_id}/model"
    try:
        original_model = mlflow.sklearn.load_model(model_uri)
        model_name = original_model.named_steps['model'].__class__.__name__
    except Exception as e:
        logging.error(f"Failed to load model from run {best_run_id}: {e}")
        raise ValueError(f"Failed to load model from run {best_run_id}: {e}")

    # Clean and convert hyperparameters
    cleaned_params = {}
    for key, value in best_params.items():
        clean_key = key.replace('model__', '')  # Remove pipeline prefix
        try:
            # Convert to int for integer parameters
            if clean_key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'n_neighbors', 'max_iter']:
                cleaned_params[clean_key] = int(value)
            # Convert to float for float parameters
            elif clean_key in ['learning_rate', 'subsample', 'alpha', 'C', 'epsilon', 'gamma']:
                cleaned_params[clean_key] = float(value)
            else:
                cleaned_params[clean_key] = value
        except (ValueError, TypeError):
            cleaned_params[clean_key] = value  # Keep as is if conversion fails

    # Instantiate the model with tuned parameters
    try:
        model_instance = get_model(model_name)
        model_instance.set_params(**cleaned_params)
    except Exception as e:
        logging.error(f"Failed to instantiate model {model_name} with params {cleaned_params}: {e}")
        raise ValueError(f"Failed to instantiate model {model_name}: {e}")

    # Define preprocessing (same as model_building_step)
    binary_cols = ["car_name"]
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in binary_cols]
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    logging.info(f"Binary columns: {binary_cols}")
    logging.info(f"Categorical columns: {categorical_cols}")
    logging.info(f"Numerical columns: {numerical_cols}")

    # BinaryEncoder for car_name
    binary_transformer = BinaryEncoder()

    # OneHotEncoder for other categorical columns
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", binary_transformer, binary_cols),
            ("num", "passthrough", numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model_instance)])

    # Train the model with MLflow logging
    if not mlflow.active_run():
        run = mlflow.start_run(run_name=f"retrained_{model_name}_with_best_params")
        run_id = run.info.run_id
    else:
        run_id = mlflow.active_run().info.run_id

    try:
        mlflow.sklearn.autolog()
        logging.info(f"Retraining {model_name} with parameters: {cleaned_params}")
        pipeline.fit(X_train, y_train)
        logging.info("Model retraining completed.")

        # Evaluate on test set if provided
        test_r2 = None
        if X_test is not None and y_test is not None:
            test_r2 = pipeline.score(X_test, y_test)
            mlflow.log_metric("test_r2", test_r2)
            logging.info(f"Test R² score: {test_r2}")

        # Log feature names
        binary_encoder = pipeline.named_steps["preprocessor"].named_transformers_["binary"]
        cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        binary_encoder.fit(X_train[binary_cols])
        cat_encoder.fit(X_train[categorical_cols])
        binary_feature_names = binary_encoder.get_feature_names_out(binary_cols)
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
        expected_columns = numerical_cols + list(binary_feature_names) + list(cat_feature_names)
        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model retraining: {e}")
        raise e
    finally:
        mlflow.end_run()

    return pipeline, run_id