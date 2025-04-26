import logging
from typing import Annotated
from category_encoders.binary import BinaryEncoder
import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    binary_cols = ["car_name"]
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in binary_cols]
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    logging.info(f"Binary columns: {binary_cols}")
    logging.info(f"Categorical columns: {categorical_cols}")
    logging.info(f"Numerical columns: {numerical_cols}")

    # BinaryEncoder for binary cols (car_name), no imputer
    binary_transformer = BinaryEncoder()

    # OneHotEncoder for other categorical cols, no imputer
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # ColumnTransformer (No imputation used)
    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", binary_transformer, binary_cols),
            ("num", "passthrough", numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        logging.info("Building and training the Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # To get encoded column names
        binary_encoder = pipeline.named_steps["preprocessor"].named_transformers_["binary"]
        cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]

        binary_encoder.fit(X_train[binary_cols])
        cat_encoder.fit(X_train[categorical_cols])

        binary_feature_names = binary_encoder.get_feature_names_out(binary_cols)
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
        expected_columns = numerical_cols + list(binary_feature_names) + list(cat_feature_names)

        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e
    finally:
        mlflow.end_run()

    return pipeline
