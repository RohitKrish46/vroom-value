from zenml import step
from zenml.client import Client
import pandas as pd
from typing import Tuple

# Define a step to load the data from the artifact store
@step
def load_data(data_engineering_run_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    client = Client()
    pipeline_run = client.get_pipeline_run(data_engineering_run_id)
    data_split_step_outputs = pipeline_run.steps["data_splitter_step"].outputs
    
    # Load the artifacts (adjust output names as per your data_splitter_step)
    X_train = data_split_step_outputs["output_0"][0].load()
    X_test = data_split_step_outputs["output_1"][0].load()
    y_train = data_split_step_outputs["output_2"][0].load()
    y_test = data_split_step_outputs["output_3"][0].load()

    return X_train, X_test, y_train, y_test