from steps.model_selection_step import select_top_models_step
from typing import List
from zenml import pipeline

@pipeline(
    enable_cache=False,
    name="prices_predictor",
)
def model_selection_pipeline()-> List[str]:
    top_run_ids = select_top_models_step(experiment_name="prices_predictor", top_k=3)
    return top_run_ids

if __name__ == "__main__":
    run = model_selection_pipeline()
