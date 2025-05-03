import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Here, we simulate importing or generating some data.
    # In a real-world scenario, this could be an API call, database query, or loading from a file.
    data = {
        "car_name": ["Hyundai i20", "Renault Duster"],
        "vehicle_age": [11, 5],
        "km_driven": [60000, 50000],
        "seller_type": ["Individual", "Individual"],
        "fuel_type": ["Petrol", "Diesel"],
        "transmission_type": ["Manual", "Manual"],
        "mileage": [17.0, 19.64],
        "engine": [1197, 1461],
        "max_power": [80.0, 108.45],
        "seats": [5, 5],
    }

    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data
