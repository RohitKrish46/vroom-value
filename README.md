# Used Car Price Predictor

A comprehensive machine learning pipeline for predicting used car prices. Built with ZenML and MLflow, this project implements end-to-end ML pipelines for data processing, model training, and evaluation.

## 🚗 Overview

This project predicts used car prices using various regression models. It features a modular pipeline architecture that handles everything from data ingestion to model deployment, making it easy to train, evaluate, and deploy price prediction models.

## 🌟 Key Features

### Data Pipeline
- Automated data ingestion from ZIP/CSV files
- Handles missing values with configurable strategies
- Advanced feature engineering:
  - Log transformation for price and mileage
  - Categorical encoding for car names and features
  - Outlier detection and handling
  - Automated data cleaning

### Model Pipeline
- Supports multiple regression models:
  - Random Forest
  - Linear Regression
  - Support Vector Regression
  - Gradient Boosting
  - And more...
- Automated model selection
- Cross-validation and hyperparameter tuning
- Comprehensive model evaluation metrics

### MLOps Features
- Experiment tracking with MLflow
- Model versioning
- Pipeline caching for faster iterations
- Reproducible training workflows

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/used-car-price-predictor.git
cd used-car-price-predictor
```

2. Create and activate a Python 3.10 environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## 📊 Data Format

The model expects car data in CSV format with the following features:
- `car_name`: Name/model of the car
- `selling_price`: Target variable (price in currency units)
- `km_driven`: Mileage of the car
- Additional features as per your dataset

## 🚀 Quick Start

### Training a Model

1. Run the model selection pipeline to evaluate multiple models:
```bash
python run_pipeline.py
```

2. View training results in MLflow UI:
```bash
mlflow ui --backend-store-uri '{tracking_uri}'
```

### Using the Model

```python
from pipelines.training_pipeline import ml_pipeline

# Train the model
model = ml_pipeline()

# Make predictions
predictions = model.predict(new_data)
```

## 📈 Model Performance

Our best performing model achieves:
- R² Score: [Your Score]
- Mean Squared Error: [Your Score]
- Mean Absolute Error: [Your Score]

## 🔄 Pipeline Architecture

```
Data Ingestion → Missing Value Handling → Feature Engineering → Outlier Detection → Model Training → Evaluation
```

### Pipeline Components:
1. **Data Ingestion**: Loads and validates raw car data
2. **Missing Value Handler**: Implements multiple strategies (mean, median, mode)
3. **Feature Engineering**: Applies transformations and encoding
4. **Outlier Detection**: Uses IQR and Z-score methods
5. **Model Training**: Trains and validates multiple models
6. **Model Evaluation**: Computes comprehensive metrics

## 📁 Project Structure

```
used-car-price-predictor/
├── pipelines/
│   ├── model_selection_pipeline.py
│   └── training_pipeline.py
├── steps/
│   ├── data_ingestion_step.py
│   ├── feature_engineering_step.py
│   ├── model_building_step.py
│   └── model_evaluator_step.py
├── src/
│   ├── feature_engineering.py
│   ├── ingest_data.py
│   └── model_evaluator.py
└── README.md
```

## 🔧 Configuration

Key configurations in `pyproject.toml`:
```toml
[project]
name = "used-car-price-predictor"
version = "0.1.0"
requires-python = ">=3.10"
```

## 📈 Future Improvements

- [ ] Add more feature engineering techniques
- [ ] Implement deep learning models
- [ ] Add API endpoint for predictions
- [ ] Deploy model as a service
- [ ] Add real-time prediction capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📫 Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/used-car-price-predictor

## 📝 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.