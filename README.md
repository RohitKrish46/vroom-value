# Vroom Value

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.22%2B-orange)](https://mlflow.org/)
[![ZenML](https://img.shields.io/badge/ZenML-0.81.0%2B-blueviolet)](https://zenml.io/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![CI/CD](https://img.shields.io/badge/CI%2FCD-ZenML%20-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Vroom Value is an end-to-end MLOps solution for predicting used car prices in the Indian market. This production-grade implementation combines robust machine learning pipelines with a Flask web interface, powered by ZenML and MLflow for experiment tracking and model management.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture and Workflow](#architecture-and-workflow)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Application Screenshots](#application-screenshots)
- [Folder Structure](#folder-structure)
- [Technologies Stack](#technologies-stack)
- [Future Improvements](#future-improvements)

## Overview

The project predicts the resale price of used cars in India using regression models trained on real-world data sourced from [Cardekho](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data). The solution includes automated pipelines using ZenML for data preprocessing, model training, and hyperparameter tuning with all experiments logged in MLflow. A Flask web application provides an interactive interface for users to input car details and receive price estimations instantly.

## Features

- **Accurate Price Predictions**: Predicts used car resale values using the best regression model trained on features such as mileage, engine power, kilometers driven, fuel type, and more.
  
- **Robust ML Pipelines**: Managed with ZenML, covering data engineering, model training, top model selection, and hyperparameter tuning.
  
- **Experiment Tracking**: MLflow tracks experiments, logs metrics, and manages model versions.
  
- **User-Friendly Web Interface**: A Flask app with pages for inputting car details and viewing predicted prices in Indian Rupees (₹).
  
- **Real-Time Predictions**: The best-performing model is deployed and integrated into the Flask app for seamless user interaction.

## Architecture and Workflow

![image](https://github.com/user-attachments/assets/98c7b910-ca9a-4d4d-8442-f2f13caa95cb)


This project follows a modular and production-grade machine learning lifecycle, built for scalability, reproducibility, and ease of deployment:

1. **Data Collection**

    Collected comprehensive data from Cardekho, covering key attributes relevant to the Indian used car resale market.

2. **Data Engineering Pipeline**
   
    Managed using ZenML to ensure robust preprocessing, the pipeline includes:
  
    - Automated data ingestion
    
    - Handling of missing and inconsistent values
    
    - Domain-specific feature engineering
    
    - Outlier detection and treatment
    
    - Stratified train-test split for balanced model training

3. **Model Experimentation**

    Multiple supervised regression algorithms were trained and benchmarked:
    
      - Linear Regression
      - Ridge
      - Lasso
      - K-Nearest Neighbors
      - Decision Trees
      - Random Forest Regressor
      - AdaBoost
      - Gradient Boosting Regressor
      - Support Vector Regressor (SVR)

4. **Model Selection**

    - Shortlisted top-k performing models based on R² Score and Mean Squared Error (MSE)

5. **Hyperparameter Optimization**
  
    - Conducted exhaustive GridSearchCV on the top-k models
    
    - Tuned key hyperparameters to minimize overfitting and maximize accuracy

6. **Experiment Tracking with MLflow**

    - Tracked metrics, visualizations, and parameters for every experiment
    
    - Logged all artifacts including models, pipelines, and transformers for easy versioning

7. **Pipeline Orchestration with ZenML**

    - Enabled clean separation of stages (data engineering, model training, hyperparameter tuning, deployment)
    
    - Designed for reproducibility, scalability, and seamless CI/CD integration

8. **Model Deployment**

    - Final retrained model served via MLflow model registry
    
    - Integrated into a responsive Flask web app to deliver real-time price predictions based on user input

### Pipeline Workflow
  
  ![image](https://github.com/user-attachments/assets/94cf5ddc-d5f3-47d0-bfd2-19033729c0d1)


##  Installation & Setup

### Prerequisites
- Python 3.10+
- [UV](https://github.com/astral-sh/uv) (recommended) or pip
- Virtual environment manager (included in instructions)

### Initial Setup Using UV (Recommended)

1. **Install UV** (if not already installed):
  ```
  # bash
  pipx install uv

  # Using curl
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Or with pip
  python -m pip install uv
  ```
2. **Create & activate a virtual environment**:
```
uv venv <virtual-env-name>
```
3. **Activate Envoirnment**
```
# Linux/macOS
source <virtual-env-name>/bin/activate

# Windows
.\<virtual-env-name>\Scripts\Activate
```
4. **Install Dependencies**:
```
uv pip install -r requirements.txt
```

### ZenML Setup
```
# Initialize ZenML
zenml init

# Install MLflow integration
zenml integration install mlflow -y

# Register components (MLflow and Model Deployer)
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```
###  Usage
1. **Run the entire Pipeline**
```
python run_pipelines.py
```

2. **Start MLflow Dashboard**
```
# on a separate terminal instance
mlflow ui --backend-store-uri <mlflow_tracking_uri>
```

> Can get <mlflow_tracking_uri> using get_tracking_uri()

Access dashboard at: `http://localhost:5000`
3. **Launch Web App**
```
python app.py
```
Access Web App at: `http://localhost:5002`

4. **Optional: Access ZenML Dashboard**
```
zenml login --local --blocking
```
Access dashboard at: `http://localhost:8237`



## Application Screenshots

**Home Page** : A page with an introduction to the app

![CarPredict _home](https://github.com/user-attachments/assets/2b43f12c-c7c3-46c6-a4fa-57c794a838c2)

**Predict Page**: Form interface for users to input car details

![CarPredict _predict](https://github.com/user-attachments/assets/241444d8-bcef-416a-b6a2-f1efd44d8bc7)


**Result Page**: Estimated resale price shown in Indian Rupees

![CarPredict _result](https://github.com/user-attachments/assets/c090cfa6-34d7-4dfc-82d5-9fa0af23e29d)


## Folder Structure
```
vroom-value/
├── analysis/            # Notebooks or scripts for exploratory data analysis (EDA)
├── configs/             # YAML/JSON config files for pipelines and parameters
├── data/                # Directory for raw input data
├── extracted_data/      # Cleaned and structured data extracted to CSVs
├── pipelines/           # ZenML pipeline definitions and orchestration logic
├── src/                 # Core machine learning logic and helper modules
├── static/              # Static assets like CSS, images, and JS files
├── steps/               # Custom ZenML steps used in pipelines (e.g., preprocessing, training)
├── templates/           # HTML templates for the Flask frontend
├── utils/               # Shared utility functions across the project
├── app.py               # Entry point for running the Flask web app
├── pyproject.toml       # Project metadata and dependency management (via uv or poetry)
├── requirements.txt     # Explicit list of Python dependencies
├── run_pipeline.py      # Script to trigger ZenML pipeline and launch the model
└── README.md            # Project overview and documentation
```
## Technologies Stack

**Programming Language** 

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Machine Learning and MLOps**  

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ff9933?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-45c4e6?style=for-the-badge&logo=mlflow&logoColor=white)
![ZenML](https://img.shields.io/badge/ZenML-421c92?style=for-the-badge&logo=zig&logoColor=white)
![Category Encoders](https://img.shields.io/badge/CategoryEncoders-DB3069?style=for-the-badge&logo=databricks&logoColor=white)

**Data Manipulation** 

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

**Visualization** 

![Seaborn](https://img.shields.io/badge/Seaborn-7db0bc?style=for-the-badge&logo=chartdotjs&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)

**Web Framework**  

![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

**Frontend**  

![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

**Package Manager**

![uv](https://img.shields.io/badge/uv-311f3a?style=for-the-badge&logo=zap&logoColor=white)

## Future Improvements

- **Advanced Feature Engineering**: Introduce sophisticated feature engineering techniques to improve model accuracy

- **Exploration of Additional Models**: Expand the range of regression models to capture more complex patterns in the data.

- **Database Integration for Data Ingestion**: Enhance data ingestion to support dynamic and scalable data sources.

- **Enhanced Pipeline Automation**: Further streamline the ZenML pipelines for greater efficiency and flexibility.

- **Cloud Deployment**: Transition the application to a cloud-based infrastructure for scalability


