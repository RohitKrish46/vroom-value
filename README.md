# Vroom Value

Welcome to Vroom Value, a machine learning-powered web application that helps you estimate the resale value of used cars in the Indian market. This project combines data science and web development to provide accurate, real-time price predictions based on various car features.

## 🔍 Project Overview
This repository presents a regression-based machine learning solution that predicts the price of used cars in India. The dataset used for training is sourced from Cardekho. Multiple models were explored, evaluated, and the best-performing one was integrated into a Flask-powered web app for end users.

## 🧠 Machine Learning Workflow

- ✅ Data Collection from the Indian car market (Cardekho)

- 🔧 Preprocessing including feature engineering and encoding

- 🧪 Model experimentation with a wide range of regression models

- 🏁 Model selection based on RMSE and performance

- 🧵 Integration with Flask for real-time predictions

- 🧊 Pickle used for saving the best model and transformer

## 🧪Models Explored

The following regression models were trained and compared:
- Linear Regression
- Lasso Regression
- Ridge Regression
- K-Nearest Neighbours
- Decision-Trees
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- Support Vector Regressor

## 🖥️ Web Application

### 🌐 Pages
1. 🏠 Home Page
 Provides an introduction to the app and its purpose.

2. 📋 Predict Page
  Allows users to fill in car details like:
 
    - Brand & Model 
    - Year of Manufacture 
    - Kilometers Driven 
    - Fuel Type 
    - Transmission 
    - Engine & Power 
    - Mileage 
    - Number of Seats 
    - Seller Type

3. 💸 Result Page
 Displays the predicted resale price in Indian currency format (₹).


## 🚀 How to Run Locally
1. Clone the Repository
```
git clone https://github.com/your-username/vroom-value.git
cd vroom-value
```
2. Set Up Virtual Environment
```
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```
3. Install Requirements
```
pip install -r requirements.txt
```
4. Start the Flask App
```
python app/app.py
```
5. Open the Web App
Navigate to port 5000

## 📸 App Screenshots

🏠 Home Page
"A page with an introduction to the app"

![CarPredict _home](https://github.com/user-attachments/assets/2b43f12c-c7c3-46c6-a4fa-57c794a838c2)

📋 Predict Page
"Form interface for users to input car details"

![CarPredict _predict](https://github.com/user-attachments/assets/241444d8-bcef-416a-b6a2-f1efd44d8bc7)


💸 Result Page
"Estimated resale price shown in Indian Rupees"

![CarPredict _result](https://github.com/user-attachments/assets/c090cfa6-34d7-4dfc-82d5-9fa0af23e29d)

## 🎯 Goal
Empower users to make smarter decisions when buying or selling used cars by leveraging machine learning predictions based on real-world data from the Indian automotive market.

## Language and Libraries

<p>
<a><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" alt="python"/></a>
<a><img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/></a>
<a><img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/></a>
 <a><img src="https://matplotlib.org/_static/logo2_compressed.svg"width="110"/></a>
<a><img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" alt="Seaborn"width="110"/></a>
<a><img src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white"></a>
</p>
