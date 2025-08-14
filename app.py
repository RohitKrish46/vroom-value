from flask import Flask, render_template, request
import json
import requests
import numpy as np
import locale
app = Flask(__name__)


# Set locale to Indian format
locale.setlocale(locale.LC_ALL, 'en_IN')

def format_currency(value):
    return locale.format_string("%d", value, grouping=True)

# URL of the MLflow prediction server
url = "http://127.0.0.1:5001/invocations"

# # Load the trained model and preprocessor
# with open('./models/model.pkl', 'rb') as file:
#     model = pickle.load(file)
    
# with open('./models/preprocessor.pkl', 'rb') as file:
#     preprocessor = pickle.load(file)

@app.route('/')
def home():
    return render_template('carprice.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data and create DataFrame with proper column names
            features = {
                "dataframe_records" : 
                [
                    {   'car_name': request.form['car_name'],
                        'vehicle_age':int(request.form['vehicle_age']),
                        'km_driven': float(request.form['km_driven']),
                        'seller_type': request.form['seller_type'],
                        'fuel_type': request.form['fuel_type'],
                        'transmission_type': request.form['transmission_type'],
                        'mileage': float(request.form['mileage']),
                        'engine': float(request.form['engine']),
                        'max_power': float(request.form['max_power']),                        
                        'seats': int(request.form['seats']),                        
                    }
                ]

            }
            car_name = features['dataframe_records'][0]['car_name']
            # Convert the input data to JSON format
            json_data = json.dumps(features)
            print(json_data)

            # Set the headers for the request
            headers = {"Content-Type": "application/json"}

            # Send the POST request to the server
            response = requests.post(url, headers=headers, data=json_data)

            # Check the response status code
            if response.status_code == 200:
                # If successful, print the prediction result
                prediction = response.json()
                # Check if it's a list or dict; MLflow usually returns a list of predictions
                if isinstance(prediction, list):
                    log_prediction = prediction[0]
                else:
                    log_prediction = prediction["predictions"][0]  # fallback in case it's a dict
                original_scale_prediction = np.expm1(log_prediction)
                print("Prediction (original scale):", original_scale_prediction)
                formatted_prediction = format_currency(original_scale_prediction)

            else:
                # If there was an error, print the status code and the response
                print(f"Error: {response.status_code}")
                print(response.text)

            return render_template('result.html', price=formatted_prediction, car_name=car_name)
        
        except Exception as e:
            return render_template('index.html', 
                                error_text=f'Error making prediction: {str(e)}')
        
    return render_template('predict.html')

@app.route('/about')
def about():
    return "About Page (Coming Soon!)"

@app.route('/contact')
def contact():
    return "Contact Page (Coming Soon!)"

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5002, debug=True)
