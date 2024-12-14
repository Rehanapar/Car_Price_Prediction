from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import logging
 
# Initialize Flask app
app = Flask(__name__)
 
# Load the pre-trained model
with open('car_predict.pkl', 'rb') as file:
    model = pickle.load(file)
 
# Load the StandardScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
 
# Load LabelEncoders for categorical columns
label_encoders = {}
categorical_columns = ['model', 'motor_type', 'type', 'status']
for col in categorical_columns:
    with open(f'{col}_encoder.pkl', 'rb') as file:
        label_encoders[col] = pickle.load(file)
 
car = pd.read_csv('train.csv')
 
# Define a route for the main page
@app.route('/')
def index():
    models = sorted(car['model'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    motor_type = sorted(car['motor_type'].unique())
    # wheel = sorted(car['wheel'].unique())
    # color = sorted(car['color'].unique())
    type = sorted(car['type'].unique())
    status = sorted(car['status'].unique())
 
    return render_template('index.html', models=models, years=year, motor_types=motor_type,  types = type,status=status)
 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        model_input = request.form.get('model')
        year = request.form.get('year')
        motor_type = request.form.get('motor_type')
        running = request.form.get('running')
        type_input = request.form.get('type')
        status = request.form.get('status')
        motor_volume = request.form.get('motor_volume')
 
        # Encode categorical features using LabelEncoders
        encoded_model = label_encoders['model'].transform([model_input])[0]
        encoded_motor_type = label_encoders['motor_type'].transform([motor_type])[0]
        encoded_type = label_encoders['type'].transform([type_input])[0]
        encoded_status = label_encoders['status'].transform([status])[0]
 
        # Prepare input for prediction
        input_data_array = np.array([[encoded_model, year, encoded_motor_type, running, encoded_type, encoded_status, motor_volume]]).reshape(1, 7)
 
        # Create the DataFrame
        input_data = pd.DataFrame(columns=['model', 'year', 'motor_type', 'running', 'type', 'status', 'motor_volume'],
                                  data=input_data_array)
 
        # Scale the input data
        scaled_input = scaler.transform(input_data)
 
        # Use the loaded model to predict
        prediction = model.predict(scaled_input)
 
        # Log the input and prediction
        logging.info(f'Prediction input: {input_data}')
        logging.info(f'Prediction result: {np.round(prediction[0], 2)}')
 
        # Render the prediction result in a new HTML page
        return render_template('prediction.html', prediction=np.round(prediction[0], 2))
 
    except Exception as e:
        logging.error(f'Error during prediction: {str(e)}')
        return jsonify(error=str(e)), 500
 
 
if __name__ == '__main__':
    app.run(debug=True)
 