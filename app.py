from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import joblib 

app = Flask(_name_)
CORS(app)
print("Running ")
# Load the pre-trained model and scaler
ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
# Load your scaler (you should use the same scaler that you used during training)
scaler = StandardScaler()
scaler_mean = ...  # Load the mean from your training data
scaler_scale = ...  # Load the scale from your training data
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = data['input_data']

    # Standardize the input data
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    # Make a prediction using the model
    prediction = ada_clf.predict(std_data)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction[0]})

if _name_ == '_main_':
    app.run(port=5000)