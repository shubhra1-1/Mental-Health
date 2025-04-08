from flask import Flask, request, jsonify
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from pppp import X_train, y_train

# Assuming X_train and y_train are already available
app = Flask(__name__)

# Train and save the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'random_forest_model.pkl')

# Train, fit, and save the scaler
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'scaler.pkl')

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)

    # Scale the input features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
