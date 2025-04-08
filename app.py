import os

from flask import Flask, request, jsonify, render_template, url_for, redirect
import joblib
import numpy as np
import json


app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

#Ensure the folder for saving input data exists
DATA_FOLDER = 'data'
os.makedirs(DATA_FOLDER, exist_ok=True)
INPUT_FILE = os.path.join(DATA_FOLDER, 'user_input.csv')

# Write the header to the file if it does not exist
if not os.path.exists(INPUT_FILE):
    with open(INPUT_FILE, 'w') as file:
        file.write("age,gender,employment_status,work_interfere\n")


@app.route('/')
def index():
    return render_template('prediction_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:  # Check if request is JSON
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()  # Parse JSON data

        # Extract features
        age = int(data['age'])
        gender = data['gender'].lower()
        employment_status = data['employment_status'].lower()
        work_interfere = data['work_interfere'].lower()
        family_history = data['family_history'].lower()

        # Map categorical variables
        gender_mapping = {'male': 0, 'female': 1}
        employment_mapping = {'employed': 0, 'unemployed': 1, 'student': 2, 'retired': 3}
        work_interfere_mapping = {'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3}
        family_history_mapping = {'yes': 1, 'no': 0}

        # Validate input mappings
        if (
            gender not in gender_mapping or
            employment_status not in employment_mapping or
            work_interfere not in work_interfere_mapping or
            family_history not in family_history_mapping
        ):
            return jsonify({'error': 'Invalid input values'}), 400

        # Convert categorical values to numerical ml
        gender_num = gender_mapping[gender]
        employment_num = employment_mapping[employment_status]
        work_interfere_num = work_interfere_mapping[work_interfere]
        family_history_num = family_history_mapping[family_history]

        # array
        features = np.array([[age, gender_num, employment_num, work_interfere_num, family_history_num]])

        # Scale the input data
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]  # 0 = No risk, 1 = At risk
        print(f"Prediction result: {prediction}")
        # Send response based on prediction
        if prediction == 1:
            return jsonify({'prediction': 1, 'redirect': url_for('chatbot')})
        else:
            return jsonify({'prediction': 0, 'message': "You seem to be doing fine! Stay healthy!"})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot')
def chatbot():
    return redirect("http://127.0.0.1:8501")  # Redirects to the Streamlit app

# # @app.route('/save_input', methods=['POST'])
# .....

if __name__ == '__main__':
    app.run(debug=True)


