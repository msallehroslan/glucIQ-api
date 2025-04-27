# --- GlucIQ Flask API Server ---

from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Step 1: Initialize Flask App ---
app = Flask(__name__)

# --- Step 2: Load GlucIQ Model and Scalers ---
model = load_model('gluciq_8features_model.keras')
input_scaler_glucose = joblib.load('input_scaler_glucose.pkl')
input_scaler_profile = joblib.load('input_scaler_profile.pkl')
output_scaler = joblib.load('output_scaler.pkl')

print("✅ GlucIQ Model and Scalers loaded successfully.")

# --- Step 3: Define Prediction Function ---
def gluciq_predict(glucose_sequence, user_profile):
    glucose_sequence = np.array(glucose_sequence).reshape(1, -1)
    glucose_sequence_scaled = input_scaler_glucose.transform(glucose_sequence).reshape(1, 30, 1)

    user_profile = np.array(user_profile).reshape(1, -1)
    user_profile_scaled = input_scaler_profile.transform(user_profile)

    prediction_scaled = model.predict([glucose_sequence_scaled, user_profile_scaled])
    prediction = output_scaler.inverse_transform(prediction_scaled)[0][0]

    if prediction < 3.9:
        action = "🚨 Hypoglycemia risk detected. Please consume fast-acting carbohydrates and consult healthcare provider."
    elif prediction > 10.0:
        action = "⚠️ Hyperglycemia risk detected. Please consult your healthcare provider to review your diabetes management plan."
    else:
        action = "✅ Glucose level stable. Continue monitoring."

    return prediction, action

# --- Step 4: Define Flask API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'glucose_sequence' not in data or 'user_profile' not in data:
        return jsonify({'error': 'Invalid input. Please provide glucose_sequence and user_profile.'}), 400

    glucose_sequence = data['glucose_sequence']
    user_profile = data['user_profile']

    if len(glucose_sequence) != 30 or len(user_profile) != 8:
        return jsonify({'error': 'Glucose sequence must have 30 values and user profile must have 8 features.'}), 400

    predicted_glucose, action = gluciq_predict(glucose_sequence, user_profile)

    response = {
        'predicted_glucose': round(float(predicted_glucose), 2),  # ✅ fix float32 to normal float
        'ai_agent_action': action
    }
    return jsonify(response)

# --- Step 5: Run the Flask Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
