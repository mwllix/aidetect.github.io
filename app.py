import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

try:
    model = pickle.load(open("svc.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError as e:
    print(f"Error loading pickle files: {e}. Make sure all required files are in the correct location.")
    exit()

@app.route("/")
def home():
    return "AI is ready!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data provided"}), 400

        expected_columns = [
            "0-1", "5-15", "10-20", "40+", "45+", "50+", "60+", "65+",
            "<500", "<800", "350-550", "800-2000", "2000-3000", ">2000", ">3000",
            ">=18.5", ">=25", "N/a", "<=2700", ">=3700", "120/80", ">130/80",
            "<130/80", ">=130/80", ">140/80", "95-145/80", "Mass", "Negligible",
            "Overweight", "M+/-", "M+7Kg", "-M+7Kg or 10Kg", "M minus 1Kg",
            "M minus 5Kg", "M minus 10Kg", "M minus 0.5-1Kg", "<M", "No change",
            "Negligible.1", "Male", "Female", "Wheezing", "Headache",
            "Short Breaths", "Rapid Breathing", "Anxiety", "Urine at Night",
            "Irritability", "Blurred Vision", "Slow Healing", "Dry Mouth",
            "Muscle Aches", "Nausea/Vomiting", "Insomnia", "Chest Pain",
            "Dizziness", "Nosebleeds", "Foamy Urine", "Abdominal Pain",
            "Itchy Skin", "Dark Urine", "Bone Pain"
        ]


        input_vector = [data.get(col, 0) for col in expected_columns]
        
        predicted_label = model.predict([input_vector])[0]
        predicted_disease = le.inverse_transform([predicted_label])[0]

        return jsonify({"prediction": predicted_disease})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

@app.route("/notify-on-deploy", methods=["POST"])
def notify_on_deploy():
    data = request.get_json()
    status = data.get("status")
    service_name = data.get("service", {}).get("name")
    updated_at = data.get("updatedAt")
    
    print(f"[{datetime.now()}] Webhook received: Service '{service_name}' deployed successfully at {updated_at}. Status: {status}")
    
    return jsonify({"message": "Notification received and processed."}), 200

@app.route("/api/wakeup", methods=["GET", "POST"])
def wakeup():
    return jsonify({"message": "Server is awake and ready."}), 200

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({"status": "online"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
