from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import os # Import the os module to access environment variables

app = Flask(__name__)
CORS(app)

# Load SVC model
# Ensure these files (svc.pkl, label_encoder.pkl) are in the same directory as app.py
# or provide a correct path relative to the app.py file on Render.com
try:
    model = pickle.load(open("svc.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError as e:
    print(f"Error loading pickle files: {e}. Make sure svc.pkl and label_encoder.pkl are in the correct location.")
    # You might want to handle this more gracefully in a production environment,
    # e.g., by returning an error message or stopping the app.

bp_map = {"95/80": 0, "120/80": 1, "130/80": 2, "140/80": 3, "145/80": 4}
gender_map = {"Male": 0, "Female": 1, "Both": 2}
yn_map = {"Yes": 1, "No": 0}

@app.route("/")
def home():
    return "AI is ready!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data provided"}), 400

        # Validate required fields
        required_fields = ["age", "bp", "massBefore", "massAfter", "urine", "water", "gender", "fatigue", "edema", "confusion", "cold", "thirst"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        input_vector = [
            float(data["age"]),
            bp_map.get(data["bp"], 0),
            float(data["massBefore"]),
            float(data["massAfter"]),
            float(data["urine"]),
            float(data["water"]),
            gender_map.get(data["gender"], 0),
            yn_map.get(data["fatigue"], 0),
            yn_map.get(data["edema"], 0),
            yn_map.get(data["confusion"], 0),
            yn_map.get(data["cold"], 0),
            yn_map.get(data["thirst"], 0),
        ]

        # Predict disease label
        predicted_label = model.predict([input_vector])[0]
        predicted_disease = le.inverse_transform([predicted_label])[0]

        return jsonify({"prediction": predicted_disease})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

if __name__ == "__main__":
    # Get the port from the environment variable provided by Render.com
    # If not found (e.g., running locally), default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Run the Flask app, binding to 0.0.0.0 to be accessible externally
    app.run(host="0.0.0.0", port=port, debug=True) # debug=True is good for development, but consider False for production
