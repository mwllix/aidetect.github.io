from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open("svc.pkl", "rb"))

bp_map = {"95/80": 0, "120/80": 1, "130/80": 2, "140/80": 3, "145/80": 4}
gender_map = {"Male": 0, "Female": 1, "Both": 2}
yn_map = {"Yes": 1, "No": 0}

@app.route("/")
def home():
    return "AI is ready!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_vector = [
        float(data["age"]),
        bp_map.get(data["bp"], 0),
        float(data["massBefore"]),
        float(data["massAfter"]),
        float(data["urine"]),
        float(data["water"]),
        gender_map.get(data["gender"], 0),
        yn_map.get(data["fatigue"]),
        yn_map.get(data["edema"]),
        yn_map.get(data["confusion"]),
        yn_map.get(data["cold"]),
        yn_map.get(data["thirst"]),
    ]
    prediction = model.predict([input_vector])[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
