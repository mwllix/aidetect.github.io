from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import os
import pandas as pd # นำเข้าไลบรารี pandas เพื่อจัดการข้อมูล

app = Flask(__name__)
CORS(app)

# โหลดโมเดล SVC และ Label Encoder
try:
    model = pickle.load(open("svc.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    
    # กำหนดรายการคอลัมน์ทั้งหมด 62 คอลัมน์ที่โมเดลคาดหวัง
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

        # ตรวจสอบว่ามีข้อมูลครบถ้วน
        required_fields = ["age", "bp", "massBefore", "massAfter", "urine", "water", "gender", "fatigue", "edema", "confusion", "cold", "thirst"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # สร้าง DataFrame จากข้อมูลที่ได้รับ
        input_df = pd.DataFrame([data])
        
        # จัดการข้อมูลให้อยู่ในรูปแบบเดียวกับตอนเทรนโมเดล (One-Hot Encoding)
        # เนื่องจากฟีเจอร์ส่วนใหญ่เป็น categorical ที่มาจากฟอร์มเก่า 
        # เราจึงต้องสร้างคอลัมน์ใหม่ให้ครบถ้วน
        
        # สร้าง DataFrame เปล่าที่มีคอลัมน์ครบ 62 คอลัมน์และค่าเริ่มต้นเป็น 0
        input_vector = pd.DataFrame(0, index=[0], columns=expected_columns)
        
        # ใส่ค่าจากข้อมูลที่ได้รับลงในคอลัมน์ที่เกี่ยวข้อง
        
        # ใส่ค่าตัวเลข (ถ้ามี)
        input_vector["age"] = float(data.get("age", 0))
        input_vector["massBefore"] = float(data.get("massBefore", 0))
        input_vector["massAfter"] = float(data.get("massAfter", 0))
        input_vector["urine"] = float(data.get("urine", 0))
        input_vector["water"] = float(data.get("water", 0))
        
        # ใส่ค่าสำหรับข้อมูล categorical (One-Hot Encoded)
        # ตรวจสอบว่าคอลัมน์นั้นมีอยู่ใน expected_columns ก่อน
        if f"bp_{data['bp']}" in expected_columns:
            input_vector[f"bp_{data['bp']}"] = 1
        
        if f"gender_{data['gender']}" in expected_columns:
            input_vector[f"gender_{data['gender']}"] = 1
            
        if f"fatigue_{data['fatigue']}" in expected_columns:
            input_vector[f"fatigue_{data['fatigue']}"] = 1
            
        if f"edema_{data['edema']}" in expected_columns:
            input_vector[f"edema_{data['edema']}"] = 1
            
        if f"confusion_{data['confusion']}" in expected_columns:
            input_vector[f"confusion_{data['confusion']}"] = 1
            
        if f"cold_{data['cold']}" in expected_columns:
            input_vector[f"cold_{data['cold']}"] = 1
            
        if f"thirst_{data['thirst']}" in expected_columns:
            input_vector[f"thirst_{data['thirst']}"] = 1

        # ทำนายโรค
        predicted_label = model.predict(input_vector)[0]
        predicted_disease = le.inverse_transform([predicted_label])[0]

        return jsonify({"prediction": predicted_disease})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
