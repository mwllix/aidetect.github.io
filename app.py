from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load SVC model
# ตรวจสอบให้แน่ใจว่าไฟล์เหล่านี้ (svc.pkl, label_encoder.pkl) อยู่ในไดเรกทอรีเดียวกันกับ app.py
try:
    model = pickle.load(open("svc.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError as e:
    print(f"Error loading pickle files: {e}. ตรวจสอบให้แน่ใจว่า svc.pkl และ label_encoder.pkl อยู่ในตำแหน่งที่ถูกต้อง")
    # ในสภาพแวดล้อมการผลิต คุณอาจต้องการจัดการข้อผิดพลาดนี้อย่างละเอียดมากขึ้น
    # เช่น ส่งคืนข้อความแสดงข้อผิดพลาดหรือหยุดแอปพลิเคชัน

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
            return jsonify({"error": "ไม่พบข้อมูล JSON ที่ถูกต้อง"}), 400

        # Validate required fields and their values
        required_fields = ["age", "bp", "massBefore", "massAfter", "urine", "water", "gender", "fatigue", "edema", "confusion", "cold", "thirst"]
        for field in required_fields:
            if field not in data or data[field] is None: # เพิ่มการตรวจสอบ data[field] is None
                return jsonify({"error": f"ขาดฟิลด์ที่จำเป็นหรือค่าเป็น null: {field}"}), 400

        # แปลงค่าตัวเลขอย่างปลอดภัย
        try:
            age = float(data["age"])
            massBefore = float(data["massBefore"])
            massAfter = float(data["massAfter"])
            urine = float(data["urine"])
            water = float(data["water"])
        except ValueError:
            return jsonify({"error": "ค่าตัวเลขสำหรับ age, massBefore, massAfter, urine, หรือ water ไม่ถูกต้อง"}), 400

        input_vector = [
            age,
            bp_map.get(data["bp"], 0), # ใช้ .get() เพื่อให้ค่าเริ่มต้นเป็น 0 หากไม่พบ
            massBefore,
            massAfter,
            urine,
            water,
            gender_map.get(data["gender"], 0), # ใช้ .get() เพื่อให้ค่าเริ่มต้นเป็น 0 หากไม่พบ
            yn_map.get(data["fatigue"], 0), # ใช้ .get() เพื่อให้ค่าเริ่มต้นเป็น 0 หากไม่พบ
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
        return jsonify({"error": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์", "details": str(e)}), 500

if __name__ == "__main__":
    # รับพอร์ตจากตัวแปรสภาพแวดล้อมที่ Render.com จัดหาให้
    # หากไม่พบ (เช่น รันในเครื่อง) ให้ใช้ค่าเริ่มต้นเป็น 5000
    port = int(os.environ.get("PORT", 5000))
    # รันแอป Flask โดยผูกกับ 0.0.0.0 เพื่อให้สามารถเข้าถึงได้จากภายนอก
    app.run(host="0.0.0.0", port=port, debug=True) # debug=True เหมาะสำหรับการพัฒนา แต่ควรเป็น False สำหรับการผลิต
