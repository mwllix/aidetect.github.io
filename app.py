import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# --------------------
# 1. การตั้งค่าและโหลดโมเดล
# --------------------

app = Flask(__name__)
# อนุญาตให้ Frontend (React) เรียกใช้ API ข้ามโดเมนได้
CORS(app) 

model = None
le = None # Label Encoder for converting numeric prediction to disease name
MODEL_CLASSES = []

try:
    # โหลดโมเดล SVC
    model = joblib.load('svc.pkl')
    # โหลด Label Encoder สำหรับแปลงผลลัพธ์ตัวเลขกลับเป็นชื่อโรค
    le = joblib.load('label_encoder.pkl')
    
    # ดึงรายชื่อคลาสที่โมเดลรู้จักจาก Label Encoder หรือโมเดล
    # (สมมติว่า le.classes_ เก็บชื่อโรค เช่น ['Diabetes', 'Kidney Failure', 'Normal'])
    if hasattr(le, 'classes_'):
        MODEL_CLASSES = le.classes_.tolist()
    elif hasattr(model, 'classes_'):
        MODEL_CLASSES = model.classes_.tolist()
    
    print("AI Model (svc.pkl) and Label Encoder loaded successfully.")
    print(f"Known Classes: {MODEL_CLASSES}")
    
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    model = None
    le = None

# รายการ Feature ทั้ง 62 ตัวตามที่ Frontend ส่งมา
# **สำคัญ:** ต้องตรงกับลำดับที่ใช้ในการฝึกโมเดล!
FEATURE_NAMES_62 = [
    "0-1","5-15","10-20","40+","45+","50+","60+","65+","<500","<800","350-550","800-2000",
    "2000-3000",">2000",">3000",">=18.5",">=25","N/a","<=2700",">=3700","120/80",">130/80",
    "<130/80",">=130/80",">140/80","95-145/80","Mass","Negligible","Overweight","M+/-","M+7Kg",
    "-M+7Kg or 10Kg","M minus 1Kg","M minus 5Kg","M minus 10Kg","M minus 0.5-1Kg","<M",
    "No change","Negligible.1","Male","Female","Wheezing","Headache","Short Breaths",
    "Rapid Breathing","Anxiety","Urine at Night","Irritability","Blurred Vision",
    "Slow Healing","Dry Mouth","Muscle Aches","Nausea/Vomiting","Insomnia","Chest Pain",
    "Dizziness","Nosebleeds","Foamy Urine","Abdominal Pain","Itchy Skin","Dark Urine",
    "Bone Pain"
]

# --------------------
# 2. API Endpoint สำหรับการทำนาย (ส่วนที่ปรับปรุง)
# --------------------

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None:
        return jsonify({"error": "AI model or Label Encoder is not loaded."}), 500
        
    try:
        data = request.get_json()
        
        # 2.1 เตรียมข้อมูล Feature Vector (62-dimensional array)
        feature_vector = [data.get(name, 0) for name in FEATURE_NAMES_62]
        X = np.array(feature_vector).reshape(1, -1)
        
        # 2.2 ทำนายผลลัพธ์ (ผลลัพธ์จะเป็นตัวเลข)
        # SVC.predict() จะคืนค่าเป็นตัวเลข (เช่น 0, 1, 2)
        prediction_numeric = model.predict(X) 
        
        # 2.3 แปลงผลลัพธ์ตัวเลขกลับเป็นชื่อโรค
        prediction_class = le.inverse_transform(prediction_numeric)[0]
        
        confidence_score = 0.0
        
        # 2.4 คำนวณความน่าจะเป็น (Confidence Score)
        try:
            # ⭐️ วิธีที่ 1: ใช้ .predict_proba() (ถ้าโมเดลถูกฝึกด้วย probability=True)
            probabilities = model.predict_proba(X)[0] 
            
            # หา index ของคลาสที่ทำนายได้จากผลลัพธ์ตัวเลข
            prediction_index = prediction_numeric[0]
            confidence_score = probabilities[prediction_index]
            
        except AttributeError:
            # ⭐️ วิธีที่ 2: ถ้า .predict_proba() ไม่พร้อมใช้งาน (SVC ทั่วไป)
            # เราใช้ .predict() และตั้งค่าความมั่นใจเป็น 1.0 (100%) 
            # หรือใช้ decision_function เพื่อหาค่าความมั่นใจสัมพัทธ์
            
            # หากต้องการใช้ 100%
            confidence_score = 1.0 
            
            # หากต้องการใช้ decision_function (ซับซ้อนกว่า แต่ให้ค่าความมั่นใจสัมพัทธ์)
            # decision = model.decision_function(X)[0]
            # confidence_score = np.max(decision) / (np.sum(np.abs(decision))) # เป็นค่าสัมพัทธ์ ไม่ใช่ probability แท้จริง

            app.logger.warning("SVC model does not support predict_proba. Using default confidence score.")

        # 2.5 ส่งผลลัพธ์กลับไปยัง Frontend
        return jsonify({
            "prediction": str(prediction_class),
            "probability": float(confidence_score) 
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# --------------------
# 3. API Endpoint สำหรับตรวจสอบสถานะ
# --------------------

@app.route('/api/status', methods=['GET'])
def get_status():
    status = "online" if model and le else "offline (Model or Encoder Error)"
    return jsonify({"status": status})

if __name__ == '__main__':
    # รันเซิร์ฟเวอร์
    app.run(debug=True, host='0.0.0.0', port=5000)
