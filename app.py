import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging
import os

# --------------------
# 1. การตั้งค่าและโหลดโมเดล
# --------------------

app = Flask(__name__)
# อนุญาตให้ Frontend (React) เรียกใช้ API ข้ามโดเมนได้
CORS(app)

# ตั้งค่า Logger ให้แสดงข้อมูลเวลาด้วย
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = None
le = None # Label Encoder
MODEL_CLASSES = []
MODEL_LOADED = False # สถานะสำหรับการตรวจสอบโมเดล

# 🎯 เพิ่ม: กำหนดเกณฑ์ความมั่นใจ (CONFIDENCE_THRESHOLD)
CONFIDENCE_THRESHOLD = 0.50 # 50%

try:
    # โหลดโมเดล SVC และ Label Encoder (สมมติว่าไฟล์อยู่ใน Path ที่ถูกต้อง)
    model = joblib.load('svc.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # ดึงรายชื่อคลาสที่โมเดลรู้จัก
    if hasattr(le, 'classes_'):
        MODEL_CLASSES = le.classes_.tolist()
    elif hasattr(model, 'classes_'):
        MODEL_CLASSES = model.classes_.tolist()
    
    app.logger.info("AI Model (svc.pkl) and Label Encoder loaded successfully.")
    app.logger.info(f"Known Classes: {MODEL_CLASSES}")
    MODEL_LOADED = True
    
except Exception as e:
    app.logger.error(f"Error loading model or label encoder: {e}")
    model = None
    le = None
    MODEL_LOADED = False

# รายการ Feature ทั้ง 62 ตัว
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
# 2. API Endpoint สำหรับหน้าแรก (Health Check)
# --------------------
@app.route('/', methods=['GET'])
def home():
    """จัดการ Request GET ไปยัง URL หลัก"""
    return jsonify({
        "message": "AI Detection API Server is running.",
        "status": "online" if MODEL_LOADED else "model_error",
        "instructions": "Use the /predict endpoint with a POST request and JSON body to get a prediction."
    }), 200

# --------------------
# 3. API Endpoint สำหรับการทำนาย
# --------------------

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "AI model or Label Encoder is not loaded."}), 500
        
    try:
        data = request.get_json()
        
        # 3.1 เตรียมข้อมูล Feature Vector
        feature_vector = [data.get(name, 0) for name in FEATURE_NAMES_62]
        X = np.array(feature_vector).reshape(1, -1)
        
        # 3.2 ทำนายความน่าจะเป็นของทุกคลาส
        probas = model.predict_proba(X)[0] 
        
        # 3.3 หาความน่าจะเป็นสูงสุด (Confidence Score) และดัชนีคลาสที่ทำนาย
        max_proba = np.max(probas)
        predicted_class_index = np.argmax(probas)
        
        # 3.4 แปลงดัชนีเป็นชื่อโรคที่โมเดลมั่นใจที่สุด (Best Candidate)
        predicted_disease = le.inverse_transform([model.classes_[predicted_class_index]])[0]
        
        # -----------------------------------------------------------
        # 🎯 โค้ดที่เพิ่ม: ตรวจสอบเกณฑ์ Confidence Threshold (50%)
        # -----------------------------------------------------------
        if max_proba < CONFIDENCE_THRESHOLD:
            final_prediction = "No Matching Disease"
            confidence_score = max_proba
        else:
            final_prediction = predicted_disease
            confidence_score = max_proba

        # 3.5 ส่งผลลัพธ์กลับไปยัง Frontend
        app.logger.info(f"Prediction: {final_prediction}, Confidence: {confidence_score*100:.2f}%")
        return jsonify({
            # ส่งผลลัพธ์สุดท้าย (เป็นชื่อโรค หรือ "No Matching Disease")
            "prediction": str(final_prediction), 
            # ส่งคะแนนความมั่นใจสูงสุด
            "probability": float(confidence_score) 
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# --------------------
# 4. API Endpoint สำหรับตรวจสอบสถานะ (Health Check)
# --------------------

@app.route('/api/wakeup', methods=['GET'])
def wakeup_server():
    """ใช้สำหรับปลุกเซิร์ฟเวอร์ Render ที่ Sleep อยู่"""
    app.logger.info("Received wake-up request from frontend.")
    return jsonify({
        "message": "Server received wake-up call.",
        "status": "initializing" if not MODEL_LOADED else "online"
    }), 200
    
# ⭐️ Endpoint ใหม่: /api/status ที่ Frontend ใช้ตรวจสอบสถานะ
@app.route('/api/status', methods=['GET'])
def check_status():
    """ใช้สำหรับ Frontend ตรวจสอบสถานะว่าโมเดลพร้อมใช้งานหรือไม่"""
    status = "online" if MODEL_LOADED else "waiting"
    app.logger.info(f"Status check requested. Status: {status}")
    return jsonify({
        "message": f"AI Server status: {status}",
        "status": status, # คืนค่า 'online' หรือ 'waiting'
        "model_loaded": MODEL_LOADED
    }), 200

# --------------------
# 5. การเริ่มต้น Server
# --------------------

if __name__ == '__main__':
    print("\nRegistered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
        
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
