import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging # เพิ่ม logging เพื่อจัดการข้อความเตือน

# --------------------
# 1. การตั้งค่าและโหลดโมเดล
# --------------------

app = Flask(__name__)
# อนุญาตให้ Frontend (React) เรียกใช้ API ข้ามโดเมนได้
CORS(app)

# ตั้งค่า Logger
app.logger.setLevel(logging.INFO)

model = None
le = None # Label Encoder for converting numeric prediction to disease name
MODEL_CLASSES = []

try:
    # โหลดโมเดล SVC
    model = joblib.load('svc.pkl')
    # โหลด Label Encoder สำหรับแปลงผลลัพธ์ตัวเลขกลับเป็นชื่อโรค
    le = joblib.load('label_encoder.pkl')
    
    # ดึงรายชื่อคลาสที่โมเดลรู้จักจาก Label Encoder
    if hasattr(le, 'classes_'):
        MODEL_CLASSES = le.classes_.tolist()
    elif hasattr(model, 'classes_'):
        MODEL_CLASSES = model.classes_.tolist()
    
    app.logger.info("AI Model (svc.pkl) and Label Encoder loaded successfully.")
    app.logger.info(f"Known Classes: {MODEL_CLASSES}")
    
except Exception as e:
    app.logger.error(f"Error loading model or label encoder: {e}")
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
# 2. API Endpoint สำหรับหน้าแรก (แก้ไข 404 Error)
# --------------------
@app.route('/', methods=['GET'])
def home():
    """จัดการ Request GET ไปยัง URL หลักเพื่อป้องกัน 404 Error"""
    # เพื่อให้เบราว์เซอร์ไม่แสดงข้อผิดพลาด "Not Found" เมื่อผู้ใช้เข้าสู่ URL หลัก
    return jsonify({
        "message": "AI Detection API Server is running.",
        "status": "online",
        "instructions": "Use the /predict endpoint with a POST request and JSON body to get a prediction."
    }), 200


# --------------------
# 3. API Endpoint สำหรับการทำนาย
# --------------------

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None:
        return jsonify({"error": "AI model or Label Encoder is not loaded."}), 500
        
    try:
        data = request.get_json()
        
        # 3.1 เตรียมข้อมูล Feature Vector (62-dimensional array)
        feature_vector = [data.get(name, 0) for name in FEATURE_NAMES_62]
        X = np.array(feature_vector).reshape(1, -1)
        
        # 3.2 ทำนายผลลัพธ์ (ผลลัพธ์จะเป็นตัวเลข)
        prediction_numeric = model.predict(X) 
        
        # 3.3 แปลงผลลัพธ์ตัวเลขกลับเป็นชื่อโรค
        prediction_class = le.inverse_transform(prediction_numeric)[0]
        
        confidence_score = 0.0
        
        # 3.4 คำนวณความน่าจะเป็น (Confidence Score)
        try:
            # ⭐️ วิธีที่ 1: ใช้ .predict_proba() (ถ้าโมเดลถูกฝึกด้วย probability=True)
            probabilities = model.predict_proba(X)[0]
            
            # หา index ของคลาสที่ทำนายได้จากผลลัพธ์ตัวเลข
            prediction_index = prediction_numeric[0]
            confidence_score = probabilities[prediction_index]
            
        except AttributeError:
            # ⭐️ วิธีที่ 2: ถ้า .predict_proba() ไม่พร้อมใช้งาน (SVC ทั่วไป)
            confidence_score = 1.0 
            app.logger.warning("SVC model does not support predict_proba. Using default confidence score.")

        # 3.5 ส่งผลลัพธ์กลับไปยัง Frontend
        return jsonify({
            "prediction": str(prediction_class),
            "probability": float(confidence_score) 
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# --------------------
# 4. API Endpoint สำหรับตรวจสอบสถานะ (Health Check)
# --------------------

@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint สำหรับ Health Check หรือตรวจสอบสถานะโมเดล"""
    status = "online" if model and le else "offline (Model or Encoder Error)"
    return jsonify({"status": status, "model_loaded": bool(model), "encoder_loaded": bool(le)})
