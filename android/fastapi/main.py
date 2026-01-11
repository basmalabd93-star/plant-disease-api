import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import json
import os

# FastAPI app
app = FastAPI()

# تحديد مجلد العمل على Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تحميل نموذج TFLite
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.tflite")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# تحميل التسميات (labels)
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# تحميل بيانات العلاجات
TREATMENTS_PATH = os.path.join(BASE_DIR, "treatments.json")
with open(TREATMENTS_PATH, "r", encoding="utf-8") as f:
    treatments = json.load(f)

# دالة لتحضير الصورة للنموذج
def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Endpoint لاستقبال POST مع الصورة
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
    except Exception:
        return JSONResponse(
            {"error": "ملف غير صالح، الرجاء إرسال صورة"},
            status_code=400
        )

    input_data = preprocess_image(image).astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    class_idx = int(np.argmax(output_data))
    disease_name = labels[class_idx]
    treatment = treatments.get(disease_name, "لا يوجد علاج")

    return {
        "disease": disease_name,
        "treatment": treatment
    }

# للتشغيل المحلي (اختياري)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
