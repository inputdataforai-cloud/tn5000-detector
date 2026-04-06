import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import requests as req
import traceback

app = Flask(__name__)

TFLITE_PATH = "model.tflite"
IMG_SIZE    = (224, 224)
interpreter = None

def download_tflite_if_missing():
    if os.path.exists(TFLITE_PATH):
        print(f"TFLite model found!", flush=True)
        return
    model_url = os.environ.get("MODEL_URL", "")
    if not model_url:
        raise FileNotFoundError("MODEL_URL not set")
    print(f"Downloading TFLite model...", flush=True)
    r = req.get(model_url, stream=True, timeout=120)
    r.raise_for_status()
    with open(TFLITE_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    size = os.path.getsize(TFLITE_PATH) / (1024*1024)
    print(f"TFLite downloaded! Size: {size:.2f} MB", flush=True)

def load_interpreter():
    global interpreter
    import tensorflow as tf
    print("Loading TFLite interpreter...", flush=True)
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    print("TFLite interpreter ready!", flush=True)

def predict_image(img_array):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return float(output[0][0])

try:
    download_tflite_if_missing()
    load_interpreter()
except Exception as e:
    print(f"STARTUP ERROR: {e}", flush=True)
    traceback.print_exc()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("Predict called", flush=True)
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        print(f"File: {file.filename}", flush=True)

        img_bytes   = file.read()
        img         = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_array   = np.expand_dims(
                          np.array(img_resized, dtype=np.float32), axis=0)

        print("Running prediction...", flush=True)
        prob = predict_image(img_array)
        print(f"Prediction: {prob:.4f}", flush=True)

        benign_pct      = round((1 - prob) * 100, 2)
        malignant_pct   = round(prob * 100, 2)
        predicted_class = "Malignant" if prob >= 0.5 else "Benign"
        confidence      = malignant_pct if prob >= 0.5 else benign_pct

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        print(f"Result: {predicted_class} {confidence}%", flush=True)
        return jsonify({
            "predicted_class": predicted_class,
            "confidence":      confidence,
            "benign_pct":      benign_pct,
            "malignant_pct":   malignant_pct,
            "image_b64":       img_b64,
        })

    except Exception as e:
        print(f"PREDICT ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": interpreter is not None,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)