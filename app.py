import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import requests as req
import traceback

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "best_tn5000_model.keras")
IMG_SIZE   = (224, 224)
model      = None

def download_model_if_missing():
    if os.path.exists(MODEL_PATH):
        print(f"Model found: {MODEL_PATH}", flush=True)
        return
    model_url = os.environ.get("MODEL_URL", "")
    if not model_url:
        raise FileNotFoundError("MODEL_URL not set")
    print(f"Downloading model...", flush=True)
    r = req.get(model_url, stream=True, timeout=600)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    size = os.path.getsize(MODEL_PATH) / (1024*1024)
    print(f"Model downloaded! Size: {size:.1f} MB", flush=True)

def load_model():
    global model
    if model is not None:
        return model
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    print("Loading model...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!", flush=True)
    return model

try:
    download_model_if_missing()
    load_model()
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
        print(f"File received: {file.filename}", flush=True)

        img_bytes   = file.read()
        print(f"Image bytes: {len(img_bytes)}", flush=True)

        img         = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_array   = np.expand_dims(
                          np.array(img_resized, dtype=np.float32), axis=0)
        print("Image preprocessed", flush=True)

        m    = load_model()
        print("Running prediction...", flush=True)
        prob = float(m.predict(img_array, verbose=0)[0][0])
        print(f"Prediction done: {prob}", flush=True)

        benign_pct      = round((1 - prob) * 100, 2)
        malignant_pct   = round(prob * 100, 2)
        predicted_class = "Malignant" if prob >= 0.5 else "Benign"
        confidence      = malignant_pct if prob >= 0.5 else benign_pct

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = {
            "predicted_class": predicted_class,
            "confidence":      confidence,
            "benign_pct":      benign_pct,
            "malignant_pct":   malignant_pct,
            "image_b64":       img_b64,
        }
        print(f"Returning result: {predicted_class}", flush=True)
        return jsonify(result)

    except Exception as e:
        print(f"PREDICT ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    