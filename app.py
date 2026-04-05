import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
import requests as req

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "best_tn5000_model.keras")
IMG_SIZE   = (224, 224)

def download_model_if_missing():
    if os.path.exists(MODEL_PATH):
        print(f"Model found at: {MODEL_PATH}")
        return
    model_url = os.environ.get("MODEL_URL", "")
    if not model_url:
        raise FileNotFoundError("MODEL_URL env var not set")
    print(f"Downloading model from: {model_url}")
    r = req.get(model_url, stream=True, timeout=300)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded!")

download_model_if_missing()
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        img_bytes   = file.read()
        img         = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_array   = np.expand_dims(
                          np.array(img_resized, dtype=np.float32), axis=0)
        prob              = float(model.predict(img_array, verbose=0)[0][0])
        benign_pct        = round((1 - prob) * 100, 2)
        malignant_pct     = round(prob * 100, 2)
        predicted_class   = "Malignant" if prob >= 0.5 else "Benign"
        confidence        = malignant_pct if prob >= 0.5 else benign_pct
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({
            "predicted_class": predicted_class,
            "confidence":      confidence,
            "benign_pct":      benign_pct,
            "malignant_pct":   malignant_pct,
            "image_b64":       img_b64,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)