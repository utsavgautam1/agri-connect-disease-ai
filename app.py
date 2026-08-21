import io
import json
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "disease_model.tflite")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "model", "class_names.json")
IMG_SIZE = 224  # EfficientNetB0 default input size — adjust if yours differs

# ── Load TFLite model once at startup ──────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Falls back to full tensorflow if tflite_runtime isn't installed
    import tensorflow as tf
    tflite = tf.lite

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # IMPORTANT: do NOT divide by 255 here. EfficientNetB0's preprocessing
    # is baked into the model itself (confirmed in the training notebook,
    # cell 13) — it expects raw [0, 255] float32 values, not [0, 1].
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    # Frontend (diseaseService.js) sends the image under the key 'file'
    if "file" not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'file'."}), 400

    file = request.files["file"]
    image_bytes = file.read()

    try:
        input_data = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_idx = int(np.argmax(output_data))
    confidence = float(output_data[predicted_idx])

    # Frontend (diseaseService.js) reads `predicted_class` and `confidence`
    result = {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence, 4),
        "top_3": [
            {"class": CLASS_NAMES[i], "confidence": round(float(output_data[i]), 4)}
            for i in np.argsort(output_data)[-3:][::-1]
        ],
    }
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)