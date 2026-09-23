"""
python_api.py  –  Thin Flask wrapper around run_yolo.py
Node.js posts an uploaded file here, gets JSON back.
"""

import os
import sys
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify

# Make sure run_yolo imports work from this directory
sys.path.insert(0, str(Path(__file__).parent))
from run_yolo import YOLOInference

app = Flask(__name__)
MODEL_PATH = "model/best.pt"

# Load the model once at startup (expensive – don't reload per request)
print("[Python API] Loading YOLO model...")
inference = YOLOInference(model_path=MODEL_PATH)
inference.load_model()
print("[Python API] Model ready. Listening on :5000")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart file upload (field name: 'file').
    Returns JSON with detection results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    uploaded = request.files["file"]
    suffix = Path(uploaded.filename).suffix or ".mp4"

    # Save to a temp file so run_yolo can read it from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = inference.predict_video(tmp_path, frames_to_process=10)
    finally:
        os.unlink(tmp_path)   # clean up immediately

    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify({
        "prediction": result["prediction"].upper(),   # "FAKE" | "REAL"
        "status":     result.get("status", ""),
        "fake_ratio": round(result["fake_ratio"], 1),
        "real_ratio": round(result["real_ratio"], 1),
        "confidence": round(result["confidence"] * 100, 2),
        "fake_frames": result["fake_frames"],
        "real_frames": result["real_frames"],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
