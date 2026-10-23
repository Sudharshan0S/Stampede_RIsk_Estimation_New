"""Flask API for CrowdVision AI.

Endpoints (unchanged for the frontend):  GET /  /health  /progress  /outputs/<file>   POST /predict/image  /predict/video  /preview/video  /cancel
New:  GET /models/status  (which model files were found / missing and what was loaded)
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config import OUTPUT_DIR
from model_loader import try_load, models_status, get_models
from predictor import predict_image, predict_video, get_progress, request_cancel, recover_outputs, Busy
from visualization import remux_for_preview

app = Flask(__name__)


def _cors_origins():
    """CORS_ORIGINS = comma-separated list of allowed browser origins, e.g. 'https://my-app.vercel.app,http://localhost:5173'.
    Entries may be regular expressions (e.g. 'https://my-app-.*\\.vercel\\.app' for Vercel preview URLs). Unset or '*' -> any origin (local development)."""
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    if raw in ("", "*"): return "*"
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


# React dev server (5173) -> Flask (5000) locally; in production set CORS_ORIGINS to the deployed frontend origin.
CORS(app, origins=_cors_origins(), expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"])
app.config["MAX_CONTENT_LENGTH"] = int(float(os.environ.get("MAX_UPLOAD_MB", "2048")) * 1024 ** 2)      # default 2 GB uploads

ALLOWED_IMAGES = {"jpg", "jpeg", "png", "bmp"}
ALLOWED_VIDEOS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


@app.route("/")
def home():
    return jsonify({"project": "AI Based Stampede Risk Estimation", "status": "Running", "version": "3.0"})


@app.route("/health")
def health():
    st = models_status()
    return jsonify({"status": "healthy", "models_ready": st["ready"], "models_error": st["error"]})


@app.route("/models/status")
def models_status_route():
    st = models_status()
    if st["ready"]: st["info"] = get_models()["info"]
    return jsonify(st)


@app.route("/progress", methods=["GET"])
def analysis_progress():
    return jsonify(get_progress(request.args.get("since", default=None)))


@app.route("/cancel", methods=["POST"])
def cancel_analysis():
    request_cancel()
    return jsonify({"success": True, "message": "Cancellation requested."})


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    if any(part.startswith(".") for part in Path(filename).parts):      # e.g. .run_manifest.json is internal bookkeeping, never served
        return jsonify({"error": "Not found."}), 404
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return send_from_directory(str(OUTPUT_DIR), filename)               # conditional + Range support (needed for video seeking in browsers)


@app.route("/preview/video", methods=["POST"])
def preview_video():
    if "video" not in request.files: return jsonify({"success": False, "error": "No video uploaded."}), 400
    file = request.files["video"]
    if file.filename == "": return jsonify({"success": False, "error": "Empty filename."}), 400
    if not allowed_file(file.filename, ALLOWED_VIDEOS): return jsonify({"success": False, "error": "Unsupported video format."}), 400
    extension = file.filename.rsplit(".", 1)[1].lower(); temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp:
            file.save(temp.name); temp_path = temp.name
        return jsonify(remux_for_preview(temp_path, extension))
    except Exception as error:
        return jsonify({"success": False, "error": str(error)}), 500
    finally:
        if temp_path is not None: Path(temp_path).unlink(missing_ok=True)


@app.route("/predict/image", methods=["POST"])
def image_prediction():
    if "image" not in request.files: return jsonify({"error": "No image uploaded."}), 400
    file = request.files["image"]
    if file.filename == "": return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename, ALLOWED_IMAGES): return jsonify({"error": "Unsupported image format."}), 400
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None: return jsonify({"error": "Unable to decode image."}), 400
    area = request.form.get("area", default=100.0, type=float)
    try:
        return jsonify(predict_image(image, area))
    except Busy as e:
        return jsonify({"error": str(e)}), 409
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/predict/video", methods=["POST"])
def video_prediction():
    if "video" not in request.files: return jsonify({"error": "No video uploaded."}), 400
    file = request.files["video"]
    if file.filename == "": return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename, ALLOWED_VIDEOS): return jsonify({"error": "Unsupported video format."}), 400
    suffix = "." + file.filename.rsplit(".", 1)[1]; temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            file.save(temp.name); temp_path = temp.name
        area = request.form.get("area", default=100.0, type=float)
        return jsonify(predict_video(temp_path, area))
    except Busy as e:
        return jsonify({"error": str(e)}), 409
    except Exception as error:
        return jsonify({"error": str(error)}), 500
    finally:
        if temp_path is not None: Path(temp_path).unlink(missing_ok=True)


recover_outputs()   # clean up after a crash / restart: discard a half-written run, retry deletions that failed earlier
try_load()          # load models once at start-up; if a file is missing the server still starts and /models/status explains what to copy where

if __name__ == "__main__":
    # Local development server. debug=False / no reloader: the reloader would load the (large) models twice and restart in the middle of an analysis.
    # Production: gunicorn -c gunicorn.conf.py app:app   (see README "Deployment"). PORT is set by most hosting platforms.
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "5000")), debug=False, threaded=True)
