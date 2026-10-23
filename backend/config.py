"""Paths and runtime settings.

Folder layout (see README.md):

    project/
      models/      <- COPY the contents of  crowdrisk_outputs/models/  HERE   (trained files written by notebooks 01-04)
      outputs/     <- generated at run time (annotated media, JSON, CSV, reports, charts)  - safe to delete
      backend/     <- this folder
      frontend/
      notebooks/   <- the four training notebooks

Everything can be overridden with environment variables (useful when the models live elsewhere):
    CROWDRISK_MODELS_DIR   folder that contains crowd_count_model.pth, flow_selection.json, abnormal_motion.json, risk_model.pkl ...
    CROWDRISK_OUTPUT_DIR   folder for generated results
    CROWDRISK_DEVICE       auto | cpu          (notebooks: USE_GPU = True -> "auto", False -> "cpu")
    CROWDRISK_COUNT_EVERY  run the crowd-count model every N analysis steps (notebook 04: COUNT_EVERY = 5)
    CROWDRISK_MAX_SECONDS  optional: only analyse the first N seconds of a video

Server / deployment variables (read in app.py and gunicorn.conf.py):
    PORT                   port to listen on (hosting platforms set it; default 5000)           HOST  bind address for `python app.py` (default 0.0.0.0)
    CORS_ORIGINS           allowed browser origins, comma separated, e.g. https://my-app.vercel.app (regex allowed); unset or * = any origin (local development)
    MAX_UPLOAD_MB          maximum upload size (default 2048)
    WEB_THREADS            gunicorn threads (default 8; keep exactly ONE worker process, see gunicorn.conf.py)
    GUNICORN_TIMEOUT       seconds before gunicorn kills a silent worker (default 900; a video analysis is one long request)
"""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

MODELS_DIR = Path(os.environ.get("CROWDRISK_MODELS_DIR", PROJECT_ROOT / "models")).resolve()
OUTPUT_DIR = Path(os.environ.get("CROWDRISK_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()

IMAGES_DIR = OUTPUT_DIR / "images"
VIDEOS_DIR = OUTPUT_DIR / "videos"
JSON_DIR = OUTPUT_DIR / "json"
CSV_DIR = OUTPUT_DIR / "csv"
REPORTS_DIR = OUTPUT_DIR / "reports"
CHARTS_DIR = OUTPUT_DIR / "charts"
PREVIEW_DIR = OUTPUT_DIR / "previews"

for _d in (OUTPUT_DIR, IMAGES_DIR, VIDEOS_DIR, JSON_DIR, CSV_DIR, REPORTS_DIR, CHARTS_DIR, PREVIEW_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- model files written by the notebooks (name -> (required?, written by, description)) ---------------------------------
MODEL_FILES = {
    "crowd_count_model.pth": (True,  "01_Crowd_Count_Model",            "crowd-count network (CrowdPointNet) + inference settings"),
    "flow_selection.json":   (True,  "02_Motion_Model",                 "which optical-flow estimator was selected + working width"),
    "abnormal_motion.json":  (True,  "03_Abnormal_Motion_Detection",    "feature config, change model, abnormality detector"),
    "risk_model.pkl":        (True,  "04_Risk_Analysis_and_Video_Inference", "LightGBM risk + motion-state models, alarm rule"),
    "motion_model.pth":      (False, "02_Motion_Model",                 "MotionNetV2 weights - only needed if flow_selection.json says 'motionnet_v2'"),
    "feature_config.json":   (False, "03_Abnormal_Motion_Detection",    "copy of the feature config (also embedded in abnormal_motion.json)"),
    "label_rules.json":      (False, "03_Abnormal_Motion_Detection",    "training label rules (not used at inference)"),
    "crowd_count_info.json": (False, "01_Crowd_Count_Model",            "selected count variant + test metrics (shown in the UI if present)"),
    "motion_model_last.pth": (False, "02_Motion_Model",                 "training resume file (not used at inference)"),
}

DEVICE_PREF = os.environ.get("CROWDRISK_DEVICE", "auto")                  # "auto" or "cpu"
COUNT_EVERY = int(os.environ.get("CROWDRISK_COUNT_EVERY", "5"))            # notebook 04 cell 2: COUNT_EVERY = 5
_ms = os.environ.get("CROWDRISK_MAX_SECONDS")
MAX_SECONDS = float(_ms) if _ms else None
