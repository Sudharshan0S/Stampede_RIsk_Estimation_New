"""Loads every trained artefact exactly the way notebook 04 does (cells 32, 33 and 43):

    abnormal_motion.json -> FeatureConfig, ChangeModel, AbnormalityDetector
    flow_selection.json  -> selected estimator name + working width  (+ motion_model.pth when MotionNetV2 was selected)
    crowd_count_model.pth-> CountModel (CrowdPointNet + inference config baked into the checkpoint)
    risk_model.pkl       -> RiskModel + alarm-fusion rule

Loading is done once and cached.  Problems are reported in plain language (which file is missing and which notebook writes it).
"""
import json
import threading

from config import MODELS_DIR, MODEL_FILES, DEVICE_PREF
from common import pick_device
from crowd_count import CountModel, file_fingerprint
from features import FeatureConfig
from motion import load_selected_estimator
from risk_model import ChangeModel, AbnormalityDetector, load_risk_bundle

_LOCK = threading.Lock()
_MODELS = None
_ERROR = None


def models_status():
    """Which model files exist in MODELS_DIR (used by /models/status and by the error messages)."""
    files = []
    for name, (required, notebook, desc) in MODEL_FILES.items():
        p = MODELS_DIR / name
        files.append(dict(file=name, required=required, found=p.is_file(), size_mb=round(p.stat().st_size / 1e6, 3) if p.is_file() else None,
                          written_by=notebook, description=desc))
    return dict(models_dir=str(MODELS_DIR), files=files, ready=_MODELS is not None, error=_ERROR)


def _require(name):
    p = MODELS_DIR / name
    if not p.is_file():
        required, notebook, desc = MODEL_FILES.get(name, (True, "?", ""))
        raise FileNotFoundError(f"Model file not found: {p}\n-> copy '{name}' from <crowdrisk_outputs>/models/ into {MODELS_DIR} "
                                f"(written by notebook {notebook}: {desc}).")
    return str(p)


def _load():
    warnings = []
    ab = json.load(open(_require("abnormal_motion.json")))                                  # notebook 04 cell 32
    cfg = FeatureConfig.from_dict(ab["feature_config"])
    cm_pred = ChangeModel.from_dict(ab["change_model"])
    det = AbnormalityDetector.from_dict(ab["detector"], cm_pred)

    flow_sel = json.load(open(_require("flow_selection.json")))                              # notebook 04 cell 33
    selected_flow, width = flow_sel["selected"], int(flow_sel["work_width"])
    if int(cfg.work_width) != width:
        warnings.append(f"feature config work_width ({cfg.work_width}) differs from flow_selection.json ({width}); the feature config is used (as in the notebook).")

    device = pick_device(DEVICE_PREF)
    motion_path = str(MODELS_DIR / "motion_model.pth")
    if selected_flow == "motionnet_v2": motion_path = _require("motion_model.pth")
    estimator = load_selected_estimator(selected_flow, device, motion_path)

    count_path = _require("crowd_count_model.pth")
    count_model = CountModel(count_path, device=DEVICE_PREF)
    fp = file_fingerprint(count_path)
    if ab.get("count_model_fingerprint") and ab["count_model_fingerprint"] != fp:
        warnings.append("crowd_count_model.pth is not the file that notebook 03 used to build the risk features "
                        f"(fingerprint {fp} vs {ab['count_model_fingerprint']}). Re-run notebooks 03 and 04 or copy the matching files.")

    rb = load_risk_bundle(_require("risk_model.pkl"))                                          # notebook 04 cell 43
    risk_model, fusion_mode = rb["risk_model"], rb["fusion_mode"]
    bundle = dict(risk_model=risk_model, change_model=cm_pred, detector=det, feature_config=cfg.to_dict(), fusion_mode=fusion_mode,
                  config_name=rb.get("config_name"), meta=getattr(risk_model, "meta", {}))

    count_info = {}
    p = MODELS_DIR / "crowd_count_info.json"
    if p.is_file():
        try: count_info = json.load(open(p))
        except Exception: pass

    info = dict(device=str(device), count_model=count_model.meta, count_refinement=count_model.refine_on, flow_estimator=selected_flow,
                work_width=width, risk_algorithm=getattr(risk_model, "algorithm", None), risk_config=rb.get("config_name"), fusion_mode=fusion_mode,
                level_edges=list(risk_model.cfg.level_edges), window_len=cfg.window_len, hop=cfg.hop, fps_assumption=cfg.fps,
                n_risk_features=len(risk_model.features), count_test_metrics=count_info.get("test"), warnings=warnings)
    return dict(count_model=count_model, estimator=estimator, bundle=bundle, feature_config=cfg, flow_selection=flow_sel, device=device, info=info)


def get_models():
    """Cached loader. Raises a readable exception if anything is missing/invalid."""
    global _MODELS, _ERROR
    if _MODELS is not None: return _MODELS
    with _LOCK:
        if _MODELS is None:
            try:
                _MODELS = _load(); _ERROR = None
            except Exception as e:
                _ERROR = f"{type(e).__name__}: {e}"
                raise
    return _MODELS


def try_load():
    """Used at server start-up: never raises, records the error for /health and /models/status."""
    try:
        get_models(); return True
    except Exception as e:
        print("\n[model_loader] MODELS NOT READY ->", e, "\n"); return False
