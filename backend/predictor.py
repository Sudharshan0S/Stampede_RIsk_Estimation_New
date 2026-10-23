"""Orchestration: image analysis, video analysis, live progress, output files.

The ANALYSIS itself is the notebooks' code (crowd_count.py, motion.py, features.py, risk_model.py, pipeline.py - all verbatim copies).
This file only
   * feeds frames to them exactly like notebook 04 cell 48 `run_video` does (frame stride from the FPS assumption, resize to the
     working width with INTER_AREA, BGR->RGB, crowd-count model every COUNT_EVERY steps, HeadTracker, events, annotated video),
   * reports progress to the frontend, writes JSON / CSV / TXT / PDF / PNG artefacts and builds the API response.
"""
import os
import re
import json
import time
import uuid
import threading
import traceback
from datetime import datetime, timezone
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np

from config import OUTPUT_DIR, VIDEOS_DIR, IMAGES_DIR, CHARTS_DIR, REPORTS_DIR, JSON_DIR, CSV_DIR, COUNT_EVERY, MAX_SECONDS
from common import clean_for_json
from model_loader import get_models
from motion import work_size
from features import HeadTracker, RISK_LEVELS, crowd_regime
from risk_model import SIGNALS, ABN_TAGS
from pipeline import RiskStreamer, group_events
from visualization import (draw_overlay_v2, draw_flow_arrows, finalize_video, render_timeline_figure, render_diagnostics,
                           draw_predictions, density_overlay, render_panel)
from utils import to_output_url, output_path_from_url, save_json, save_csv, write_text_report
import reporting

# ============================================================
# LIVE ANALYSIS PROGRESS  (same contract as before: GET /progress?since=<seq>)
# ============================================================
STAGE_LABELS = {
    "IDLE": "Waiting for analysis", "PREPARING": "Preparing Workspace...", "CONNECTING": "Connecting Backend...", "UPLOADING": "Uploading Media...",
    "EXTRACTING": "Extracting Frames...", "CROWD_COUNTING": "Running Crowd Counting...", "MOTION_ANALYSIS": "Running Motion Analysis...",
    "RISK_ESTIMATION": "Estimating Risk...", "GENERATING_VIDEO": "Generating Annotated Video...", "GENERATING_IMAGE": "Generating Annotated Image...",
    "EXPORTING": "Generating Reports...", "COMPLETED": "Completed Successfully", "FAILED": "Backend Error", "CANCELLED": "Cancelled by User",
}
CURRENT_PROGRESS = {"running": False, "completed": False, "cancelled": False, "current_stage": "IDLE", "stage_label": STAGE_LABELS["IDLE"], "progress": 0,
                    "processed_frames": 0, "total_frames": 0, "message": STAGE_LABELS["IDLE"], "started_at": None, "elapsed_seconds": 0.0, "eta_seconds": None,
                    "frames_remaining": 0, "fps_current": 0.0, "fps_average": 0.0}
PROGRESS_LOG = []
_LOG_SEQ = 0
_TIMING = {"start": None, "last_frame_count": 0, "last_frame_time": None, "fps_window": deque(maxlen=20)}
CANCEL_REQUESTED = {"flag": False}
_ANALYSIS_LOCK = threading.Lock()          # models + progress are process-wide: one analysis at a time


class Busy(RuntimeError):
    pass


def _artifact_paths_from_result(result, downloads):
    paths = []
    for key in ("output_image", "processed_video"):
        if result.get(key): paths.append(Path(result[key]))
    for key, url in downloads.items():
        if key == "charts":
            for u in (url or {}).values():
                p = output_path_from_url(u)
                if p: paths.append(p)
        else:
            p = output_path_from_url(url)
            if p: paths.append(p)
    return paths


# ============================================================
# ARTIFACT TRACKING + SAFE REPLACEMENT OF THE PREVIOUS RESULT
# ============================================================
# Only the latest SUCCESSFUL run is kept on disk. The record of what belongs to it lives in OUTPUT_DIR/.run_manifest.json (persistent, so cleanup
# also works after a server restart). Life cycle of a run (everything below happens while holding _ANALYSIS_LOCK):
#   _begin_run   -> manifest.pending = this run        (so a crash can be recovered on the next start)
#   ... analysis writes its files; every file name contains the run id (32 hex chars) ...
#   _commit_run  -> 1. verify EVERY new artifact (exists, non-empty, readable, inside the generated-output folders)   [raises -> old result untouched]
#                   2. atomically write manifest.committed = new run                                                  [new result is now durable]
#                   3. only then delete the previous run's files (best effort; a failed deletion is retried later, it never fails the run)
#   _discard_run -> failed / cancelled / report error: delete only THIS run's own files, previous result stays.
#   recover_outputs (server start) -> discard a pending run left by a crash, retry deletions that failed earlier.
MANIFEST_PATH = OUTPUT_DIR / ".run_manifest.json"       # dot-file: not served by /outputs (see app.py)
_MANIFEST_VERSION = 1
_GENERATED_DIRS = tuple(d.resolve() for d in (IMAGES_DIR, VIDEOS_DIR, JSON_DIR, CSV_DIR, REPORTS_DIR, CHARTS_DIR))   # previews/ and everything else is out of scope
_RUN_ID_RE = re.compile(r"[0-9a-f]{32}")
_REQUIRED_DOWNLOADS = {"image": ("annotated_image", "json", "csv", "report", "report_pdf"), "video": ("annotated_video", "json", "csv", "report", "report_pdf")}


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _scoped_file(path, run_id=None):
    """Resolve `path` and return it only if it is a regular file inside one of the generated-output folders whose name carries a run id
    (and, if given, THIS run id). Anything else - models, source files, notes dropped into outputs/, '../' tricks, symlinks - returns None."""
    try:
        p = Path(path)
        if p.is_symlink(): return None
        p = p.resolve()
        if not any(d == p.parent or d in p.parents for d in _GENERATED_DIRS): return None
        m = _RUN_ID_RE.search(p.name)
        if not m or (run_id is not None and run_id not in p.name): return None
        return p
    except (OSError, ValueError):
        return None


def _rel_to_output(path):
    try:
        return Path(path).resolve().relative_to(OUTPUT_DIR.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _empty_manifest():
    return {"version": _MANIFEST_VERSION, "committed": None, "pending": None, "stale": []}


def _read_manifest():
    """Never raises. Missing -> empty. Corrupt / unreadable -> moved aside (.corrupt) and treated as empty: with no trustworthy record NOTHING is deleted."""
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f: data = json.load(f)
        c = data.get("committed")
        ok = isinstance(data, dict) and (c is None or (isinstance(c, dict) and isinstance(c.get("run_id"), str) and isinstance(c.get("files"), list)))
        ok = ok and isinstance(data.get("stale", []), list) and (data.get("pending") is None or isinstance(data.get("pending"), dict))
        if not ok: raise ValueError("unexpected manifest structure")
        data.setdefault("stale", []); data.setdefault("pending", None); data.setdefault("committed", None); data.setdefault("version", _MANIFEST_VERSION)
        return data
    except FileNotFoundError:
        return _empty_manifest()
    except Exception as e:
        print(f"[artifacts] run manifest unreadable ({type(e).__name__}: {e}) - ignoring it; no previous files will be deleted this time.")
        try: os.replace(MANIFEST_PATH, MANIFEST_PATH.with_name(MANIFEST_PATH.name + ".corrupt"))
        except OSError: pass
        return _empty_manifest()


def _write_manifest(data):
    """Atomic: write a temp file in the same folder, fsync, os.replace. Readers see either the old or the new complete file."""
    tmp = MANIFEST_PATH.with_name(f"{MANIFEST_PATH.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, MANIFEST_PATH)
    finally:
        try: tmp.unlink()
        except OSError: pass


def _delete_files(rel_paths, protect=()):
    """Delete recorded files, but only inside the generated-output scope and never anything in `protect`. Returns the entries that could NOT be deleted."""
    protect = {str(Path(p).resolve()) for p in protect}; failed = []
    for rel in rel_paths:
        if not isinstance(rel, str): continue
        p = _scoped_file(OUTPUT_DIR / rel)
        if p is None or str(p) in protect:
            continue                                                    # outside scope / belongs to the current run -> never touched
        try:
            p.unlink()
        except FileNotFoundError:
            pass                                                        # already gone
        except OSError as e:
            print(f"[artifacts] could not delete {rel}: {e} (will retry later)"); failed.append(rel)
    return failed


def _begin_run(run_id):
    """Record the run as pending so a crash mid-analysis can be cleaned up on the next start. A pending run left over from an earlier crash is discarded first
    (we hold the analysis lock, so it cannot still be running). Never raises."""
    try:
        m = _read_manifest(); old = m.get("pending")
        if old and old.get("run_id") and old.get("run_id") != run_id: _discard_run(old["run_id"], _manifest=m)
        m["pending"] = {"run_id": run_id, "started_at": _utc_now()}; _write_manifest(m)
    except Exception as e:
        print(f"[artifacts] could not record the pending run: {e}")


def _discard_run(run_id, _manifest=None):
    """Failed / cancelled run: remove only the files whose name carries THIS run id (annotated media, raw video, JSON, CSV, reports, charts) and clear
    the pending record. The previously committed result is never touched. Never raises."""
    removed = 0
    try:
        if not run_id or not _RUN_ID_RE.fullmatch(run_id): return 0
        for d in _GENERATED_DIRS:
            if not d.is_dir(): continue
            for f in d.iterdir():
                p = _scoped_file(f, run_id)
                if p is None: continue
                try: p.unlink(); removed += 1
                except OSError as e: print(f"[artifacts] could not remove partial file {p.name}: {e}")
        m = _manifest if _manifest is not None else _read_manifest()
        if (m.get("pending") or {}).get("run_id") == run_id:
            m["pending"] = None; _write_manifest(m)
    except Exception as e:
        print(f"[artifacts] discard of run {run_id} incomplete: {e}")
    return removed


def _verify_artifacts(result, downloads, media_type):
    """Make sure the new result is COMPLETE and readable before anything old is deleted. Raises RuntimeError naming the first problem."""
    paths = _artifact_paths_from_result(result, downloads)
    for key in _REQUIRED_DOWNLOADS[media_type]:
        if not downloads.get(key): raise RuntimeError(f"incomplete result: '{key}' was not generated")
    seen = {}
    for p in paths:
        sp = _scoped_file(p)
        if sp is None: raise RuntimeError(f"artifact outside the generated-output folders or without a run id: {p}")
        if not sp.is_file() or sp.stat().st_size == 0: raise RuntimeError(f"artifact missing or empty: {sp.name}")
        seen[str(sp)] = sp
    primary = Path(result["output_image" if media_type == "image" else "processed_video"]).resolve()
    if media_type == "image":
        if cv2.imread(str(primary)) is None: raise RuntimeError("annotated image is not readable")
    else:
        cap = cv2.VideoCapture(str(primary)); ok = cap.isOpened() and cap.read()[0]; cap.release()
        if not ok: raise RuntimeError("annotated video is not readable")
    try:
        with open(output_path_from_url(downloads["json"]), "r", encoding="utf-8") as f: json.load(f)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"JSON output is not readable: {e}")
    try:
        with open(output_path_from_url(downloads["report_pdf"]), "rb") as f: head = f.read(5)
    except OSError as e:
        raise RuntimeError(f"PDF report is not readable: {e}")
    if head != b"%PDF-": raise RuntimeError("PDF report is not a valid PDF")
    return list(seen.values())


def _commit_run(run_id, media_type, result, downloads):
    """Verify the new run, persist it as the current result, THEN remove the previous one. Raises only if verification fails (nothing old was touched yet).
    A problem while writing the manifest or deleting old files is logged and never turns a successful prediction into a failure."""
    files = _verify_artifacts(result, downloads, media_type)
    rels = sorted({r for r in (_rel_to_output(p) for p in files) if r})
    m = _read_manifest(); old = m.get("committed") or {}; stale = [x for x in m.get("stale", []) if isinstance(x, str)]
    new_manifest = {"version": _MANIFEST_VERSION, "pending": None, "stale": stale,
                    "committed": {"run_id": run_id, "media_type": media_type, "completed_at": _utc_now(), "files": rels}}
    try:
        _write_manifest(new_manifest)
    except Exception as e:
        print(f"[artifacts] could not persist the run manifest ({e}); previous files are kept (nothing is deleted).")
        return
    try:
        failed = _delete_files(list(old.get("files", [])) + stale, protect=files)
        if failed != stale:
            new_manifest["stale"] = failed; _write_manifest(new_manifest)
    except Exception as e:
        print(f"[artifacts] cleanup of the previous result incomplete: {e}")


def recover_outputs():
    """Call once at server start (app.py): discard a run that crashed mid-analysis and retry deletions that failed earlier. Never raises."""
    if not _ANALYSIS_LOCK.acquire(blocking=False): return
    try:
        m = _read_manifest(); pend = m.get("pending")
        if pend and pend.get("run_id"):
            n = _discard_run(pend["run_id"], _manifest=m); print(f"[artifacts] recovered from an interrupted run: removed {n} partial file(s).")
        stale = [x for x in m.get("stale", []) if isinstance(x, str)]
        if stale:
            keep = [Path(OUTPUT_DIR / r) for r in (m.get("committed") or {}).get("files", [])]
            failed = _delete_files(stale, protect=keep)
            if failed != stale: m["stale"] = failed; _write_manifest(m)
    except Exception as e:
        print(f"[artifacts] recovery skipped: {e}")
    finally:
        _ANALYSIS_LOCK.release()


def _push_log():
    global _LOG_SEQ
    _LOG_SEQ += 1
    entry = CURRENT_PROGRESS.copy(); entry["seq"] = _LOG_SEQ; entry["time"] = time.strftime("%H:%M:%S"); PROGRESS_LOG.append(entry)
    if len(PROGRESS_LOG) > 8000: del PROGRESS_LOG[: len(PROGRESS_LOG) - 8000]


def reset_progress():
    global _LOG_SEQ
    _LOG_SEQ = 0; PROGRESS_LOG.clear(); CANCEL_REQUESTED["flag"] = False; now = time.time()
    _TIMING.update({"start": now, "last_frame_count": 0, "last_frame_time": now, "fps_window": deque(maxlen=20)})
    CURRENT_PROGRESS.update({"running": True, "completed": False, "cancelled": False, "current_stage": "PREPARING", "stage_label": STAGE_LABELS["PREPARING"], "progress": 0,
                             "processed_frames": 0, "total_frames": 0, "message": STAGE_LABELS["PREPARING"], "started_at": now, "elapsed_seconds": 0.0, "eta_seconds": None,
                             "frames_remaining": 0, "fps_current": 0.0, "fps_average": 0.0})
    _push_log()


def update_progress(stage, progress, message=None, processed_frames=None, total_frames=None):
    now = time.time()
    CURRENT_PROGRESS["current_stage"] = stage; CURRENT_PROGRESS["stage_label"] = STAGE_LABELS.get(stage, stage.replace("_", " ").title())
    CURRENT_PROGRESS["progress"] = int(progress); CURRENT_PROGRESS["message"] = message if message is not None else CURRENT_PROGRESS["stage_label"]
    if total_frames is not None: CURRENT_PROGRESS["total_frames"] = int(total_frames)
    if processed_frames is not None:
        processed_frames = int(processed_frames); d_f = processed_frames - _TIMING["last_frame_count"]; d_t = (now - _TIMING["last_frame_time"]) if _TIMING["last_frame_time"] else 0
        if d_f > 0 and d_t > 0: _TIMING["fps_window"].append(d_f / d_t)
        _TIMING["last_frame_count"] = processed_frames; _TIMING["last_frame_time"] = now; CURRENT_PROGRESS["processed_frames"] = processed_frames
    elapsed = (now - _TIMING["start"]) if _TIMING["start"] else 0.0; CURRENT_PROGRESS["elapsed_seconds"] = round(elapsed, 2)
    fps_cur = float(np.mean(_TIMING["fps_window"])) if _TIMING["fps_window"] else 0.0
    fps_avg = CURRENT_PROGRESS["processed_frames"] / elapsed if elapsed > 0 and CURRENT_PROGRESS["processed_frames"] else 0.0
    CURRENT_PROGRESS["fps_current"] = round(fps_cur, 2); CURRENT_PROGRESS["fps_average"] = round(fps_avg, 2)
    remaining = max(0, CURRENT_PROGRESS["total_frames"] - CURRENT_PROGRESS["processed_frames"]); CURRENT_PROGRESS["frames_remaining"] = remaining
    CURRENT_PROGRESS["eta_seconds"] = round(remaining / fps_avg, 2) if fps_avg > 0 and CURRENT_PROGRESS["total_frames"] else None
    _push_log()


def finish_progress():
    CURRENT_PROGRESS["processed_frames"] = CURRENT_PROGRESS["total_frames"]
    CURRENT_PROGRESS.update({"running": False, "completed": True, "cancelled": False, "current_stage": "COMPLETED", "stage_label": STAGE_LABELS["COMPLETED"], "progress": 100,
                             "message": "Completed Successfully", "eta_seconds": 0, "frames_remaining": 0}); _push_log()


def fail_progress(message):
    CURRENT_PROGRESS.update({"running": False, "completed": False, "cancelled": False, "current_stage": "FAILED", "stage_label": STAGE_LABELS["FAILED"], "message": str(message)}); _push_log()


def cancel_progress():
    CURRENT_PROGRESS.update({"running": False, "completed": False, "cancelled": True, "current_stage": "CANCELLED", "stage_label": STAGE_LABELS["CANCELLED"],
                             "message": "Analysis cancelled by user"}); _push_log()


def request_cancel(): CANCEL_REQUESTED["flag"] = True
def is_cancel_requested(): return CANCEL_REQUESTED["flag"]


def get_progress(since=None):
    try: since = int(since) if since is not None else None
    except (TypeError, ValueError): since = None
    new = [e for e in PROGRESS_LOG if e["seq"] > since] if since is not None else (PROGRESS_LOG[-1:] if PROGRESS_LOG else [])
    return {"current": CURRENT_PROGRESS.copy(), "log": new, "latest_seq": PROGRESS_LOG[-1]["seq"] if PROGRESS_LOG else 0}


# ============================================================
# helpers
# ============================================================
def _fin(x):
    try: return x is not None and np.isfinite(x)
    except TypeError: return False


def _mean(v):
    v = [float(x) for x in v if _fin(x)]; return float(np.mean(v)) if v else None


def _max(v):
    v = [float(x) for x in v if _fin(x)]; return float(np.max(v)) if v else None


def model_summary(M):
    return M["info"]


# timeline fields exported to the frontend / JSON (the full ~200-column table is written to the CSV)
TIMELINE_KEYS = (["time_s", "start_s", "count", "density", "occupancy", "occupied_area_ratio", "packed_density", "density_concentration", "mean_speed", "p95_speed", "max_speed",
                  "mean_abs_acceleration", "direction_change_rate", "direction_coherence", "direction_variance", "counter_flow_ratio", "compression_score", "scattering_score",
                  "convergence", "speed_change_score", "acceleration_change_score", "direction_change_score", "crowd_count_change_score", "abnormality_score", "abnormality_level",
                  "n_active_signals", "n_active_groups", "sustained_high_speed", "risk_score_raw", "risk_score", "risk_change", "risk_level", "risk_level_idx", "p_high",
                  "motion_state", "motion_state_model", "state_confidence", "motion_level", "motion", "direction", "acceleration_level", "scattering_level", "counter_flow_level",
                  "density_level", "occupancy_level", "alarm_raw", "abnormal_online", "tags", "evidence", "reasons"]
                 + [f"E_{s}" for s in SIGNALS] + [f"active_{s}" for s in SIGNALS])


def _timeline_rows(rows, events):
    out = []
    for i, r in enumerate(rows):
        d = {k: r.get(k) for k in TIMELINE_KEYS if k in r}; d["window_index"] = i
        d["abnormal_event"] = 0
        for n, e in enumerate(events, 1):
            if e["start_s"] + 1e-9 <= r["time_s"] <= e["end_s"] + 1e-9: d["abnormal_event"] = n
        out.append(d)
    return clean_for_json(out)


def _risk_trend(scores):
    if len(scores) < 6: return "Stable"
    k = max(2, len(scores) // 3); a, b = float(np.mean(scores[:k])), float(np.mean(scores[-k:]))
    return "Increasing" if b - a > 5 else "Decreasing" if a - b > 5 else "Stable"


def _events_for_json(events):
    return clean_for_json(events)


# ============================================================
# IMAGE
# ============================================================
def predict_image(image_bgr, area=100.0):
    """Single image: crowd count + head points + density map from the crowd-count model (notebook 01).
    A single image has NO motion, so the motion-risk model (notebooks 03/04) cannot be applied; the static crowd regime
    (sparse / medium / dense / extreme, thresholds fitted on TRAIN scenes in notebook 03) is reported instead."""
    if not _ANALYSIS_LOCK.acquire(blocking=False): raise Busy("Another analysis is already running. Wait for it to finish (or cancel it).")
    t_start = time.perf_counter()
    reset_progress()
    run_id = uuid.uuid4().hex
    try:
        _begin_run(run_id)
        update_progress("UPLOADING", 10, STAGE_LABELS["UPLOADING"], 0, 1)
        M = get_models(); cm, cfg = M["count_model"], M["feature_config"]
        if not isinstance(image_bgr, np.ndarray): image_bgr = cv2.imread(str(image_bgr))
        if image_bgr is None: raise FileNotFoundError("Could not load the image.")
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
        update_progress("CROWD_COUNTING", 35, STAGE_LABELS["CROWD_COUNTING"])
        t0 = time.perf_counter(); r = cm.predict(rgb, frame_id=0); inference_ms = (time.perf_counter() - t0) * 1000.0
        count = int(r["count"]); pts, unc = r["head_points"], r["uncertain_points"]; dm, cell = r["density_map"], r["density_cell_xy"]
        ws = work_size(W, H, cfg.work_width); density = count / (ws[0] * ws[1]) * 1e4               # people / 10k work-resolution px (feature definition of notebook 03)
        regime = crowd_regime(density, cfg); occ = float((dm > cfg.density_region_thr).mean()) if dm is not None and dm.size else None
        dsum = r["uncertainty"]["count_density_sum"]; dsum = float(dsum) if _fin(dsum) else None
        update_progress("GENERATING_IMAGE", 75, STAGE_LABELS["GENERATING_IMAGE"])
        final = dict(accepted=pts, uncertain=unc, acc_scores=r["confidence"], unc_scores=r["uncertain_confidence"])           # original-image pixel coordinates
        scale = min(1.0, 2000.0 / max(H, W))                                                                                   # keep the saved JPEGs a sensible size
        base = cv2.resize(rgb, (int(round(W * scale)), int(round(H * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb.copy()
        ann = draw_predictions(base.copy(), final, scale, r=max(3, int(round(0.005 * max(base.shape[:2])))))                   # notebook 01 marker colours: green accepted, yellow uncertain
        cv2.putText(ann, f"COUNT {count}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * max(0.6, base.shape[1] / 1200), (255, 255, 255), 2, cv2.LINE_AA)
        pred_id = run_id
        ann_path = IMAGES_DIR / f"annotated_{pred_id}.jpg"; cv2.imwrite(str(ann_path), cv2.cvtColor(ann, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        den_path = None
        if dm is not None:
            den = density_overlay(base, dm, cell * scale); den_path = IMAGES_DIR / f"density_{pred_id}.jpg"; cv2.imwrite(str(den_path), cv2.cvtColor(den, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        stats = [f"count {count}", f"uncertain {len(unc)}", f"density-map sum {dsum:.1f}" if dsum is not None else "no density map", f"regime {regime}"]
        panel = render_panel(rgb, final, density=dm, cell_xy=cell, title="crowd count", stats=stats)
        panel_path = IMAGES_DIR / f"panel_{pred_id}.jpg"; cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

        result = dict(success=True, media_type="image", area=float(area), image_width=W, image_height=H, crowd_count=count, count_interval=[count, count + len(unc)], n_uncertain=int(len(unc)),
                      density_map_sum=dsum, count_calibrated=r.get("count_calibrated"), uncertainty=r["uncertainty"], density=float(density), crowd_regime=regime, regime_edges=list(cfg.regime_edges),
                      occupied_area_ratio=occ, people_per_m2=float(count / area) if area and area > 0 else None, risk_level="N/A", risk_score=None, risk_available=False,
                      risk_reason="A single image contains no motion, so the motion-risk model (video only) is not applied. Crowd regime: "
                                  f"{regime.upper()} ({density:.2f} people per 10k px at the {cfg.work_width}px working width).",
                      head_points=np.round(pts[:5000], 1).tolist(), uncertain_points=np.round(unc[:5000], 1).tolist(),            # lists capped for the JSON; counts are exact
                      points_truncated=bool(len(pts) > 5000 or len(unc) > 5000), output_image=str(ann_path), models=model_summary(M),
                      inference_ms=round(inference_ms, 1), processing_time=round(time.perf_counter() - t_start, 2))
        update_progress("RISK_ESTIMATION", 85, f"Crowd {count} | uncertain {len(unc)} | regime {regime} | Inference {inference_ms:.0f}ms")
        update_progress("EXPORTING", 92, "Generating Reports...")
        clean = clean_for_json(result)
        json_path = save_json(clean, f"json/image_{pred_id}.json")
        flat = {k: v for k, v in clean.items() if not isinstance(v, (dict, list))}
        csv_path = save_csv([flat], f"csv/image_{pred_id}.csv")
        heads = [("Estimated crowd count (accepted heads)", count), ("Count interval (accepted .. + uncertain)", f"{count} .. {count + len(unc)}"),
                 ("Density-map sum (support only)", f"{dsum:.1f}" if dsum is not None else "n/a"), ("Uncertainty level (heuristic)", r["uncertainty"]["level"]),
                 ("Density (people / 10k px, work res)", f"{density:.2f}"), ("Crowd regime", regime.upper()), ("People per m2 (count / area)", f"{count / area:.3f}" if area else "n/a"),
                 ("Occupied area (density-map cells)", f"{100 * occ:.0f}%" if occ is not None else "n/a"), ("Image size", f"{W} x {H}"), ("Analysis mode", "Single image (no motion risk)")]
        secs = [("Notes", [result["risk_reason"], r["uncertainty"]["note"]])]
        txt = write_text_report(reporting.build_report_lines("STAMPEDE RISK ANALYSIS - IMAGE REPORT", heads, secs), f"reports/image_{pred_id}_report.txt")
        pdf = reporting.generate_pdf_report("Crowd Analysis - Image Report", heads, secs, REPORTS_DIR / f"image_{pred_id}_report.pdf")
        update_progress("EXPORTING", 96, "Generating Charts...")
        charts = reporting.generate_image_charts(clean, pred_id)
        downloads = {"annotated_image": to_output_url(ann_path), "density_image": to_output_url(den_path), "panel_image": to_output_url(panel_path), "json": to_output_url(json_path),
                     "csv": to_output_url(csv_path), "report": to_output_url(txt), "report_pdf": to_output_url(pdf), "charts": {k: to_output_url(v) for k, v in charts.items()}}
        result["output_image_url"] = downloads["annotated_image"]; result["downloads"] = downloads
        response = clean_for_json(result)                                   # build the response first ...
        _commit_run(run_id, "image", result, downloads)                     # ... verify + persist the new result, then (and only then) delete the previous one
        finish_progress()
        return response
    except Exception as error:
        _discard_run(run_id)                                                # failed: remove only this run's partial files, the previous result stays
        fail_progress(error)
        return {"success": False, "error": str(error), "trace": traceback.format_exc()}
    finally:
        _ANALYSIS_LOCK.release()


# ============================================================
# VIDEO   (port of notebook 04 cell 48 `run_video`)
# ============================================================
def predict_video(video_path, area=100.0):
    if not _ANALYSIS_LOCK.acquire(blocking=False): raise Busy("Another analysis is already running. Wait for it to finish (or cancel it).")
    t_start = time.perf_counter(); cap = None; writer = None; raw_path = None
    reset_progress()
    run_id = uuid.uuid4().hex
    try:
        _begin_run(run_id)
        update_progress("CONNECTING", 3, STAGE_LABELS["CONNECTING"])
        M = get_models(); est, count_model, bundle = M["estimator"], M["count_model"], M["bundle"]
        update_progress("UPLOADING", 7, STAGE_LABELS["UPLOADING"])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise RuntimeError("Unable to open the video file (unsupported codec or corrupt file).")
        S_ = RiskStreamer(bundle, est, count_model, count_every=COUNT_EVERY); cfg = S_.cfg
        nat_fps = cap.get(cv2.CAP_PROP_FPS)
        if not nat_fps or not np.isfinite(nat_fps) or nat_fps <= 0: nat_fps = cfg.fps               # guard only; valid files behave exactly like the notebook
        stride = max(1, int(round(nat_fps / cfg.fps))); eff_fps = nat_fps / stride; S_.eff_fps = eff_fps
        W0, H0 = int(cap.get(3)), int(cap.get(4)); ws = work_size(W0, H0, cfg.work_width); total_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        update_progress("EXTRACTING", 12, f"{W0}x{H0} @ {nat_fps:.1f} fps -> analysis {eff_fps:.1f} steps/s at {ws[0]}x{ws[1]}", 0, total_native)
        pred_id = run_id; raw_path = VIDEOS_DIR / f"_raw_{pred_id}.mp4"; final_path = VIDEOS_DIR / f"annotated_{pred_id}.mp4"
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), eff_fps, (W0, H0))
        if not writer.isOpened(): raise RuntimeError("Unable to create the annotated output video.")
        tracker = HeadTracker(max_dist=0.04 * ws[0])
        hud = dict(risk_level="n/a", risk_score=0.0, risk_level_idx=0, abnormal=False, tags=[], motion_state="NORMAL", occupancy_level="n/a", motion_level="n/a", count=None)
        peak = dict(score=-1); t0 = time.time(); k = -1; n_step = 0; limit = int(MAX_SECONDS * nat_fps) if MAX_SECONDS else None; last_flow = None; cancelled = False
        while True:
            if is_cancel_requested(): cancelled = True; break
            ok, frame = cap.read(); k += 1
            if not ok or (limit and k >= limit): break
            if k % stride: continue
            rgb = cv2.cvtColor(cv2.resize(frame, ws, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            prev_cinfo = S_.cinfo; res = S_.push(rgb, k / nat_fps); counted = S_.cinfo is not prev_cinfo; vis = frame
            pct = 12 + int(76 * (k + 1) / max(total_native, 1)) if total_native else 50
            if res is not None:
                n_step += 1; last_flow = res["flow"]; ci = res["count_info"]
                if ci["points"] is not None and len(ci["points"]) and n_step % COUNT_EVERY == 0: tracker.update(ci["points"], res["flow"])
                if res["window"] is not None:
                    w = res["window"]
                    hud = dict(risk_level=w["risk_level"], risk_score=w["risk_score"], risk_level_idx=w["risk_level_idx"], abnormal=w["abnormal_online"], tags=w["tags"], motion_state=w["motion_state"],
                               occupancy_level=w["occupancy_level"], motion_level=w["motion_level"], count=w["count"] if _fin(w.get("count")) else None)
                    sc_ = 0.5 * w["risk_score"] + 50 * w["abnormality_score"]
                    if sc_ > peak["score"]:
                        peak = dict(score=sc_, rgb=rgb.copy(), flow=res["flow"].copy(), region=res["region"].copy(), tracks=None if res["tracks"] is None else np.array(res["tracks"]),
                                    count_info=dict(ci), window=dict(w))
                    update_progress("RISK_ESTIMATION", min(pct, 88), f"t={k / nat_fps:.1f}s | window {len(S_.rows)} | Crowd {hud['count'] if hud['count'] is None else int(hud['count'])} | "
                                    f"Risk {w['risk_level']} ({w['risk_score']:.0f}/100) | {w['motion_state']} | Abnormal {'YES' if w['abnormal_online'] else 'NO'}", k + 1, total_native)
                elif counted:
                    update_progress("CROWD_COUNTING", min(pct, 88), f"t={k / nat_fps:.1f}s | Crowd {ci['count']} people ({0 if ci['points'] is None else len(ci['points'])} heads)", k + 1, total_native)
                else:
                    update_progress("MOTION_ANALYSIS", min(pct, 88), f"t={k / nat_fps:.1f}s | optical flow + features (step {n_step})", k + 1, total_native)
                vis = frame.copy()
                if last_flow is not None:
                    arrows = draw_flow_arrows(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.resize(last_flow, (W0, H0)) * (W0 / ws[0]), None, step=max(12, W0 // 40)); vis = cv2.cvtColor(arrows, cv2.COLOR_RGB2BGR)
                pts = S_.cinfo["points"]
                if pts is not None and len(pts):
                    sx, sy = W0 / ws[0], H0 / ws[1]
                    for (x, y) in pts: cv2.circle(vis, (int(x * sx), int(y * sy)), 3, (60, 60, 255), -1)
            writer.write(draw_overlay_v2(vis, hud))
        cap.release(); cap = None; writer.release(); writer = None
        if cancelled:
            _discard_run(run_id); cancel_progress()                         # remove this run's partial files (incl. the raw video); the previous result stays
            return {"success": False, "cancelled": True, "error": "Analysis cancelled by user"}

        TL = S_.timeline()
        if len(TL) == 0:
            raw_path.unlink(missing_ok=True)
            need = cfg.window_len + 1
            raise RuntimeError(f"The video is too short: one temporal window needs {need} analysis steps (about {need / eff_fps:.1f} s at {eff_fps:.1f} steps/s); only {n_step} were available.")
        update_progress("GENERATING_VIDEO", 90, STAGE_LABELS["GENERATING_VIDEO"], total_native, total_native)
        EVENTS = group_events(TL, S_.rows, S_.det, S_.cm, S_.model.cfg, S_.fusion)
        TL["abnormal_event"] = 0
        for i, e in enumerate(EVENTS, 1): TL.loc[(TL.time_s >= e["start_s"] + 1e-9) & (TL.time_s <= e["end_s"] + 1e-9), "abnormal_event"] = i
        finalize_video(raw_path, final_path)

        # ---------------- aggregate statistics (computed from the window rows only) ----------------
        rows = S_.rows; sc = [float(r["risk_score"]) for r in rows]; lv = [int(r["risk_level_idx"]) for r in rows]; win_s = cfg.hop / eff_fps
        mstate = Counter(r["motion_state"] for r in rows); tagc = Counter(t for r in rows for t in r["tags"] if t != "NORMAL")
        pk_i = int(np.argmax(sc)); timeline = _timeline_rows(rows, EVENTS); events_json = _events_for_json(EVENTS)
        pred_summary = dict(video=Path(video_path).name, native_fps=nat_fps, analysis_fps=eff_fps, frame_stride=stride, analysis_steps=n_step, windows=len(rows), window_len=cfg.window_len,
                            hop=cfg.hop, seconds_processing=time.time() - t0, flow_estimator=M["info"]["flow_estimator"], fusion_mode=S_.fusion, used_ground_truth=False,
                            n_events=len(EVENTS), identity_tracking_ok=bool(tracker.identity_ok))
        result = dict(
            success=True, media_type="video", area=float(area), fps=nat_fps, analysis_fps=eff_fps, total_frames=total_native, frames_processed=n_step, windows=len(rows),
            window_seconds=cfg.window_len / eff_fps, duration_s=(k if k > 0 else n_step) / nat_fps, processing_time=round(time.perf_counter() - t_start, 2),
            average_crowd_count=_mean(r.get("count") for r in rows), maximum_crowd_count=_max(r.get("count") for r in rows),
            average_density=_mean(r.get("density") for r in rows), maximum_density=_max(r.get("density") for r in rows),
            average_people_per_m2=(_mean(r.get("count") for r in rows) or 0) / area if area else None, maximum_people_per_m2=(_max(r.get("count") for r in rows) or 0) / area if area else None,
            average_occupancy=_mean(r.get("occupancy") for r in rows), average_speed=_mean(r["mean_speed"] for r in rows), maximum_speed=_max(r["max_speed"] for r in rows),
            average_acceleration=_mean(r["mean_abs_acceleration"] for r in rows), maximum_acceleration=_max(r["mean_abs_acceleration"] for r in rows),
            average_direction_change_rate=_mean(r["direction_change_rate"] for r in rows), average_counter_flow_ratio=_mean(r["counter_flow_ratio"] for r in rows),
            average_compression=_mean(r["compression_score"] for r in rows), average_scattering=_mean(r["scattering_score"] for r in rows),
            average_abnormality=_mean(r["abnormality_score"] for r in rows), maximum_abnormality=_max(r["abnormality_score"] for r in rows),
            average_risk_score=float(np.mean(sc)), peak_risk_score=float(np.max(sc)), peak_risk_time_s=float(rows[pk_i]["time_s"]), current_risk_score=sc[-1],
            risk_score=float(np.max(sc)), risk_percent=float(np.max(sc)), risk_level=RISK_LEVELS[int(max(lv))], current_risk_level=RISK_LEVELS[lv[-1]], risk_trend=_risk_trend(sc),
            time_high_risk_s=float(sum(1 for x in lv if x >= 2) * win_s), time_abnormal_s=float(sum(1 for r in rows if r["abnormal_online"]) * win_s),
            windows_by_level={RISK_LEVELS[i]: int(sum(1 for x in lv if x == i)) for i in range(len(RISK_LEVELS))}, motion_state_counts=dict(mstate), tag_counts=dict(tagc),
            dominant_motion_state=mstate.most_common(1)[0][0], n_events=len(EVENTS), events=events_json, alarm=bool(EVENTS),
            risk_reason=(" \n ".join(f" • {e}" for e in (rows[pk_i]["evidence"] or [])) or "no strong abnormal-motion factor") if EVENTS else "No abnormal event detected.",
            average_inference_ms=None, summary=pred_summary, models=model_summary(M), timeline=timeline, risk_available=True,
            processed_video=str(final_path))

        # ---------------- artefacts ----------------
        update_progress("EXPORTING", 94, "Generating Reports...", total_native, total_native)
        json_path = save_json(result, f"json/video_{pred_id}.json")
        csv_path = save_csv(TL, f"csv/video_{pred_id}_risk_timeline.csv")                       # same table as notebook 04 `*_risk_timeline.csv`
        ev_rows = [{k2: (", ".join(f"{a}:{b}" for a, b in v.items()) if isinstance(v, dict) else ("; ".join(v) if isinstance(v, list) else v)) for k2, v in e.items() if k2 != "explanation"} for e in events_json]
        ev_csv = save_csv(ev_rows, f"csv/video_{pred_id}_events.csv") if ev_rows else None   # same layout as notebook 04 `detected_events.csv`
        ev_json = save_json(dict(summary=pred_summary, events=events_json), f"json/video_{pred_id}_events.json")
        update_progress("EXPORTING", 96, "Generating Charts...", total_native, total_native)
        chart_paths = reporting.generate_video_charts(timeline, events_json, pred_id)
        tl_png = CHARTS_DIR / f"timeline_chart_{pred_id}.png"
        try: render_timeline_figure(TL, EVENTS, f"{Path(video_path).name} (red = detected abnormal event)", str(tl_png)); chart_paths["timeline_chart"] = str(tl_png)
        except Exception as e: print("[charts] timeline figure skipped:", e)
        dg_png = CHARTS_DIR / f"diagnostics_chart_{pred_id}.png"
        try:
            if peak["score"] >= 0 and render_diagnostics(peak, SIGNALS, str(dg_png)): chart_paths["diagnostics_chart"] = str(dg_png)
        except Exception as e: print("[charts] diagnostics figure skipped:", e)
        heads = [("Video", Path(video_path).name), ("Duration analysed", f"{result['duration_s']:.1f} s"), ("Analysis rate", f"{eff_fps:.1f} steps/s (window {cfg.window_len} steps, hop {cfg.hop})"),
                 ("Windows analysed", len(rows)), ("Average / peak crowd count", f"{result['average_crowd_count']:.0f} / {result['maximum_crowd_count']:.0f}"),
                 ("Average / max speed (px per step)", f"{result['average_speed']:.2f} / {result['maximum_speed']:.2f}"),
                 ("Average / peak risk score", f"{result['average_risk_score']:.1f} / {result['peak_risk_score']:.1f} (0-100)"), ("Peak risk level", result["risk_level"]),
                 ("Risk trend", result["risk_trend"]), ("Time at HIGH+ risk", f"{result['time_high_risk_s']:.1f} s"), ("Abnormal events detected", len(EVENTS)),
                 ("Flow estimator", M["info"]["flow_estimator"]), ("Alarm rule", S_.fusion), ("Processing time", f"{result['processing_time']:.1f} s")]
        secs = [(f"Event {i}", e["explanation"].split("\n")) for i, e in enumerate(events_json, 1)] or [("Events", ["No abnormal event detected."])]
        txt = write_text_report(reporting.build_report_lines("STAMPEDE RISK ANALYSIS - VIDEO REPORT", heads, secs), f"reports/video_{pred_id}_report.txt")
        pdf = reporting.generate_pdf_report("Crowd Risk Analysis - Video Report", heads, secs, REPORTS_DIR / f"video_{pred_id}_report.pdf")
        downloads = {"annotated_video": to_output_url(final_path), "json": to_output_url(json_path), "csv": to_output_url(csv_path), "events_csv": to_output_url(ev_csv),
                     "events_json": to_output_url(ev_json), "report": to_output_url(txt), "report_pdf": to_output_url(pdf), "charts": {k2: to_output_url(v) for k2, v in chart_paths.items()}}
        result["output_video_url"] = downloads["annotated_video"]; result["downloads"] = downloads
        response = clean_for_json(result)                                   # build the response first ...
        _commit_run(run_id, "video", result, downloads)                     # ... verify + persist the new result, then (and only then) delete the previous one
        finish_progress()
        return response
    except Exception as error:
        if cap is not None: cap.release(); cap = None                       # release file handles before deleting (required on Windows)
        if writer is not None: writer.release(); writer = None
        _discard_run(run_id)                                                # failed: remove only this run's partial files, the previous result stays
        fail_progress(error)
        return {"success": False, "error": str(error), "trace": traceback.format_exc()}
    finally:
        if cap is not None: cap.release()
        if writer is not None: writer.release()
        _ANALYSIS_LOCK.release()
