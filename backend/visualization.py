"""
Drawing code.

VERBATIM from the notebooks:
   notebook 01 cell 24 : _ring, _cross, density_overlay, put_lines, draw_predictions, render_panel   (crowd-count panels: green = accepted, yellow = uncertain)
   notebook 02/04      : flow_to_color, draw_points, draw_flow_arrows, draw_tracks, speed_heatmap, viz_density_overlay, show_grid
   notebook 04 cell 48 : RISK_BGR, draw_overlay_v2 (video HUD)          notebook 04 cell 52: plot_full_timeline          cell 54: fig_to_rgb
BACKEND-ONLY (bottom of the file): ffmpeg transcode for browsers, input-preview remux, render_diagnostics (the 10 diagnostics of notebook 04 cell 54 as one PNG).
"""
import os, shutil, subprocess, uuid
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")                      # headless server: never open a window
import matplotlib.pyplot as plt

from config import PREVIEW_DIR
from pipeline import TAG_PHRASE


# ---- notebook 01 cell 24: colours (RGB) --------------------
COL_ACC, COL_UNC, COL_FP, COL_GT = (0, 220, 0), (255, 215, 0), (255, 0, 0), (40, 110, 255)     # RGB
COL_REGION = (255, 140, 0)


# --- notebook 01, cell 24: _ring --------------------
def _ring(canvas, pts, color, r, th=1):
    for x, y in pts: cv2.circle(canvas, (int(round(x)), int(round(y))), r, color, th, cv2.LINE_AA)


# --- notebook 01, cell 24: _cross --------------------
def _cross(canvas, pts, color, r, th=1):
    for x, y in pts: cv2.drawMarker(canvas, (int(round(x)), int(round(y))), color, cv2.MARKER_TILTED_CROSS, 2 * r + 2, th, cv2.LINE_AA)


# --- notebook 01, cell 24: density_overlay --------------------
def density_overlay(img, den, cell_xy, alpha=0.55):
    """Heat-map of the density branch (people per cell) resized to the image. Returns RGB uint8."""
    H, W = img.shape[:2]
    if den is None: return img.copy()
    d = cv2.resize(den.astype(np.float32), (max(1, int(round(den.shape[1] * cell_xy[0]))), max(1, int(round(den.shape[0] * cell_xy[1])))), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((H, W), np.float32); hh, ww = min(H, d.shape[0]), min(W, d.shape[1]); full[:hh, :ww] = d[:hh, :ww]
    hi = max(float(np.percentile(full, 99.5)), 1e-6); norm = np.clip(full / hi, 0, 1); heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    a = (alpha * np.clip(norm * 4, 0, 1))[..., None]; return np.clip(img * (1 - a) + heat * a, 0, 255).astype(np.uint8)


# --- notebook 01, cell 24: put_lines --------------------
def put_lines(canvas, lines, org=(12, 26), scale=0.55, color=(235, 235, 235), dy=22):
    for i, t in enumerate(lines): cv2.putText(canvas, t, (org[0], org[1] + i * dy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return canvas


# --- notebook 01, cell 24: draw_predictions --------------------
def draw_predictions(canvas, final, sc, status=None, gt=None, r=3):
    """Draw markers on an already-scaled canvas. `sc` scales original coordinates to canvas coordinates."""
    if gt is not None and len(gt): _ring(canvas, gt * sc, COL_GT, r + 2)
    acc = final["accepted"] * sc; unc = final["uncertain"] * sc
    if status is None: _ring(canvas, acc, COL_ACC, r)
    else:
        st = np.asarray(status); ok = st == "tp"
        _ring(canvas, acc[ok], COL_ACC, r); _cross(canvas, acc[~ok], COL_FP, r)
    _ring(canvas, unc, COL_UNC, r); return canvas


# --- notebook 01, cell 24: render_panel --------------------
def render_panel(img, final, gt=None, status=None, density=None, cell_xy=None, regions=None, title="", stats=(), panel_w=800):
    """2x2 panel: [original (+GT)] [predicted markers] [density heat-map + flagged regions] [statistics]."""
    H, W = img.shape[:2]; sc = panel_w / W; ph = max(8, int(round(H * sc))); base = cv2.resize(img, (panel_w, ph), interpolation=cv2.INTER_AREA); r = max(3, int(round(0.005 * panel_w)))
    p1 = base.copy()
    if gt is not None: _ring(p1, gt * sc, COL_GT, r)
    p2 = draw_predictions(base.copy(), final, sc, status, gt if gt is not None else None, r)
    p3 = density_overlay(base, density, (cell_xy * sc) if cell_xy is not None else None) if density is not None else base.copy()
    if density is None: put_lines(p3, ["density branch not available for this model variant"], org=(12, 44), scale=0.5)
    for rg in (regions or []): x0, y0, x1, y1 = [int(round(v * sc)) for v in rg["box"]]; cv2.rectangle(p3, (x0, y0), (x1, y1), COL_REGION, 1, cv2.LINE_AA)
    p4 = np.full_like(base, 24); put_lines(p4, list(stats), org=(12, 44), scale=0.5)
    for p, t in ((p1, "original" + (" + ground truth (blue)" if gt is not None else "")), (p2, "predictions: green=accepted yellow=uncertain" + (" red x=FP blue=GT" if status is not None else "")), (p3, "density heat-map" + (" + re-analysed regions" if regions else "")), (p4, "statistics")):
        cv2.rectangle(p, (0, 0), (panel_w, 20), (0, 0, 0), -1); cv2.putText(p, t, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.concatenate([np.concatenate([p1, p2], 1), np.concatenate([p3, p4], 1)], 0)
    bar = np.full((34, grid.shape[1], 3), 0, np.uint8); cv2.putText(bar, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, grid], 0)


# --- notebook 04, cell 18: LEVEL_BGR --------------------
LEVEL_BGR = {0: (80, 160, 46), 1: (37, 168, 249), 2: (0, 108, 239), 3: (40, 40, 198)}


# --- notebook 04, cell 18: flow_to_color --------------------
def flow_to_color(flow, max_mag=None):
    mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)); mm = max_mag or max(float(np.percentile(mag, 99)), 1e-3)
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8); hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8); hsv[..., 1] = 255; hsv[..., 2] = np.clip(mag / mm * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# --- notebook 04, cell 18: draw_points --------------------
def draw_points(img, pts, conf=None, color=(255, 40, 40), r=3):
    out = img.copy()
    for k, (x, y) in enumerate(np.asarray(pts).reshape(-1, 2)): cv2.circle(out, (int(x), int(y)), r, color if conf is None or conf[k] >= 0.5 else (255, 200, 0), -1); cv2.circle(out, (int(x), int(y)), r + 1, (255, 255, 255), 1)
    return out


# --- notebook 04, cell 18: draw_flow_arrows --------------------
def draw_flow_arrows(img, flow, region=None, step=16, scale=None):
    out = img.copy(); H, W = flow.shape[:2]; mag = np.linalg.norm(flow, axis=-1); sc = scale or (step * 0.9 / max(float(np.percentile(mag, 98)), 1e-3))
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            if region is not None and not region[y, x]: continue
            dx, dy = flow[y, x] * sc; 
            if abs(dx) + abs(dy) < 0.5: continue
            cv2.arrowedLine(out, (x, y), (int(x + dx), int(y + dy)), (255, 255, 0), 1, tipLength=0.35)
    return out


# --- notebook 04, cell 18: draw_tracks --------------------
def draw_tracks(img, tracks, color=(0, 255, 255), max_tracks=150):
    out = img.copy()
    if tracks is None: return out
    idx = np.random.default_rng(0).permutation(tracks.shape[1])[:max_tracks]
    for i in idx:
        p = tracks[:, i]; ok = np.isfinite(p[:, 0]); q = p[ok]
        if len(q) >= 2: cv2.polylines(out, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA); cv2.circle(out, tuple(np.round(q[-1]).astype(int)), 2, (255, 0, 255), -1)
    return out


# --- notebook 04, cell 18: speed_heatmap --------------------
def speed_heatmap(flow, img=None, alpha=0.55):
    mag = np.linalg.norm(flow, axis=-1); m = np.clip(mag / max(float(np.percentile(mag, 99)), 1e-3), 0, 1); hm = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return hm if img is None else cv2.addWeighted(img, 1 - alpha, hm, alpha, 0)


# --- notebook 04, cell 18: viz_density_overlay --------------------
def viz_density_overlay(img, density, alpha=0.5):
    if density is None: return img
    d = cv2.resize(np.asarray(density, np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC); d = np.clip(d / max(float(d.max()), 1e-6), 0, 1)
    hm = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]; return cv2.addWeighted(img, 1 - alpha, hm, alpha, 0)


# --- notebook 04, cell 18: show_grid --------------------
def show_grid(images, titles, ncols=3, figsize=None, save=None):
    n = len(images); nr = int(np.ceil(n / ncols)); fig, axes = plt.subplots(nr, ncols, figsize=figsize or (5 * ncols, 3.6 * nr)); axes = np.atleast_1d(axes).reshape(-1)
    for a, im, t in zip(axes, images, titles): a.imshow(im); a.set_title(t, fontsize=10); a.axis("off")
    for a in axes[n:]: a.axis("off")
    plt.tight_layout()
    if save: fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


# --- notebook 04, cell 48: RISK_BGR --------------------
RISK_BGR = {0: (60, 160, 60), 1: (30, 190, 240), 2: (20, 130, 240), 3: (40, 40, 220)}


# --- notebook 04, cell 48: draw_overlay_v2 --------------------
def draw_overlay_v2(frame_bgr, info):
    """HUD: COUNT / MOTION / MOTION STATE / ABNORMAL / ABNORMAL TAGS / OCCUPANCY / RISK / RISK SCORE + coloured frame border for abnormal intervals."""
    out = frame_bgr.copy(); H, W = out.shape[:2]; sc = max(0.5, min(W, H) / 720.0); abn = bool(info.get("abnormal", False)); lv = int(info.get("risk_level_idx", 0)); col = RISK_BGR[lv] if not abn else (30, 30, 230)
    tags = [TAG_PHRASE.get(t, t) for t in info.get("tags", []) if t not in ("NORMAL", "MULTI_SIGNAL_ABNORMAL")][:3] if abn else []
    lines = [("COUNT", "n/a" if info.get("count") is None else str(int(info["count"]))), ("MOTION", info.get("motion_level", "n/a")), ("MOTION STATE", info.get("motion_state", "NORMAL")), ("ABNORMAL", "YES" if abn else "NO")]
    lines += [("ABNORMAL TAGS", ", ".join(tags) if tags else "-")] if abn else []
    lines += [("OCCUPANCY", info.get("occupancy_level", "n/a")), ("RISK", info.get("risk_level", "n/a")), ("RISK SCORE", f"{info.get('risk_score', 0):.0f} / 100")]
    fs, lh = 0.62 * sc, int(26 * sc); pw = int(max(cv2.getTextSize(f"{k}: {v}", cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0] for k, v in lines) + 24 * sc); ph = lh * len(lines) + int(14 * sc); ov = out.copy(); cv2.rectangle(ov, (8, 8), (8 + pw, 8 + ph), (20, 20, 20), -1); out = cv2.addWeighted(ov, 0.62, out, 0.38, 0)
    for i, (k, v) in enumerate(lines):
        vc = (255, 255, 255) if k not in ("ABNORMAL", "RISK", "RISK SCORE") else col if k != "ABNORMAL" else ((60, 60, 255) if abn else (120, 220, 120))
        cv2.putText(out, f"{k}: ", (18, 8 + int(lh * (i + 0.85))), cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA); off = cv2.getTextSize(f"{k}: ", cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
        cv2.putText(out, str(v), (18 + off, 8 + int(lh * (i + 0.85))), cv2.FONT_HERSHEY_SIMPLEX, fs, vc, 2 if k in ("ABNORMAL", "RISK") else 1, cv2.LINE_AA)
    if abn or lv >= 1:
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), col, max(4, int(8 * sc)))
        if abn: cv2.rectangle(out, (W // 2 - int(190 * sc), 0), (W // 2 + int(190 * sc), int(34 * sc)), (30, 30, 230), -1); cv2.putText(out, "ABNORMAL BEHAVIOUR", (W // 2 - int(170 * sc), int(25 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# --- notebook 04, cell 52: plot_full_timeline --------------------
def plot_full_timeline(TL, events, save=None, title=""):
    x = TL["time_s"].values; fig, ax = plt.subplots(8, 1, figsize=(13, 17), sharex=True)
    panels = [("crowd count / density", [("count", "count"), ("density", "density x1")], None), ("occupancy", [("occupancy", "crowd-region fraction"), ("occupied_area_ratio", "occupied-cell fraction")], None), ("speed (px/frame)", [("mean_speed", "mean"), ("p95_speed", "p95")], None),
              ("acceleration", [("mean_abs_acceleration", "|accel|")], None), ("direction instability / counter-flow", [("direction_change_rate", "direction change rate"), ("direction_variance", "direction variance"), ("counter_flow_ratio", "counter-flow ratio")], None),
              ("compression / scattering", [("compression_score", "compression"), ("scattering_score", "scattering")], None), ("abnormality evidence", [("abnormality_score", "abnormality score (noisy-OR)")], None), ("risk score", [("risk_score_raw", "raw"), ("risk_score", "smoothed")], (0, 100))]
    for a, (t, cols, yl) in zip(ax, panels):
        for c, lab in cols:
            if c in TL: a.plot(x, TL[c].values, lw=1.5, label=lab)
        a.set_ylabel(t, fontsize=8); a.grid(alpha=0.3); a.legend(fontsize=7, loc="upper left")
        if yl: a.set_ylim(*yl)
        for e in events: a.axvspan(e["start_s"], e["end_s"], color="red", alpha=0.12, lw=0)
    for e in (25, 50, 75): ax[-1].axhline(e, color="gray", ls="--", lw=0.6)
    ax[-1].set_xlabel("time (s)"); fig.suptitle(title or "timeline (red = detected abnormal event)"); plt.tight_layout()
    if save: fig.savefig(save, dpi=100, bbox_inches="tight")
    return fig


# --- notebook 04, cell 54: fig_to_rgb --------------------
def fig_to_rgb(fig):
    fig.canvas.draw(); a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig); return a


# ============================================================================================================================
# BACKEND-ONLY helpers
# ============================================================================================================================
def render_timeline_figure(TL, events, title, save_path):
    """notebook 04 cell 52: plot_full_timeline(...) saved as PNG (default matplotlib look, exactly as in the notebook)."""
    with plt.style.context("default"):
        fig = plot_full_timeline(TL, events, save_path, title)
        plt.close(fig)
    return save_path


def render_diagnostics(PK, SIGNALS, save_path):
    """notebook 04 cell 54: the ten visual diagnostics of the PEAK window (original frame, head points, density map, predicted optical flow,
    speed heat-map, direction vectors, flow-advected trajectories, crowd region/occupancy, abnormality evidence, final risk) as one PNG."""
    if PK is None: return None
    with plt.style.context("default"):
        _rgb, _fl, _reg, _w, _ci = PK["rgb"], PK["flow"], PK["region"], PK["window"], PK["count_info"]; _step = max(8, _rgb.shape[1] // 40)
        _pts = _ci["points"]
        _t = [(_rgb, "1. original frame"), (draw_points(_rgb, _pts, None) if _pts is not None else _rgb, f"2. crowd head points (count {int(_ci['count']) if _ci['count'] is not None else 'n/a'})"),
              (viz_density_overlay(_rgb, _ci["density"], 0.45) if _ci["density"] is not None else _rgb, "3. density map"), (flow_to_color(_fl), "4. optical flow (predicted)"),
              (speed_heatmap(_fl, _rgb), "5. speed heatmap"), (draw_flow_arrows(_rgb, _fl, _reg, step=_step), "6. direction vectors"),
              (draw_tracks(_rgb, PK["tracks"]) if PK["tracks"] is not None else _rgb, "7. flow-advected trajectories")]
        _ov = _rgb.copy(); _ov[_reg] = (0.55 * _ov[_reg] + 0.45 * np.array([255, 160, 0])).astype(np.uint8)
        _t.append((_ov, f"8. crowd region / occupancy ({100 * _reg.mean():.0f}% of image, level {_w['occupancy_level']})"))
        fg, ax = plt.subplots(figsize=(5, 3.4)); _E = [_w[f"E_{s}"] for s in SIGNALS]
        ax.barh(SIGNALS, _E, color=["tab:red" if _w[f"active_{s}"] else "tab:gray" for s in SIGNALS]); ax.set_xlim(0, 1)
        ax.set_title(f"abnormality {_w['abnormality_score']:.2f} | {_w['n_active_signals']} signals / {_w['n_active_groups']} groups", fontsize=9); plt.tight_layout(); _t.append((fig_to_rgb(fg), "9. abnormality evidence (red = active)"))
        fg, ax = plt.subplots(figsize=(5, 3.4)); ax.axis("off")
        _txt = f"RISK: {_w['risk_level']}  ({_w['risk_score']:.0f}/100)\nmotion state: {_w['motion_state']}\nabnormal: {'YES' if (_w['abnormal_online'] or _w['alarm_raw']) else 'NO'}\ntags: {', '.join(t for t in _w['tags'] if t != 'NORMAL') or '-'}\n\n" + "\n".join("- " + x for x in _w["evidence"][:4])
        ax.text(0, 1, _txt, va="top", fontsize=8, wrap=True); _t.append((fig_to_rgb(fg), "10. final risk"))
        fig = show_grid([x[0] for x in _t], [x[1] for x in _t], ncols=5, figsize=(22, 9), save=save_path)
        plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------------------------------------------------------- ffmpeg
def _resolve_ffmpeg():
    """Bundled binary from the `imageio-ffmpeg` pip package (no system install needed), else a system ffmpeg on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


_FFMPEG_PATH = None
_FFMPEG_RESOLVED = False


def _ffmpeg_available():
    global _FFMPEG_PATH, _FFMPEG_RESOLVED
    if not _FFMPEG_RESOLVED:
        _FFMPEG_PATH = _resolve_ffmpeg(); _FFMPEG_RESOLVED = True
    return _FFMPEG_PATH is not None


def finalize_video(raw_path, final_path):
    """OpenCV writes mp4v (MPEG-4 Part 2) which Chrome / Firefox cannot play inside <video>. Re-encode to H.264 / yuv420p with ffmpeg;
    if ffmpeg is unavailable the raw file is kept (still downloadable)."""
    raw_path, final_path = str(raw_path), str(final_path)
    if not _ffmpeg_available():
        print("[visualization] no ffmpeg found (pip install imageio-ffmpeg) - the annotated video keeps the mp4v codec and may not play in the browser.")
        os.replace(raw_path, final_path); return final_path
    try:
        subprocess.run([_FFMPEG_PATH, "-y", "-i", raw_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart", final_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(raw_path)
    except (subprocess.CalledProcessError, OSError) as error:
        tail = error.stderr.decode(errors="ignore")[-400:] if isinstance(error, subprocess.CalledProcessError) and error.stderr else ""
        print(f"[visualization] ffmpeg transcode failed: {error}\n{tail}\nkeeping the raw mp4v file.")
        os.replace(raw_path, final_path)
    return final_path


# ---------------------------------------------------------------------------------------------------------------------------- input preview
PREVIEW_FILENAME = "live_preview.mp4"
_BROWSER_SAFE_FOURCCS = {"h264", "avc1", "x264", "avc3", "vp08", "vp80", "vp09", "av01"}


def _decode_fourcc(v):
    v = int(v)
    if v <= 0: return ""
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip().lower()


def remux_for_preview(input_path, extension):
    """Zero-re-encode container remux so a non-mp4 upload (.avi/.mov/.mkv) can be previewed in the browser before analysis starts.
    Returns {"success": True, "url": ...} | {"success": True, "no_conversion_needed": True} | {"success": False, "error": ...}.  Analysis is unaffected."""
    extension = extension.lower()
    if extension == "mp4": return {"success": True, "no_conversion_needed": True}
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        cap.release(); return {"success": False, "error": "Could not read this video file to preview it."}
    fourcc = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)); cap.release()
    if fourcc not in _BROWSER_SAFE_FOURCCS:
        return {"success": False, "codec": fourcc or "unknown",
                "error": f"This .{extension} file uses a codec browsers can't preview directly ({fourcc or 'unknown'}). Only the live preview is skipped - analysis will run normally."}
    if not _ffmpeg_available(): return {"success": False, "error": "ffmpeg is unavailable for preview conversion."}
    out = PREVIEW_DIR / PREVIEW_FILENAME; out.unlink(missing_ok=True)
    try:
        subprocess.run([_FFMPEG_PATH, "-y", "-i", str(input_path), "-c", "copy", "-movflags", "+faststart", str(out)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        out.unlink(missing_ok=True)
        return {"success": False, "error": "Couldn't repackage this file for preview. Analysis will still run normally."}
    return {"success": True, "url": f"/outputs/previews/{PREVIEW_FILENAME}?v={uuid.uuid4().hex[:8]}"}
