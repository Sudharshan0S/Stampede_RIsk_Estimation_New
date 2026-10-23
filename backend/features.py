"""
Temporal motion features  (Notebook 03_Abnormal_Motion_Detection)

Verbatim copies of FeatureConfig / RiskConfig, crowd_region, compensate_ego, step_features (36 per-step statistics), advect_tracks, track_features,
window_features (one feature row per window of `window_len` steps, hop `hop`), crowd_regime, HeadTracker and StreamExtractor.
inject_flow / injection_label are TRAINING-ONLY helpers (synthetic anomaly injection); they are kept so StreamExtractor stays identical to the notebook.
"""
import os, json, dataclasses
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter, binary_dilation


# --- notebook 04, cell 11: FeatureConfig --------------------
@dataclass
class FeatureConfig:
    # ---- working resolution / time base --------------------------------------------------------------
    work_width: int = 512            # all flows / features are computed at this width (height keeps the aspect ratio). Chosen on validation in notebook 03.
    fps: float = 25.0                # ASSUMPTION about the TUB frame rate (the dataset folder has no fps field). Only used to express rates per second.
    frame_step: int = 1              # analysis step in frames (1 = every frame pair)
    # ---- temporal windows (configurable) -------------------------------------------------------------
    window_len: int = 20             # steps (frame pairs) per window.  Try 10 / 20 / 30.
    hop: int = 5                     # steps between consecutive windows
    multiscale: tuple = (5, 10)      # extra short-term summaries inside a window (last k steps)
    # ---- flow statistics ------------------------------------------------------------------------------
    min_flow: float = 0.02           # px/frame at work resolution: below this a pixel has no defined direction
    n_dir_bins: int = 16
    block: int = 8                   # block size for local variance statistics
    smooth_sigma: float = 1.5        # smoothing before divergence / curl
    min_region_pixels: int = 200
    compensate_ego: bool = True      # subtract the median motion of the non-crowd background (camera motion) when enough background exists
    region_mode: str = "auto"        # auto | density | flow | all | mask (mask = ground truth, labels/training-evaluation only)
    density_region_thr: float = 0.02 # people per density cell for the density-based crowd region
    region_dilate: int = 5
    # ---- tracks -----------------------------------------------------------------------------------------
    track_seeds: int = 600
    neigh_k: int = 5
    neigh_radius_frac: float = 0.06  # neighbour radius as a fraction of the image width
    # ---- reference scales (FITTED ON TRAINING SCENES ONLY; filled by features.fit_refs) -----------------
    refs: dict = field(default_factory=lambda: dict(speed_ref=0.2, density_ref=1.0, scat_ref=0.01, conv_ref=0.01, counter_ref=0.05, dirchange_ref=0.2))
    # ---- crowd regimes (people per 10k work-res pixels; refined from training data) ------------------------
    regime_edges: tuple = (0.5, 2.0, 6.0)   # sparse < e0 <= medium < e1 <= dense < e2 <= extreme

    def to_dict(self): return dataclasses.asdict(self)
    @classmethod
    def from_dict(cls, d):
        d = dict(d); known = {f.name for f in dataclasses.fields(cls)}
        out = cls(**{k: (tuple(v) if isinstance(v, list) and k in ("multiscale", "regime_edges") else v) for k, v in d.items() if k in known}); return out
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True); json.dump(self.to_dict(), open(path, "w"), indent=2)
    @classmethod
    def load(cls, path): return cls.from_dict(json.load(open(path)))


# --- notebook 04, cell 11: RISK_LEVELS --------------------
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# --- notebook 04, cell 11: STATES --------------------
STATES = ["NORMAL", "COHERENT_RUSH", "SCATTERING", "COUNTER_FLOW", "CONVERGENCE_COMPRESSION", "DIRECTION_INSTABILITY", "PANIC_LIKE_MOTION"]


# --- notebook 04, cell 11: RiskConfig --------------------
@dataclass
class RiskConfig:
    level_edges: tuple = (25.0, 50.0, 75.0)   # LOW | MEDIUM | HIGH | CRITICAL  (configurable; validated in notebook 05, never assumed optimal)
    level_centers: tuple = (12.5, 37.5, 62.5, 87.5)
    smooth_alpha: float = 0.35                # EMA weight of the newest window in the risk smoother
    hysteresis: float = 3.0                   # score margin needed to leave the current level
    min_event_windows: int = 2                # an abnormal interval must last at least this many windows
    top_k_reasons: int = 4
    def to_dict(self): return dataclasses.asdict(self)
    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known})


# --- notebook 04, cell 14: EPS --------------------
EPS = 1e-6


# --- notebook 04, cell 14: _wrap --------------------
def _wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


# --- notebook 04, cell 14: crowd_region --------------------
def crowd_region(flow, shape_hw, cfg: FeatureConfig, density=None, mask=None, points=None):
    """Boolean crowd region (H, W) at working resolution.
      mask    -> ground-truth mask (labels only)         density -> count-model density map (people per cell) resized
      flow    -> moving pixels (fallback when no count model is available)                       all -> every pixel"""
    H, W = shape_hw; mode = cfg.region_mode
    if mode == "auto": mode = "density" if density is not None else "flow"
    reg = None
    if mode == "mask" and mask is not None: reg = mask.astype(bool)
    elif mode == "density" and density is not None:
        d = cv2.resize(np.asarray(density, np.float32), (W, H), interpolation=cv2.INTER_LINEAR); reg = d > cfg.density_region_thr
        if points is not None and len(points):
            pm = np.zeros((H, W), np.uint8); pts = np.round(np.asarray(points)).astype(int); ok = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
            pm[pts[ok, 1], pts[ok, 0]] = 1; reg = reg | (cv2.dilate(pm, np.ones((9, 9), np.uint8)) > 0)
    elif mode == "flow" or reg is None:
        s = np.linalg.norm(flow, axis=-1) if flow is not None else np.zeros((H, W), np.float32)
        thr = max(cfg.min_flow * 2, 2.5 * float(np.median(s))); reg = s > thr
    if mode == "all": reg = np.ones((H, W), bool)
    if cfg.region_dilate and mode != "all" and reg is not None and reg.any(): reg = binary_dilation(reg, iterations=int(cfg.region_dilate))
    return reg


# --- notebook 04, cell 14: compensate_ego --------------------
def compensate_ego(flow, region, cfg: FeatureConfig):
    """Remove global (camera) translation: subtract the median flow of the non-crowd background when enough background exists."""
    if not cfg.compensate_ego or region is None: return flow, np.zeros(2, np.float32)
    bg = ~binary_dilation(region, iterations=6)
    if bg.mean() < 0.15: return flow, np.zeros(2, np.float32)
    med = np.array([np.median(flow[..., 0][bg]), np.median(flow[..., 1][bg])], np.float32)
    return flow - med, med


# --- notebook 04, cell 14: STEP_COLS --------------------
STEP_COLS = ["mean_speed", "median_speed", "std_speed", "p90_speed", "p95_speed", "max_speed", "moving_fraction", "dominant_dir", "coherence", "circ_var", "counter_flow_ratio", "counter_flow_mass",
             "n_clusters", "motion_entropy", "mean_div", "scattering_div", "convergence_div", "curl_abs", "local_speed_var", "local_dir_var", "motion_inconsistency", "flow_energy",
             "field_accel_mean", "field_accel_p90", "occupancy", "count", "density", "scattering_proxy", "compression_proxy", "ego_speed", "region_fallback", "region_area", "packed_density", "density_concentration", "local_density_gradient", "occupied_area_ratio"]


# --- notebook 04, cell 14: step_features --------------------
def step_features(flow, region, cfg: FeatureConfig, prev_flow=None, count=None, ego=None, density=None):
    """Statistics of one flow field inside `region` (bool HxW)."""
    H, W = flow.shape[:2]; f = {c: np.nan for c in STEP_COLS}
    fallback = region is None or int(region.sum()) < cfg.min_region_pixels
    sel = np.ones((H, W), bool) if fallback else region; f["region_fallback"] = float(fallback); f["occupancy"] = float(sel.mean()); f["ego_speed"] = float(np.linalg.norm(ego)) if ego is not None else 0.0
    u, v = flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32); s = np.hypot(u, v); sv = s[sel]
    f.update(mean_speed=float(sv.mean()), median_speed=float(np.median(sv)), std_speed=float(sv.std()), p90_speed=float(np.percentile(sv, 90)), p95_speed=float(np.percentile(sv, 95)), max_speed=float(sv.max()), flow_energy=float((sv ** 2).mean()))
    mov = sel & (s > cfg.min_flow); f["moving_fraction"] = float(mov.sum() / max(sel.sum(), 1))
    if mov.sum() >= 15:
        th = np.arctan2(v[mov], u[mov]); w = s[mov]; sw = w.sum() + EPS; cx, sx = (w * np.cos(th)).sum(), (w * np.sin(th)).sum(); R = float(np.hypot(cx, sx) / sw); dom = float(np.arctan2(sx, cx))
        f.update(dominant_dir=dom, coherence=R, circ_var=float(1 - np.hypot(np.cos(th).mean(), np.sin(th).mean())))
        d = np.abs(_wrap(th - dom)); al, op = w[d < np.pi / 4].sum(), w[d > 3 * np.pi / 4].sum(); f["counter_flow_ratio"] = float(op / (al + op + EPS)) if (al + op) > 0 else 0.0; f["counter_flow_mass"] = float(op / sw)
        nb = cfg.n_dir_bins; h, _ = np.histogram((th + np.pi) % (2 * np.pi), bins=nb, range=(0, 2 * np.pi), weights=w); h = h / (h.sum() + EPS); hs = 0.25 * np.roll(h, 1) + 0.5 * h + 0.25 * np.roll(h, -1)
        f["n_clusters"] = float(((hs > np.roll(hs, 1)) & (hs >= np.roll(hs, -1)) & (hs >= 0.12)).sum()); pe = h[h > 0]; f["motion_entropy"] = float(-(pe * np.log(pe)).sum() / np.log(nb))
    us, vs = gaussian_filter(u, cfg.smooth_sigma), gaussian_filter(v, cfg.smooth_sigma); uy, ux = np.gradient(us); vy, vx = np.gradient(vs)
    div, curl = ux + vy, vx - uy; f.update(mean_div=float(div[sel].mean()), scattering_div=float(np.maximum(div, 0)[sel].mean()), convergence_div=float(np.maximum(-div, 0)[sel].mean()), curl_abs=float(np.abs(curl)[sel].mean()))
    grad = np.sqrt(ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2); f["motion_inconsistency"] = float(grad[sel].mean() / (f["mean_speed"] + 1e-3))
    B = cfg.block; hb, wb = H // B, W // B
    if hb > 0 and wb > 0:                                               # local (block) variance of speed and direction
        crop = lambda a: a[:hb * B, :wb * B].reshape(hb, B, wb, B)
        m = crop(mov & sel).astype(np.float32); n = m.sum((1, 3)); ok = n >= 8
        if ok.any():
            sb = crop(s) * m; mu = sb.sum((1, 3)) / np.maximum(n, 1); var = (sb ** 2).sum((1, 3)) / np.maximum(n, 1) - mu ** 2; f["local_speed_var"] = float(np.maximum(var, 0)[ok].mean())
            th_all = np.arctan2(v, u); cb = (crop(np.cos(th_all)) * m).sum((1, 3)) / np.maximum(n, 1); sb2 = (crop(np.sin(th_all)) * m).sum((1, 3)) / np.maximum(n, 1); f["local_dir_var"] = float((1 - np.hypot(cb, sb2))[ok].mean())
    if prev_flow is not None:
        a = np.hypot(u - prev_flow[..., 0], v - prev_flow[..., 1])[sel]; f["field_accel_mean"] = float(a.mean()); f["field_accel_p90"] = float(np.percentile(a, 90))
    if count is not None:
        f["count"] = float(count); f["density"] = float(count) / (H * W) * 1e4
        co = f["coherence"] if np.isfinite(f["coherence"]) else 1.0; f["scattering_proxy"] = f["scattering_div"] * (1 - co); f["compression_proxy"] = f["convergence_div"] * (1 + f["density"] / max(cfg.refs.get("density_ref", 1.0), 1e-3))
    else: f["scattering_proxy"] = f["scattering_div"] * (1 - (f["coherence"] if np.isfinite(f["coherence"]) else 1.0)); f["compression_proxy"] = f["convergence_div"]
    # ---- NEW: occupancy / density-map statistics (need the crowd-count model's density map; NaN when it is not available) ----
    f["region_area"] = float(sel.sum()) / 1e4
    if count is not None and f["region_area"] > 0: f["packed_density"] = float(count) / f["region_area"]          # people per 10k px of crowd region
    if density is not None and getattr(density, "size", 0) > 1:
        dm = np.asarray(density, np.float32); tot = float(dm.sum())
        if tot > 1e-6:
            flat = np.sort(dm.ravel())[::-1]; kk = max(1, int(np.ceil(0.1 * flat.size))); f["density_concentration"] = float(flat[:kk].sum() / tot)       # share of the crowd inside the densest 10% of cells
            dn = dm / (float(dm.mean()) + 1e-9); gy, gx = np.gradient(dn) if min(dn.shape) > 1 else (np.zeros_like(dn), np.zeros_like(dn)); f["local_density_gradient"] = float(np.hypot(gx, gy).mean())
            f["occupied_area_ratio"] = float((dm > cfg.density_region_thr).mean())
    return f


# --- notebook 04, cell 14: _bilinear --------------------
def _bilinear(field, pts):
    H, W = field.shape[:2]; x = np.clip(pts[:, 0], 0, W - 1.001); y = np.clip(pts[:, 1], 0, H - 1.001); x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int); ax, ay = (x - x0)[:, None], (y - y0)[:, None]
    return (field[y0, x0] * (1 - ax) * (1 - ay) + field[y0, x0 + 1] * ax * (1 - ay) + field[y0 + 1, x0] * (1 - ax) * ay + field[y0 + 1, x0 + 1] * ax * ay)


# --- notebook 04, cell 14: advect_tracks --------------------
def advect_tracks(flows, region, cfg: FeatureConfig, n_seeds=None, seed=0):
    """Pseudo-trajectories: seeds sampled inside the crowd region are moved through the flow fields of one window.
    flows: list of W flow fields (H,W,2).  Returns (W+1, N, 2) positions (NaN once a track leaves the image). Works at ANY density (no identities needed)."""
    H, W = flows[0].shape[:2]; n_seeds = n_seeds or cfg.track_seeds; rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(region if region is not None and region.sum() >= cfg.min_region_pixels else np.ones((H, W), bool))
    k = min(n_seeds, len(ys)); idx = rng.choice(len(ys), k, replace=False); pos = np.stack([xs[idx], ys[idx]], 1).astype(np.float32)
    out = np.full((len(flows) + 1, k, 2), np.nan, np.float32); out[0] = pos; cur = pos.copy(); alive = np.ones(k, bool)
    for t, fl in enumerate(flows):
        cur = cur + _bilinear(fl.astype(np.float32), np.where(alive[:, None], cur, 0.0))
        alive &= np.isfinite(cur).all(1) & (cur[:, 0] >= 0) & (cur[:, 0] <= W - 1) & (cur[:, 1] >= 0) & (cur[:, 1] <= H - 1)     # a track that left the image never comes back
        out[t + 1] = np.where(alive[:, None], cur, np.nan)
    return out


# --- notebook 04, cell 14: _seg_intersect --------------------
def _seg_intersect(p1, p2, q1, q2):
    d = lambda a, b, c: (c[:, 1] - a[:, 1]) * (b[:, 0] - a[:, 0]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return (d(p1, q1, q2) * d(p2, q1, q2) < 0) & (d(p1, p2, q1) * d(p1, p2, q2) < 0)


# --- notebook 04, cell 14: TRACK_COLS --------------------
TRACK_COLS = ["n_tracks", "track_survival", "trajectory_density", "track_speed", "turn_rate", "traj_dispersion", "traj_dispersion_ratio", "relative_speed", "relative_speed_norm", "relative_direction", "track_approach_fraction", "track_closing_rate", "track_crossing_rate"]


# --- notebook 04, cell 14: track_features --------------------
def track_features(tracks, shape_hw, cfg: FeatureConfig, max_tracks=1500):
    """tracks: (T, N, 2) positions in working-resolution pixels (NaN = undefined).  Interaction / compression motion SIGNATURES:
    relative velocity, approach (closing) rate, trajectory crossing, heading change. They are indirect signatures, not proof of pushing."""
    H, W = shape_hw; f = {c: np.nan for c in TRACK_COLS}
    if tracks is None or tracks.ndim != 3 or tracks.shape[0] < 3: f["n_tracks"] = 0.0; return f
    ok = np.isfinite(tracks[..., 0]).sum(0) >= max(3, int(0.6 * tracks.shape[0])); tr = tracks[:, ok]; f["n_tracks"] = float(tr.shape[1]); f["track_survival"] = float(ok.mean())
    if tr.shape[1] < 8: return f
    if tr.shape[1] > max_tracks: tr = tr[:, np.random.default_rng(0).choice(tr.shape[1], max_tracks, replace=False)]
    T, N, _ = tr.shape; f["trajectory_density"] = tr.shape[1] / (H * W) * 1e4
    vel = np.diff(tr, axis=0); sp = np.linalg.norm(vel, axis=-1); f["track_speed"] = float(np.nanmean(sp))
    head = np.arctan2(vel[..., 1], vel[..., 0]); dth = np.abs(_wrap(np.diff(head, axis=0))); okm = (sp[:-1] > cfg.min_flow) & (sp[1:] > cfg.min_flow); f["turn_rate"] = float(np.nanmean(np.where(okm, dth, np.nan))) if okm.any() else np.nan
    first = np.array([tr[np.argmax(np.isfinite(tr[:, i, 0])), i] for i in range(N)]); last = np.array([tr[T - 1 - np.argmax(np.isfinite(tr[::-1, i, 0])), i] for i in range(N)])
    sd = lambda p: float(np.sqrt(np.nanvar(p[:, 0]) + np.nanvar(p[:, 1]))); f["traj_dispersion"] = sd(last) / np.sqrt(H * W); f["traj_dispersion_ratio"] = sd(last) / (sd(first) + EPS)
    mv = np.nanmean(vel, axis=0); mid = tr[T // 2]; good = np.isfinite(mid).all(1) & np.isfinite(mv).all(1) & np.isfinite(first).all(1) & np.isfinite(last).all(1)
    if good.sum() >= 8:
        P, V, F0, F1 = mid[good], mv[good], first[good], last[good]; tree = cKDTree(P); k = min(cfg.neigh_k + 1, len(P)); dist, nb = tree.query(P, k=k); rad = cfg.neigh_radius_frac * W
        ii = np.repeat(np.arange(len(P)), k - 1); jj = nb[:, 1:].reshape(-1); keep = dist[:, 1:].reshape(-1) < rad; ii, jj = ii[keep], jj[keep]
        if len(ii) >= 10:
            rv = np.linalg.norm(V[ii] - V[jj], axis=1); f["relative_speed"] = float(rv.mean()); f["relative_speed_norm"] = float(rv.mean() / (np.linalg.norm(V, axis=1).mean() + 1e-3))
            si, sj = np.linalg.norm(V[ii], axis=1), np.linalg.norm(V[jj], axis=1); mm = (si > cfg.min_flow) & (sj > cfg.min_flow)
            if mm.any(): cosv = (V[ii][mm] * V[jj][mm]).sum(1) / (si[mm] * sj[mm] + EPS); f["relative_direction"] = float(np.arccos(np.clip(cosv, -1, 1)).mean())
            d0 = np.linalg.norm(F0[ii] - F0[jj], axis=1); d1 = np.linalg.norm(F1[ii] - F1[jj], axis=1); f["track_approach_fraction"] = float((d1 < 0.9 * d0).mean()); f["track_closing_rate"] = float(np.mean((d0 - d1) / (T - 1)))
            moved = (np.linalg.norm(F1[ii] - F0[ii], axis=1) > 0.5) & (np.linalg.norm(F1[jj] - F0[jj], axis=1) > 0.5); f["track_crossing_rate"] = float(_seg_intersect(F0[ii], F1[ii], F0[jj], F1[jj])[moved].mean()) if moved.any() else 0.0
    return f


# --- notebook 04, cell 14: _thirds --------------------
def _thirds(x):
    n = len(x); k = max(1, n // 3); return x[:k], x[-k:]


# --- notebook 04, cell 14: _slope --------------------
def _slope(y):
    y = np.asarray(y, float); ok = np.isfinite(y)
    if ok.sum() < 3: return np.nan
    return float(np.polyfit(np.arange(len(y))[ok], y[ok], 1)[0])


# --- notebook 04, cell 14: window_features --------------------
def window_features(step_rows, cfg: FeatureConfig, tracks=None, shape_hw=None):
    """step_rows: list of W dicts from step_features (consecutive steps).  Returns one dict (one row of motion_feature_dataset.csv)."""
    import pandas as pd
    d = pd.DataFrame(step_rows); W = len(d); refs = cfg.refs; col = lambda c: d[c].values.astype(float) if c in d else np.full(W, np.nan)
    nm = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    f = {}; ms = col("mean_speed"); a_t = np.diff(ms) if W > 1 else np.array([np.nan]); a_f, a_l = _thirds(ms); sF, sL = nm(a_f), nm(a_l)
    f.update(mean_speed=nm(ms), median_speed=nm(col("median_speed")), speed_std=nm(col("std_speed")), speed_std_time=float(np.nanstd(ms)), p90_speed=nm(col("p90_speed")), p95_speed=nm(col("p95_speed")),
             max_speed=float(np.nanmax(col("max_speed"))), speed_change=sL - sF)
    eps_s = max(0.05 * refs["speed_ref"], 1e-3); rate = cfg.fps / max(cfg.frame_step, 1)
    f.update(mean_acceleration=nm(a_t), mean_abs_acceleration=nm(np.abs(a_t)), acceleration_std=float(np.nanstd(a_t)), p90_acceleration=float(np.nanpercentile(np.abs(a_t), 90)) if np.isfinite(a_t).any() else np.nan,
             field_accel_mean=nm(col("field_accel_mean")), field_accel_p90=nm(col("field_accel_p90")), crowd_acceleration_ratio=(sL + eps_s) / (sF + eps_s),
             jerk=nm(np.abs(np.diff(a_t))) if len(a_t) > 1 else np.nan, acceleration_change_rate=_slope(np.abs(a_t)) if len(a_t) > 2 else np.nan, mean_acceleration_per_s2=nm(a_t) * rate ** 2)
    dom = col("dominant_dir"); okd = np.isfinite(dom); coh = col("coherence")
    if okd.sum() >= 2:
        w = np.where(np.isfinite(coh[okd]), coh[okd], 0.5); cx, sx = (w * np.cos(dom[okd])).sum(), (w * np.sin(dom[okd])).sum(); f["dominant_direction_deg"] = float(np.degrees(np.arctan2(sx, cx)) % 360)
        f["dominant_dir_stability"] = float(np.hypot(np.cos(dom[okd]).mean(), np.sin(dom[okd]).mean())); dch = np.abs(_wrap(np.diff(dom[okd]))); f["direction_change_rate"] = float(dch.mean())
    else: f.update(dominant_direction_deg=np.nan, dominant_dir_stability=np.nan, direction_change_rate=np.nan)
    f.update(direction_coherence=nm(coh), direction_variance=nm(col("circ_var")), counter_flow_ratio=nm(col("counter_flow_ratio")), counter_flow_mass=nm(col("counter_flow_mass")), n_direction_clusters=nm(col("n_clusters")),
             max_direction_clusters=float(np.nanmax(col("n_clusters"))) if np.isfinite(col("n_clusters")).any() else np.nan, moving_fraction=nm(col("moving_fraction")))
    mi = nm(col("motion_inconsistency"))
    f.update(divergence=nm(col("mean_div")), scattering_div=nm(col("scattering_div")), convergence=nm(col("convergence_div")), curl=nm(col("curl_abs")), local_speed_variance=nm(col("local_speed_var")),
             local_direction_variance=nm(col("local_dir_var")), flow_coherence=1.0 / (1.0 + mi) if np.isfinite(mi) else np.nan, motion_entropy=nm(col("motion_entropy")), motion_inconsistency=mi, flow_energy=nm(col("flow_energy")),
             scattering_score=nm(col("scattering_proxy")), compression_score=nm(col("compression_proxy")), ego_speed=nm(col("ego_speed")), region_fallback_fraction=nm(col("region_fallback")))
    dens = col("density"); dF, dL = _thirds(dens); f.update(crowd_count=nm(col("count")), density=nm(dens), occupancy=nm(col("occupancy")), density_change=nm(dL) - nm(dF), density_change_rate=_slope(dens) * rate if np.isfinite(dens).any() else np.nan)
    h = W // 2; f["density_acceleration"] = (_slope(dens[h:]) - _slope(dens[:h])) * rate ** 2 if W >= 8 and np.isfinite(dens).sum() >= 8 else np.nan
    # ---- NEW window-level occupancy / density-map / motion-spread features ----
    cnt_ = col("count"); occ_ = col("occupancy"); sd_ = col("std_speed")
    f.update(density_variance=float(np.nanvar(dens)) if np.isfinite(dens).any() else np.nan, crowd_count_rate=_slope(cnt_) * rate if np.isfinite(cnt_).any() else np.nan, occupancy_rate=_slope(occ_) * rate if np.isfinite(occ_).any() else np.nan,
             density_rate=f["density_change_rate"], flow_magnitude_variance=nm(sd_) ** 2 if np.isfinite(sd_).any() else np.nan, direction_entropy=f["motion_entropy"], local_density_gradient=nm(col("local_density_gradient")),
             density_concentration=nm(col("density_concentration")), crowd_region_area=nm(col("region_area")), occupied_area_ratio=nm(col("occupied_area_ratio")), packed_density=nm(col("packed_density")))
    # temporal transitions (first third -> last third); scales come from training-scene reference statistics
    lg = np.log((sL + eps_s) / (sF + eps_s)); sr = refs["speed_ref"]; cF, cL = (nm(x) for x in _thirds(coh)); scF, scL = (nm(x) for x in _thirds(col("scattering_proxy"))); coF, coL = (nm(x) for x in _thirds(col("counter_flow_ratio")))
    cvF, cvL = (nm(x) for x in _thirds(col("convergence_div"))); r0 = lambda x: max(x, 0.0) if np.isfinite(x) else 0.0
    f.update(trans_stop_to_move=r0(lg) * float(sF < 0.5 * sr), trans_move_to_rapid=r0(lg) * float(sL > 1.5 * sr), trans_normal_to_accel=r0(f["crowd_acceleration_ratio"] - 1.0),
             trans_coherent_to_scatter=r0(cF - cL) * min(r0(scL - scF) / (refs["scat_ref"] + EPS), 5.0), trans_coherent_to_counter=r0(cF - cL) * min(r0(coL - coF) / (refs["counter_ref"] + EPS), 5.0),
             trans_lowdens_to_compression=min(r0(cvL - cvF) / (refs["conv_ref"] + EPS), 5.0) / (1.0 + (nm(dF) if np.isfinite(nm(dF)) else 0.0) / refs["density_ref"]), trans_dir_instability=np.nan)
    if okd.sum() >= 6:
        dc = np.abs(_wrap(np.diff(dom[okd]))); a, b = _thirds(dc); f["trans_dir_instability"] = min(r0(nm(b) - nm(a)) / (refs["dirchange_ref"] + EPS), 5.0)
    for k in cfg.multiscale:                                                            # short-term summaries of the last k steps
        if W >= k: f[f"speed_last{k}"] = nm(ms[-k:]); f[f"field_accel_last{k}"] = nm(col("field_accel_mean")[-k:])
        else: f[f"speed_last{k}"] = np.nan; f[f"field_accel_last{k}"] = np.nan
    f.update(track_features(tracks, shape_hw, cfg) if (tracks is not None and shape_hw is not None) else {c: np.nan for c in TRACK_COLS})
    return f


# --- notebook 04, cell 14: crowd_regime --------------------
def crowd_regime(density, cfg: FeatureConfig):
    """sparse / medium / dense / extreme from the density feature (people per 10k work-res pixels). Edges are configurable and refined on training data."""
    if not np.isfinite(density): return "unknown"
    e = cfg.regime_edges; return "sparse" if density < e[0] else "medium" if density < e[1] else "dense" if density < e[2] else "extreme"


# --- notebook 04, cell 15: INJECTION_KINDS --------------------
INJECTION_KINDS = ["rush", "scatter", "converge", "counter_flow", "direction_chaos", "panic"]


# --- notebook 04, cell 15: inject_flow --------------------
def inject_flow(flows, region, kind, severity, speed_ref, seed=0):
    """Perturb a window of flow fields (list of HxWx2) INSIDE the crowd region and return (new_flows, density_multiplier_per_step).
    severity 1..3 scales the perturbation (known by construction).  The perturbation grows during the window so the transition is visible.
      rush            coherent acceleration along the dominant direction      scatter   radial outward motion from the crowd centroid
      converge        radial inward motion (+ density rise)                   counter_flow  half of the crowd (left/right) reverses direction
      direction_chaos random heading changes (varying per step)               panic     rush + scatter + chaos together"""
    rng = np.random.default_rng(seed); H, W = flows[0].shape[:2]; n = len(flows); reg = region.astype(np.float32)[..., None]
    ys, xs = np.nonzero(region) if region.any() else (np.array([H // 2]), np.array([W // 2])); cx, cy = xs.mean(), ys.mean()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); rx, ry = xx - cx, yy - cy; rn = np.sqrt(rx ** 2 + ry ** 2) + 1.0; radial = np.stack([rx / rn, ry / rn], -1)
    mean_flow = np.mean([f[region].mean(0) if region.any() else np.zeros(2) for f in flows], axis=0); dn = np.linalg.norm(mean_flow); dirv = mean_flow / dn if dn > 1e-3 else np.array([1.0, 0.0], np.float32)
    amp = speed_ref * (0.8 + 1.2 * severity); out, dm = [], []; left = (xx < cx)[..., None]
    for t, f in enumerate(flows):
        g = (t + 1) / n; f = f.astype(np.float32).copy(); mult = 1.0
        if kind in ("rush", "panic"): f = f * (1 + 1.5 * severity * g) + (dirv * amp * g * 2.0).astype(np.float32) * reg
        if kind in ("scatter", "panic"): f = f + radial * (amp * (2.0 if kind == "scatter" else 1.2) * g) * reg
        if kind == "converge": f = f - radial * (amp * 2.0 * g) * reg; mult = 1 + 0.35 * severity * g
        if kind == "counter_flow": f = np.where(left & (reg > 0), -f - dirv * amp * g, f + dirv * amp * g * 0.5)
        if kind in ("direction_chaos", "panic"):
            ang = rng.normal(0, 0.9 * severity * g * (1.0 if kind == "direction_chaos" else 0.7)); c, s = np.cos(ang), np.sin(ang); rot = np.stack([f[..., 0] * c - f[..., 1] * s, f[..., 0] * s + f[..., 1] * c], -1)
            f = np.where(reg > 0, rot + rng.normal(0, 0.15 * amp * severity * g, f.shape).astype(np.float32), f)
        out.append(f.astype(np.float32)); dm.append(mult)
    return out, dm


# --- notebook 04, cell 15: injection_label --------------------
def injection_label(kind, severity):
    """(motion_state, risk_level) of an injected window - known by construction."""
    state = dict(rush="COHERENT_RUSH", scatter="SCATTERING", converge="CONVERGENCE_COMPRESSION", counter_flow="COUNTER_FLOW", direction_chaos="DIRECTION_INSTABILITY", panic="PANIC_LIKE_MOTION")[kind]
    risk = min(3, severity) if kind != "panic" else 3; return state, int(risk)


# --- notebook 04, cell 17: HeadTracker --------------------
class HeadTracker:
    def __init__(self, max_dist=18.0, max_missed=3, min_hits=2, history=60):
        self.max_dist, self.max_missed, self.min_hits, self.history = max_dist, max_missed, min_hits, history
        self.tracks = {}; self.next_id = 0; self.frame = -1; self.last_match_rate = np.nan

    def update(self, points, flow=None, frame_index=None):
        """points: (N,2) head points in the SAME coordinates as `flow` (working resolution). flow: flow from the previous frame to this one is
        NOT needed - we pass the flow estimated between previous and current frame to predict the previous positions forward."""
        self.frame = self.frame + 1 if frame_index is None else frame_index; pts = np.asarray(points, np.float32).reshape(-1, 2)
        ids = list(self.tracks); pred = []
        for i in ids:
            p = self.tracks[i]["pos"][-1].copy()
            if flow is not None: p = p + _bilinear(flow.astype(np.float32), p[None])[0]
            pred.append(p)
        matched_t, matched_d = {}, set()
        if ids and len(pts):
            cost = cdist(np.asarray(pred), pts); cost[cost > self.max_dist] = 1e6; r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] < 1e5: matched_t[ids[a]] = b; matched_d.add(b)
        for i in ids:
            t = self.tracks[i]
            if i in matched_t: t["pos"].append(pts[matched_t[i]]); t["frames"].append(self.frame); t["missed"] = 0; t["hits"] += 1
            else: t["missed"] += 1
            if len(t["pos"]) > self.history: t["pos"] = t["pos"][-self.history:]; t["frames"] = t["frames"][-self.history:]
        for i in [i for i, t in self.tracks.items() if t["missed"] > self.max_missed]: del self.tracks[i]
        for b in range(len(pts)):
            if b not in matched_d: self.tracks[self.next_id] = dict(pos=[pts[b]], frames=[self.frame], missed=0, hits=1); self.next_id += 1
        self.last_match_rate = len(matched_t) / max(len(ids), 1) if ids else np.nan
        return self.active()

    def active(self):
        return {i: t for i, t in self.tracks.items() if t["hits"] >= self.min_hits and t["missed"] == 0}

    @property
    def identity_ok(self):
        """Identity tracking is considered reliable only when most previous heads were re-found."""
        return bool(np.isfinite(self.last_match_rate) and self.last_match_rate >= 0.6)

    def as_array(self, n_frames):
        """Recent trajectories as (n_frames, N, 2) (NaN where undefined), ending at the current frame."""
        act = self.active(); out = np.full((n_frames, len(act), 2), np.nan, np.float32)
        for k, t in enumerate(act.values()):
            for f, p in zip(t["frames"], t["pos"]):
                r = n_frames - 1 - (self.frame - f)
                if 0 <= r < n_frames: out[r, k] = p
        return out


# --- notebook 04, cell 19: StreamExtractor --------------------
class StreamExtractor:
    """push(frame, ...) one frame at a time; yields step info and, every `hop` steps once `window_len` steps exist, a window feature row."""
    def __init__(self, estimator, cfg: FeatureConfig):
        self.est, self.cfg = estimator, cfg; self.compute_windows = True; self.reset()
    def reset(self):
        c = self.cfg; self.prev_frame = None; self.prev_flow = None; self.n = 0; self.steps = deque(maxlen=c.window_len); self.flows = deque(maxlen=c.window_len); self.regions = deque(maxlen=c.window_len)
        self.counts = deque(maxlen=c.window_len); self.shape = None

    def push(self, frame, count=None, density=None, points=None, gt_flow=None, mask=None):
        cfg = self.cfg
        if self.prev_frame is None: self.prev_frame = frame; self.shape = frame.shape[:2]; return None
        flow = gt_flow if gt_flow is not None else self.est(self.prev_frame, frame)
        region = crowd_region(flow, self.shape, cfg, density=density, mask=mask, points=points); flow_c, ego = compensate_ego(flow, region, cfg)
        sf = step_features(flow_c, region, cfg, self.prev_flow, count, ego, density); self.steps.append(sf); self.flows.append(flow_c); self.regions.append(region); self.prev_flow = flow_c; self.prev_frame = frame; self.n += 1
        info = dict(step_index=self.n - 1, flow=flow_c, region=region, step=sf, window=None)
        if self.compute_windows and len(self.steps) >= cfg.window_len and (self.n - cfg.window_len) % cfg.hop == 0:
            tracks = advect_tracks(list(self.flows), self.regions[0], cfg, seed=self.n); row = window_features(list(self.steps), cfg, tracks, self.shape)
            row.update(start_frame=self.n - cfg.window_len, end_frame=self.n, regime=crowd_regime(row.get("density", np.nan), cfg)); info["window"] = row; info["tracks"] = tracks
        return info

    def injected_window(self, kind, severity, seed=0):
        """Window features of the CURRENT window after injecting a synthetic anomaly into its flow fields (labels known by construction)."""
        cfg = self.cfg; flows = list(self.flows); region = self.regions[0]; new, dm = inject_flow(flows, region, kind, severity, cfg.refs["speed_ref"], seed)
        steps, prev = [], None
        for fl, reg, st, m in zip(new, self.regions, self.steps, dm):
            cnt = st["count"] * m if np.isfinite(st.get("count", np.nan)) else None; steps.append(step_features(fl, reg, cfg, prev, cnt, None)); prev = fl
        tracks = advect_tracks(new, region, cfg, seed=seed); row = window_features(steps, cfg, tracks, self.shape); row.update(start_frame=self.n - cfg.window_len, end_frame=self.n, regime=crowd_regime(row.get("density", np.nan), cfg))
        st_name, risk = injection_label(kind, severity); row.update(motion_state=st_name, risk_label=risk, label_source="injected", injection_kind=kind, severity=severity); return row
