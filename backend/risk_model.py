"""
Risk model, temporal-change (normal-baseline) model and multi-signal abnormality detector
(Notebook 03 cells 16/39/41 and Notebook 04 cell 36; identical copies).

Verbatim copies of: FeatureScaler, _Wrap, RiskModel (LightGBM level + motion-state models, occlusion explanation), RiskSmoother (EMA + hysteresis),
ChangeModel, signal_scores, AbnormalityDetector, intervals_from_levels, fill_gaps.

IMPORTANT - models/risk_model.pkl was written with joblib from inside a notebook, so the pickle refers to these classes as `__main__.RiskModel`,
`__main__.FeatureScaler`, `__main__._Wrap`, `__main__.RiskConfig`, ... `load_risk_bundle()` at the bottom makes those names resolvable here.
"""
import os, copy, re
import numpy as np
import pandas as pd

from common import json_safe
from features import RISK_LEVELS, STATES, RiskConfig


# --- notebook 04, cell 16: MODEL_FEATURES --------------------
MODEL_FEATURES = [
    # crowd
    "crowd_count", "density", "occupancy", "density_change", "density_change_rate", "density_acceleration",
    # speed
    "mean_speed", "median_speed", "speed_std", "p90_speed", "p95_speed", "max_speed", "speed_change", "speed_std_time",
    # acceleration
    "mean_acceleration", "mean_abs_acceleration", "acceleration_std", "p90_acceleration", "field_accel_mean", "field_accel_p90", "crowd_acceleration_ratio", "jerk", "acceleration_change_rate",
    # direction
    "dominant_dir_stability", "direction_change_rate", "direction_coherence", "direction_variance", "counter_flow_ratio", "counter_flow_mass", "n_direction_clusters", "max_direction_clusters", "moving_fraction",
    # spatial
    "divergence", "scattering_div", "convergence", "curl", "local_speed_variance", "local_direction_variance", "flow_coherence", "motion_entropy", "motion_inconsistency", "flow_energy",
    "scattering_score", "compression_score",
    # transitions
    "trans_stop_to_move", "trans_move_to_rapid", "trans_normal_to_accel", "trans_coherent_to_scatter", "trans_coherent_to_counter", "trans_lowdens_to_compression", "trans_dir_instability",
    # multi-scale
    "speed_last5", "field_accel_last5", "speed_last10", "field_accel_last10",
    # trajectories
    "trajectory_density", "track_survival", "track_speed", "turn_rate", "traj_dispersion", "traj_dispersion_ratio", "relative_speed", "relative_speed_norm", "relative_direction",
    "track_approach_fraction", "track_closing_rate", "track_crossing_rate",
    # temporal context (previous window) - 'waves'
    "d_speed_prev", "d_accel_prev", "d_density_prev", "d_coherence_prev", "d_counter_prev", "d_compression_prev",
]


# --- notebook 04, cell 16: resolve_features --------------------
def resolve_features(df):
    """MODEL_FEATURES that exist in df, plus the multi-scale columns (speed_last{k}, field_accel_last{k}) for whatever window sizes were configured."""
    import re
    pat = re.compile(r"^(speed|field_accel)_last\d+$"); ms = sorted(c for c in df.columns if pat.match(c))
    return [f for f in MODEL_FEATURES if not pat.match(f) and f in df.columns] + ms


# --- notebook 04, cell 16: NOT_FOR_MODEL --------------------
NOT_FOR_MODEL = ["ego_speed", "region_fallback_fraction", "dominant_direction_deg", "n_tracks"]       # kept in the CSV; excluded from models (camera / scene-identity proxies)


# --- notebook 04, cell 16: FACTORS --------------------
FACTORS = {      # human-readable factor -> (representative feature, phrase when elevated, features it groups for the occlusion explanation)
    "crowd acceleration": ("crowd_acceleration_ratio", "sudden crowd acceleration", ["crowd_acceleration_ratio", "mean_acceleration", "mean_abs_acceleration", "p90_acceleration", "field_accel_mean", "field_accel_p90", "jerk", "acceleration_change_rate", "trans_normal_to_accel", "d_accel_prev", "speed_change"]),
    "speed": ("p95_speed", "high crowd speed", ["mean_speed", "median_speed", "p90_speed", "p95_speed", "max_speed", "flow_energy", "speed_last5", "speed_last10", "trans_move_to_rapid", "trans_stop_to_move", "d_speed_prev"]),
    "density increase": ("density_change", "increasing crowd density", ["density", "density_change", "density_change_rate", "density_acceleration", "d_density_prev", "crowd_count", "occupancy"]),
    "directional coherence": ("direction_coherence", "loss of directional coherence", ["direction_coherence", "direction_variance", "dominant_dir_stability", "moving_fraction", "flow_coherence"]),
    "counter-flow": ("counter_flow_ratio", "counter-flow (opposing directions)", ["counter_flow_ratio", "counter_flow_mass", "n_direction_clusters", "max_direction_clusters", "trans_coherent_to_counter", "d_counter_prev"]),
    "scattering": ("scattering_score", "scattering / outward motion", ["scattering_score", "scattering_div", "divergence", "traj_dispersion_ratio", "trans_coherent_to_scatter"]),
    "local compression": ("compression_score", "local compression / convergence", ["compression_score", "convergence", "trans_lowdens_to_compression", "track_approach_fraction", "track_closing_rate", "d_compression_prev"]),
    "motion instability": ("direction_change_rate", "increasing motion instability", ["direction_change_rate", "motion_inconsistency", "motion_entropy", "local_speed_variance", "local_direction_variance", "curl", "turn_rate", "trans_dir_instability", "speed_std_time", "acceleration_std"]),
    "interaction signatures": ("relative_speed_norm", "strong relative motion between neighbours", ["relative_speed", "relative_speed_norm", "relative_direction", "track_crossing_rate"]),
}


# --- notebook 04, cell 16: FeatureScaler --------------------
class FeatureScaler:
    """Median imputation + clipping to training percentiles + standardisation. FIT ON TRAINING DATA ONLY."""
    def __init__(self, features): self.features = list(features); self.fitted = False
    def fit(self, df):
        import warnings; warnings.filterwarnings('ignore', message='All-NaN slice')
        X = df.reindex(columns=self.features).astype(float).values; self.median = np.where(np.isfinite(np.nanmedian(X, 0)), np.nanmedian(X, 0), 0.0)
        Xi = np.where(np.isfinite(X), X, self.median); self.lo, self.hi = np.percentile(Xi, 0.5, 0), np.percentile(Xi, 99.5, 0); Xc = np.clip(Xi, self.lo, self.hi)
        self.mean, self.std = Xc.mean(0), Xc.std(0) + 1e-9; self.quant = np.percentile(Xi, np.linspace(0, 100, 101), axis=0).T; self.fitted = True; return self
    def raw(self, df): X = df.reindex(columns=self.features).astype(float).values; return np.where(np.isfinite(X), X, self.median)
    def transform(self, df): return ((np.clip(self.raw(df), self.lo, self.hi) - self.mean) / self.std).astype(np.float32)
    def percentile_of(self, feature, value):
        j = self.features.index(feature); return float(np.searchsorted(self.quant[j], value, side="right") - 1)
    def to_dict(self): return dict(features=self.features, median=self.median.tolist(), lo=self.lo.tolist(), hi=self.hi.tolist(), mean=self.mean.tolist(), std=self.std.tolist())


# --- notebook 04, cell 16: make_classifier --------------------
def make_classifier(n_classes, seed=42, kind="auto", n_estimators=300):
    """Returns (estimator, name). Preference order: LightGBM, XGBoost, sklearn HistGradientBoosting."""
    if kind in ("auto", "lightgbm"):
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=15, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                      class_weight="balanced", random_state=seed, verbose=-1, n_jobs=-1), "lightgbm"
        except Exception: pass
    if kind in ("auto", "xgboost"):
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, eval_metric="mlogloss"), "xgboost"
        except Exception: pass
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=0.06, max_leaf_nodes=15, class_weight="balanced", random_state=seed, early_stopping=False), "hist_gradient_boosting"


# --- notebook 04, cell 16: _Wrap --------------------
class _Wrap:
    """Maps arbitrary label values to contiguous classes and always returns probabilities for ALL n_classes columns."""
    def __init__(self, n_classes, seed, kind="auto", n_estimators=300): self.n = n_classes; self.seed = seed; self.kind = kind; self.n_estimators = n_estimators
    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).astype(int); self.classes_ = np.unique(y); self.clf, self.name = make_classifier(len(self.classes_), self.seed, self.kind, self.n_estimators); m = {c: i for i, c in enumerate(self.classes_)}
        yy = np.array([m[v] for v in y])
        if len(self.classes_) < 2: self.const = int(self.classes_[0]); return self
        self.const = None
        try: self.clf.fit(X, yy, sample_weight=sample_weight)
        except TypeError: self.clf.fit(X, yy)
        return self
    def predict_proba(self, X):
        P = np.zeros((len(X), self.n))
        if self.const is not None: P[:, self.const] = 1.0; return P
        p = self.clf.predict_proba(X)
        for k, c in enumerate(self.classes_): P[:, int(c)] = p[:, k]
        return P


# --- notebook 04, cell 16: RiskModel --------------------
class RiskModel:
    """risk level model (4 classes) + motion-state model (7 classes) on scaled window features."""
    def __init__(self, features=None, risk_cfg: RiskConfig | None = None, seed=42, kind="auto"):
        self.features = list(features or MODEL_FEATURES); self.cfg = risk_cfg or RiskConfig(); self.seed = seed; self.kind = kind; self.scaler = FeatureScaler(self.features); self.meta = {}

    def fit(self, train_df, level_col="risk_label", state_col="motion_state", weight_col=None, n_estimators=300):
        self.scaler.fit(train_df); X = self.scaler.transform(train_df); w = train_df[weight_col].values if weight_col and weight_col in train_df else None
        self.level_model = _Wrap(4, self.seed, self.kind, n_estimators).fit(X, train_df[level_col].values, w)
        yS = train_df[state_col].map({s: i for i, s in enumerate(STATES)}).values.astype(int); self.state_model = _Wrap(len(STATES), self.seed, self.kind, n_estimators).fit(X, yS, w)
        self.algorithm = self.level_model.name if hasattr(self.level_model, "name") else "constant"; return self

    def predict(self, df):
        X = self.scaler.transform(df); P = self.level_model.predict_proba(X); S = self.state_model.predict_proba(X)
        score = P @ np.asarray(self.cfg.level_centers); e = self.cfg.level_edges; level = np.digitize(score, e)
        return dict(proba=P, score=score, level=level, level_name=np.array([RISK_LEVELS[i] for i in level]), state_proba=S, state=np.array([STATES[i] for i in S.argmax(1)]))

    # ---- explanation -------------------------------------------------------------------------------------
    def factor_levels(self, row):
        """HIGH / MODERATE / LOW for each factor = percentile of its representative feature among TRAINING windows (measured, not invented)."""
        out = {}
        for name, (rep, phrase, _) in FACTORS.items():
            if rep in self.features and np.isfinite(row.get(rep, np.nan)):
                pct = self.scaler.percentile_of(rep, float(row[rep])); out[name] = dict(feature=rep, value=float(row[rep]), percentile=pct, level="HIGH" if pct >= 85 else "MODERATE" if pct >= 60 else "LOW")
        return out

    def explain(self, row, top_k=None):
        """Occlusion explanation: replace one factor group by its training median and measure the drop of the risk score (works for every model type)."""
        top_k = top_k or self.cfg.top_k_reasons; df = pd.DataFrame([row]); base = float(self.predict(df)["score"][0]); contrib = {}
        for name, (_, _, feats) in FACTORS.items():
            d2 = df.copy()
            for f in feats:
                if f in self.features: d2[f] = self.scaler.median[self.features.index(f)]
            contrib[name] = base - float(self.predict(d2)["score"][0])
        fl = self.factor_levels(row); ranked = sorted(contrib.items(), key=lambda kv: -kv[1]); reasons = []
        for name, c in ranked[:top_k]:
            if c > 0.1 and fl.get(name, {}).get("level", "LOW") != "LOW": reasons.append(dict(factor=name, phrase=FACTORS[name][1], contribution=float(c), level=fl[name]["level"]))
        if not reasons and ranked and ranked[0][1] > 0.1: n0, c0 = ranked[0]; reasons.append(dict(factor=n0, phrase=FACTORS[n0][1], contribution=float(c0), level=fl.get(n0, {}).get("level", "n/a")))
        return dict(score=base, contributions=contrib, factor_levels=fl, reasons=reasons)

    def save(self, path):
        import joblib; os.makedirs(os.path.dirname(path) or ".", exist_ok=True); joblib.dump(self, path)
    @staticmethod
    def load(path):
        import joblib; return joblib.load(path)


# --- notebook 04, cell 16: compass --------------------
def compass(deg):
    names = ["right (east)", "down-right", "down (south)", "down-left", "left (west)", "up-left", "up (north)", "up-right"]
    return names[int(((deg % 360) + 22.5) // 45) % 8]


# --- notebook 04, cell 16: describe_motion --------------------
def describe_motion(row, model: RiskModel):
    fl = model.factor_levels(row); lev = lambda n: fl.get(n, {}).get("level", "n/a"); coh = lev("directional coherence"); sp = lev("speed")
    if sp == "HIGH" and coh == "HIGH": motion = "High-speed coherent movement"
    elif sp == "HIGH": motion = "High-speed disordered movement"
    elif sp == "LOW" and lev("crowd acceleration") != "HIGH": motion = "Slow / near-static movement"
    else: motion = "Moderate movement"
    stab = row.get("dominant_dir_stability", np.nan); dd = row.get("dominant_direction_deg", np.nan)
    direction = "Mixed directions" if (not np.isfinite(stab) or stab < 0.6 or not np.isfinite(dd)) else f"Mostly {compass(dd)}"
    dens = row.get("density", np.nan); dpct = model.scaler.percentile_of("density", float(dens)) if np.isfinite(dens) else np.nan
    dlev = "n/a" if not np.isfinite(dpct) else ("HIGH" if dpct >= 85 else "MODERATE" if dpct >= 60 else "LOW")
    return dict(motion=motion, direction=direction, acceleration=lev("crowd acceleration"), scattering=lev("scattering"), counter_flow=lev("counter-flow"), density=dlev, compression=lev("local compression"))


# --- notebook 04, cell 16: RiskSmoother --------------------
class RiskSmoother:
    """EMA on the score + hysteresis on the level (a level change needs `hysteresis` points of margin) -> stable overlay, fewer flickers."""
    def __init__(self, cfg: RiskConfig): self.cfg = cfg; self.s = None; self.level = None; self.prev = None
    def update(self, score):
        a = self.cfg.smooth_alpha; self.s = score if self.s is None else a * score + (1 - a) * self.s; e, h = self.cfg.level_edges, self.cfg.hysteresis; raw = int(np.digitize(self.s, e))
        if self.level is None: self.level = raw
        elif raw > self.level and self.s >= e[self.level] + h: self.level = raw           # leave the current level upwards only with a margin
        elif raw < self.level and self.s <= e[self.level - 1] - h: self.level = raw       # ... and downwards likewise
        change = 0.0 if self.prev is None else self.s - self.prev; self.prev = self.s; return float(self.s), int(self.level), float(change)


# --- notebook 04, cell 16: smooth_series --------------------
def smooth_series(scores, cfg: RiskConfig):
    sm = RiskSmoother(cfg); out = [sm.update(s) for s in scores]; return np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out])


# --- notebook 04, cell 16: INTERVAL_NAMES --------------------
INTERVAL_NAMES = {0: "NORMAL", 1: "WARNING", 2: "HIGH RISK", 3: "CRITICAL"}


# --- notebook 04, cell 16: INTERVAL_COLORS --------------------
INTERVAL_COLORS = {0: "#2e7d32", 1: "#f9a825", 2: "#ef6c00", 3: "#c62828"}


# --- notebook 04, cell 16: intervals_from_levels --------------------
def intervals_from_levels(levels, min_len=2, min_level=1):
    """Contiguous runs with level >= min_level lasting >= min_len windows -> list of (start_idx, end_idx, peak_level)."""
    out, i, n = [], 0, len(levels)
    while i < n:
        if levels[i] >= min_level:
            j = i
            while j + 1 < n and levels[j + 1] >= min_level: j += 1
            if j - i + 1 >= min_len: out.append((i, j, int(max(levels[i:j + 1]))))
            i = j + 1
        else: i += 1
    return out


# --- notebook 04, cell 36: fill_gaps --------------------
def fill_gaps(flag, max_gap=1):
    """Close gaps of <= max_gap windows inside an abnormal run (one noisy window must not split / end an event)."""
    f = np.asarray(flag, bool).copy(); i, n = 0, len(f)
    while i < n:
        if not f[i]:
            j = i
            while j < n and not f[j]: j += 1
            if 0 < i and j < n and (j - i) <= max_gap: f[i:j] = True
            i = j
        else: i += 1
    return f


# --- notebook 04, cell 29: CHANGE_SPEC --------------------
CHANGE_SPEC = [("mean_speed", "speed"), ("p95_speed", "p95_speed"), ("mean_abs_acceleration", "acceleration"), ("direction_change_rate", "direction"), ("direction_variance", "direction_variance"),
               ("counter_flow_ratio", "counterflow"), ("scattering_score", "scattering"), ("compression_score", "compression"), ("convergence", "convergence"),
               ("crowd_count", "crowd_count"), ("density", "density"), ("packed_density", "packed_density"), ("occupancy", "occupancy"), ("occupied_area_ratio", "occupied_area"),
               ("density_concentration", "concentration"), ("local_density_gradient", "density_gradient")]


# --- notebook 04, cell 29: MOTION_SHORTS --------------------
MOTION_SHORTS = ["speed", "p95_speed", "acceleration", "direction", "direction_variance", "counterflow", "scattering", "compression", "convergence"]


# --- notebook 04, cell 29: CROWD_SHORTS --------------------
CROWD_SHORTS = ["crowd_count", "density", "packed_density", "occupancy", "occupied_area", "concentration", "density_gradient"]


# --- notebook 04, cell 29: RATIO_SHORTS --------------------
RATIO_SHORTS = {"speed", "crowd_count", "density", "occupancy"}


# --- notebook 04, cell 29: _chg --------------------
_chg = lambda shorts: [f"{s}_{k}" for s in shorts for k in ("delta", "change_score", "jump")]


# --- notebook 04, cell 29: TC_FEATURES --------------------
TC_FEATURES = _chg(MOTION_SHORTS) + ["speed_ratio", "speed_zscore", "speed_percentile", "previous_mean_speed", "speed_delta_2", "flow_magnitude_variance", "direction_entropy"]


# --- notebook 04, cell 29: OCC_FEATURES --------------------
OCC_FEATURES = _chg(CROWD_SHORTS) + ["crowd_count_ratio", "density_ratio", "occupancy_ratio", "occupancy_zscore", "density_variance", "crowd_count_rate", "occupancy_rate", "density_rate",
                                       "local_density_gradient", "density_concentration", "crowd_region_area", "occupied_area_ratio", "packed_density"]


# --- notebook 04, cell 29: ChangeModel --------------------
class ChangeModel:
    def __init__(self, k=4): self.k = int(k); self.scale, self.glob, self.fitted = {}, {}, False
    def fit(self, df):
        """FIT ON TRAIN WINDOWS ONLY (natural windows; injected ones are ignored)."""
        d = df[df["label_source"] != "injected"] if "label_source" in df else df
        for col, short in CHANGE_SPEC:
            if col not in d: continue
            v = d[col].values.astype(float); fin = v[np.isfinite(v)]
            if len(fin) < 5: continue
            med = float(np.median(fin)); mad = float(1.4826 * np.median(np.abs(fin - med))); spread = float(np.percentile(fin, 95) - np.percentile(fin, 5)); dd = []
            for _, g in d.sort_values(["scene", "start_frame"]).groupby("scene"):
                x = g[col].values.astype(float); dx = np.diff(x); dd.append(dx[np.isfinite(dx)])
            dd = np.concatenate(dd) if dd else np.zeros(0); step = float(1.4826 * np.median(np.abs(dd - np.median(dd)))) if len(dd) >= 5 else mad
            self.scale[col] = max(step, 0.05 * mad, 0.05 * spread, 1e-6); self.glob[col] = dict(median=med, mad=max(mad, 1e-9), quant=np.percentile(fin, np.linspace(0, 100, 101)).tolist())
        self.fitted = True; return self
    def to_dict(self): return dict(k=self.k, scale=self.scale, glob=self.glob, fitted=self.fitted)
    @classmethod
    def from_dict(cls, d): o = cls(d["k"]); o.scale, o.glob, o.fitted = d["scale"], d["glob"], d.get("fitted", True); return o
    def percentile(self, col, v):
        if col not in self.glob or not np.isfinite(v): return float("nan")
        return float(np.clip(np.searchsorted(self.glob[col]["quant"], v, side="right") - 1, 0, 100))
    def zscore(self, col, v):
        if col not in self.glob or not np.isfinite(v): return 0.0
        g = self.glob[col]; return float(np.clip((v - g["median"]) / max(g["mad"], self.scale.get(col, 1e-6)), -10, 10))
    def compute(self, cur, hist):
        """cur: window-feature dict; hist: list of PREVIOUS windows (dicts) of the same scene/video, oldest first. Only past windows are used."""
        out = {}; H = hist[-self.k:]
        for col, short in CHANGE_SPEC:
            sc = self.scale.get(col); x = cur.get(col, np.nan); hv = np.array([h.get(col, np.nan) for h in H], float); hv = hv[np.isfinite(hv)]
            if sc is None or not np.isfinite(x) or len(hv) == 0:
                out[f"{short}_delta"] = 0.0; out[f"{short}_change_score"] = 0.0; out[f"{short}_jump"] = 0.0
                if short in RATIO_SHORTS: out[f"{short}_ratio"] = 1.0
                if short == "speed": out.update(previous_mean_speed=float(x) if np.isfinite(x) else np.nan, speed_delta_2=0.0)
                continue
            base = float(np.median(hv)); mad = float(1.4826 * np.median(np.abs(hv - base))); delta = float(x - base); out[f"{short}_delta"] = delta; out[f"{short}_change_score"] = float(np.clip(delta / max(mad, sc), -10, 10))
            step = float(1.4826 * np.median(np.abs(np.diff(hv)))) if len(hv) >= 2 else 0.0; out[f"{short}_jump"] = float(np.clip(abs(x - hv[-1]) / max(step, sc), 0, 10))
            if short in RATIO_SHORTS: eps = max(0.05 * abs(self.glob[col]["median"]), 1e-3); out[f"{short}_ratio"] = float((x + eps) / (base + eps))
            if short == "speed": out["previous_mean_speed"] = base; out["speed_delta_2"] = float(x - hv[-2]) if len(hv) >= 2 else 0.0
        out["speed_zscore"] = self.zscore("mean_speed", cur.get("mean_speed", np.nan)); out["speed_percentile"] = self.percentile("mean_speed", cur.get("mean_speed", np.nan)); out["occupancy_zscore"] = self.zscore("occupancy", cur.get("occupancy", np.nan))
        return out


# --- notebook 04, cell 30: SIGNALS --------------------
SIGNALS = ["speed", "acceleration", "direction", "occupancy", "density", "compression", "scattering", "counterflow", "convergence", "temporal_change"]


# --- notebook 04, cell 30: SIGNAL_GROUP --------------------
SIGNAL_GROUP = dict(speed="motion_magnitude", acceleration="motion_magnitude", temporal_change="motion_magnitude", direction="direction", occupancy="crowd_size", density="crowd_size",
                    compression="interaction", convergence="interaction", scattering="dispersion", counterflow="dispersion")


# --- notebook 04, cell 30: ABN_TAGS --------------------
ABN_TAGS = ["NORMAL", "SUDDEN_SPEED_CHANGE", "SUSTAINED_HIGH_SPEED", "SUDDEN_ACCELERATION", "SUDDEN_DIRECTION_CHANGE", "DIRECTION_INSTABILITY", "OCCUPANCY_INCREASE", "OCCUPANCY_DROP",
            "DENSITY_SURGE", "DENSITY_COMPRESSION", "SCATTERING", "COUNTER_FLOW", "CONVERGENCE", "MULTI_SIGNAL_ABNORMAL"]


# --- notebook 04, cell 30: ABN_LEVELS --------------------
ABN_LEVELS = ["NONE", "WEAK", "RELATED", "MULTI_INDEPENDENT"]


# --- notebook 04, cell 30: ABN_FEATURES --------------------
ABN_FEATURES = [f"E_{s}" for s in SIGNALS] + ["abnormality_score", "n_active_signals", "n_active_groups"]


# --- notebook 04, cell 30: signal_scores --------------------
def signal_scores(df):
    """Vectorised robust deviation scores (>= 0, larger = more abnormal) of every signal, from the change-score columns."""
    g = lambda c: np.nan_to_num(df[c].values.astype(float), nan=0.0) if c in df else np.zeros(len(df)); z = pd.DataFrame(index=df.index)
    z["speed"] = np.abs(g("speed_change_score")); z["acceleration"] = np.maximum(g("acceleration_change_score"), 0); z["direction"] = np.maximum(np.maximum(g("direction_change_score"), g("direction_variance_change_score")), 0)
    occ = np.stack([g("crowd_count_change_score"), g("occupancy_change_score"), g("occupied_area_change_score")], 1); sgn = occ[np.arange(len(occ)), np.abs(occ).argmax(1)]; z["occupancy"] = np.abs(sgn); z["occ_signed"] = sgn
    z["density"] = np.maximum(np.stack([g("packed_density_change_score"), g("concentration_change_score"), g("density_gradient_change_score")], 1).max(1), 0)
    for s in ("compression", "scattering", "counterflow", "convergence"): z[s] = np.maximum(g(f"{s}_change_score"), 0)
    z["temporal_change"] = np.stack([g("speed_jump"), g("acceleration_jump"), g("direction_jump")], 1).max(1); return z


# --- notebook 04, cell 30: AbnormalityDetector --------------------
class AbnormalityDetector:
    def __init__(self, cm=None):
        self.cm = cm; self.thr = {s: 3.0 for s in SIGNALS}; self.rel = {s: 0.6 for s in SIGNALS}; self.min_signals_related = 2; self.min_groups_multi = 3; self.sustained_pct = 90.0; self.info = {}
    def to_dict(self): return dict(thr=self.thr, rel=self.rel, min_signals_related=self.min_signals_related, min_groups_multi=self.min_groups_multi, sustained_pct=self.sustained_pct, info=json_safe(self.info))
    @classmethod
    def from_dict(cls, d, cm=None):
        o = cls(cm); o.thr, o.rel, o.min_signals_related, o.min_groups_multi, o.sustained_pct, o.info = d["thr"], d["rel"], d["min_signals_related"], d["min_groups_multi"], d["sustained_pct"], d.get("info", {}); return o
    def evaluate_df(self, df):
        z = signal_scores(df); n = len(df); out = pd.DataFrame(index=df.index); act = {}; keep = np.ones(n)
        for s in SIGNALS:
            lo, hi = 0.5 * self.thr[s], 1.5 * self.thr[s]; E = np.clip((z[s].values - lo) / (hi - lo), 0, 1); out[f"E_{s}"] = E; act[s] = z[s].values >= self.thr[s]; out[f"active_{s}"] = act[s]; keep = keep * (1 - self.rel[s] * E)
        out["abnormality_score"] = 1 - keep; A = np.stack([act[s] for s in SIGNALS], 1); out["n_active_signals"] = A.sum(1).astype(int)
        groups = sorted(set(SIGNAL_GROUP.values())); out["n_active_groups"] = np.stack([np.any(np.stack([act[s] for s in SIGNALS if SIGNAL_GROUP[s] == gname], 1), 1) for gname in groups], 1).sum(1).astype(int)
        lvl = np.where(out["n_active_signals"] >= 1, 1, 0); lvl = np.where(out["n_active_signals"] >= self.min_signals_related, 2, lvl); lvl = np.where(out["n_active_groups"] >= self.min_groups_multi, 3, lvl); out["abnormality_level"] = lvl.astype(int)
        sp = df["mean_speed"].values.astype(float) if "mean_speed" in df else np.full(n, np.nan); ps = df["previous_mean_speed"].values.astype(float) if "previous_mean_speed" in df else np.full(n, np.nan)
        pc = np.array([self.cm.percentile("mean_speed", v) if self.cm else np.nan for v in sp]); pp = np.array([self.cm.percentile("mean_speed", v) if self.cm else np.nan for v in ps])
        out["sustained_high_speed"] = (np.nan_to_num(pc, nan=0) >= self.sustained_pct) & (np.nan_to_num(pp, nan=0) >= self.sustained_pct); tags = []
        for i in range(n):
            a = {s: bool(act[s][i]) for s in SIGNALS}; t = []; sudden_speed = a["speed"] and a["temporal_change"]; dj = float(df["direction_jump"].iloc[i]) if "direction_jump" in df and np.isfinite(df["direction_jump"].iloc[i]) else 0.0
            if sudden_speed: t.append("SUDDEN_SPEED_CHANGE")
            if out["sustained_high_speed"].iloc[i] and not sudden_speed: t.append("SUSTAINED_HIGH_SPEED")
            if a["acceleration"]: t.append("SUDDEN_ACCELERATION")
            if a["direction"]: t.append("SUDDEN_DIRECTION_CHANGE" if dj >= self.thr["temporal_change"] else "DIRECTION_INSTABILITY")
            if a["occupancy"]: t.append("OCCUPANCY_INCREASE" if z["occ_signed"].iloc[i] > 0 else "OCCUPANCY_DROP")
            if a["density"]: t.append("DENSITY_SURGE")
            if a["compression"]: t.append("DENSITY_COMPRESSION")
            if a["scattering"]: t.append("SCATTERING")
            if a["counterflow"]: t.append("COUNTER_FLOW")
            if a["convergence"]: t.append("CONVERGENCE")
            if out["n_active_groups"].iloc[i] >= self.min_groups_multi: t.append("MULTI_SIGNAL_ABNORMAL")
            tags.append("|".join(t) if t else "NORMAL")
        out["abn_tags"] = tags; return out
    def evaluate_row(self, w): return self.evaluate_df(pd.DataFrame([w])).iloc[0].to_dict()
    # ---- fitting / tuning ------------------------------------------------------------------------------------------------
    def _active_df(self, df): z = signal_scores(df); return pd.DataFrame({s: z[s].values >= self.thr[s] for s in SIGNALS}, index=df.index)
    def fit_reliability(self, df, truth):
        """reliability_i = (TP+1)/(predicted+2) of signal i against its derived truth on TRAIN windows (clipped to [0.2, 0.95])."""
        act = self._active_df(df)
        for s in SIGNALS:
            t = truth[f"t_{s}"].reindex(df.index).values.astype(float); m = np.isfinite(t)
            if m.sum() < 5: continue
            a = act[s].values[m]; tp = float((a & (t[m] > 0.5)).sum()); self.rel[s] = float(np.clip((tp + 1) / (a.sum() + 2), 0.2, 0.95))
    def tune(self, df, truth, grid=(1.5, 2.0, 2.5, 3.0, 4.0, 5.0), min_gain=0.02):
        """VALIDATION ONLY. Per-signal z threshold = best F1 against the derived signal truth (the default 3.0 is kept unless the gain is >= min_gain), then the level cut-offs."""
        from sklearn.metrics import f1_score
        z = signal_scores(df); self.info["thr"] = {}
        for s in SIGNALS:
            t = truth[f"t_{s}"].reindex(df.index).values.astype(float); m = np.isfinite(t)
            if m.sum() < 10 or (t[m] > 0.5).sum() < 3 or (t[m] < 0.5).sum() < 3: self.info["thr"][s] = dict(used=self.thr[s], note="not tuned (too few truth positives/negatives on validation)"); continue
            f = {g: f1_score(t[m] > 0.5, z[s].values[m] >= g, zero_division=0) for g in grid}; best = max(f, key=f.get); f0 = f1_score(t[m] > 0.5, z[s].values[m] >= 3.0, zero_division=0)
            if f[best] >= f0 + min_gain: self.thr[s] = float(best)
            self.info["thr"][s] = dict(default_f1=float(f0), best_f1=float(f[best]), used=self.thr[s])
        tn = truth[[f"t_{s}" for s in SIGNALS]].reindex(df.index).fillna(0).values.sum(1); y = tn >= 2; best = (self.min_signals_related, self.min_groups_multi); f_best = -1; f_def = None
        if y.sum() >= 3 and (~y).sum() >= 3:
            for ms in (2, 3):
                for mg in (2, 3, 4):
                    self.min_signals_related, self.min_groups_multi = ms, mg; f = f1_score(y, self.evaluate_df(df)["abnormality_level"].values >= 2, zero_division=0)
                    if (ms, mg) == (2, 3): f_def = f
                    if f > f_best + 1e-9: f_best, best = f, (ms, mg)
            self.min_signals_related, self.min_groups_multi = best if f_best >= (f_def or 0) + min_gain else (2, 3); self.info["levels"] = dict(default_f1=float(f_def or 0), best_f1=float(f_best), used=(self.min_signals_related, self.min_groups_multi))
        else: self.min_signals_related, self.min_groups_multi = 2, 3; self.info["levels"] = dict(note="not tuned (too few abnormal/normal validation windows)", used=(2, 3))


# ============================================================================================================================
# BACKEND-ONLY: loading models/risk_model.pkl
# ============================================================================================================================
def load_risk_bundle(path):
    """Load the joblib bundle written by notebook 04 cell 42:
         dict(risk_model=RiskModel, fusion_mode=str, config_name=str, meta=dict)

    The notebook pickled the objects while the classes lived in the notebook's `__main__` namespace, so the pickle stores the
    references `__main__.RiskModel`, `__main__.FeatureScaler`, `__main__._Wrap`, `__main__.RiskConfig`, ...
    We register our (verbatim) classes under those names while loading, then remove the temporary aliases again."""
    import sys, joblib
    from features import FeatureConfig
    main = sys.modules["__main__"]
    needed = dict(RiskModel=RiskModel, FeatureScaler=FeatureScaler, _Wrap=_Wrap, RiskConfig=RiskConfig, FeatureConfig=FeatureConfig,
                  ChangeModel=ChangeModel, AbnormalityDetector=AbnormalityDetector, RiskSmoother=RiskSmoother)
    added = [k for k in needed if not hasattr(main, k)]
    for k in added: setattr(main, k, needed[k])
    try:
        bundle = joblib.load(path)
    finally:
        for k in added:
            try: delattr(main, k)
            except Exception: pass
    for key in ("risk_model", "fusion_mode"):
        if key not in bundle: raise RuntimeError(f"{path}: not a risk_model.pkl written by notebook 04 (missing key '{key}'; found {list(bundle)[:6]})")
    return bundle
