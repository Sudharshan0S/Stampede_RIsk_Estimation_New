"""
End-to-end inference engine  (Notebook 04 cell 47):
   frame stream -> count + head points + density -> PREDICTED flow -> trajectories -> window features -> change scores -> abnormality evidence
   -> risk model -> smoothing -> persistence -> events.
Verbatim copies of explain_window, RiskStreamer, group_events, event_text.  No ground-truth flow / masks / trajectories / labels are used.
"""
import numpy as np
import pandas as pd

from features import FeatureConfig, StreamExtractor, RISK_LEVELS
from risk_model import RiskSmoother, describe_motion, fill_gaps, intervals_from_levels


# --- notebook 04, cell 47: TAG_PHRASE --------------------
TAG_PHRASE = {"NORMAL": "normal", "SUDDEN_SPEED_CHANGE": "SUDDEN SPEED CHANGE", "SUSTAINED_HIGH_SPEED": "SUSTAINED HIGH SPEED", "SUDDEN_ACCELERATION": "SUDDEN ACCELERATION", "SUDDEN_DIRECTION_CHANGE": "SUDDEN DIRECTION CHANGE",
              "DIRECTION_INSTABILITY": "DIRECTION INSTABILITY", "OCCUPANCY_INCREASE": "OCCUPANCY INCREASE", "OCCUPANCY_DROP": "OCCUPANCY DROP", "DENSITY_SURGE": "DENSITY SURGE", "DENSITY_COMPRESSION": "DENSITY COMPRESSION",
              "SCATTERING": "SCATTERING", "COUNTER_FLOW": "COUNTER FLOW", "CONVERGENCE": "CONVERGENCE", "MULTI_SIGNAL_ABNORMAL": "MULTI-SIGNAL ABNORMAL"}


# --- notebook 04, cell 47: _fin --------------------
def _fin(x): return x is not None and np.isfinite(x)


# --- notebook 04, cell 47: explain_window --------------------
def explain_window(r, det):
    """Human-readable evidence for the abnormal signals of ONE window, generated from the measured numbers."""
    out = []; a = lambda s: bool(r.get(f"active_{s}", False)); g = lambda k, d=np.nan: float(r.get(k, d)) if _fin(r.get(k, d)) else d; thr = det.thr
    if a("speed"):
        ratio = g("speed_ratio", 1.0); ratio = max(ratio, 1 / max(ratio, 1e-6)); up = g("speed_delta", 0) >= 0
        out.append(f"crowd speed {'increased' if up else 'dropped'} {ratio:.1f}x versus its recent baseline ({g('previous_mean_speed'):.2f} -> {g('mean_speed'):.2f} px/frame)")
    if a("temporal_change"): out.append("the change was abrupt (a sharp step, not a slow drift)")
    if a("acceleration"): out.append(f"acceleration exceeded the validated threshold ({g('acceleration_change_score', 0):.1f} robust deviations above the recent norm; limit {thr['acceleration']:.1f})")
    if a("direction"): out.append(f"movement direction became unstable (direction change score {max(g('direction_change_score', 0), g('direction_variance_change_score', 0)):.1f}; direction variance {g('direction_variance'):.2f})")
    if a("occupancy"):
        c1, d1 = g("crowd_count"), g("crowd_count_delta", 0)
        out.append(f"crowd count {'rose' if g('crowd_count_change_score', 0) + g('occupancy_change_score', 0) >= 0 else 'fell'} quickly" + (f" ({c1 - d1:.0f} -> {c1:.0f} people)" if _fin(c1) and _fin(d1) and abs(d1) > 0 else "") + " versus the recent baseline")
    if a("density"): out.append(f"people are packing more tightly / unevenly (density change score {g('packed_density_change_score', 0):.1f}, concentration {g('density_concentration'):.2f})")
    if a("compression"): out.append(f"crowd compression increased ({g('compression_change_score', 0):.1f} robust deviations above normal)")
    if a("scattering"): out.append(f"people are moving apart / scattering ({g('scattering_change_score', 0):.1f} robust deviations above normal)")
    if a("counterflow"): out.append(f"opposing-direction movement rose to {100 * g('counter_flow_ratio', 0):.0f}% of the flow")
    if a("convergence"): out.append("movement is converging toward a common point")
    if bool(r.get("sustained_high_speed", False)) and not a("speed"): out.append("speed has stayed high for several windows")
    return out


# --- notebook 04, cell 47: RiskStreamer --------------------
class RiskStreamer:
    """Stateful, frame-by-frame inference. push() returns a dict once per analysis step; ['window'] is filled when a temporal window completes."""
    def __init__(self, bundle, est, count_model, count_every=5):
        self.model, self.cm, self.det, self.cfg, self.fusion = bundle["risk_model"], bundle["change_model"], bundle["detector"], FeatureConfig.from_dict(bundle["feature_config"]), bundle["fusion_mode"]
        self.est, self.count, self.count_every = est, count_model, count_every; self.eff_fps = float(self.cfg.fps); self.reset()
    def reset(self):
        self.ex = StreamExtractor(self.est, self.cfg); self.hist, self.rows = [], []; self.smoother = RiskSmoother(self.model.cfg); self.n = 0; self.cinfo = dict(count=None, density=None, points=None); self.raw_flags = []; self.online_on = False; self.last_window = None
    def push(self, rgb, t_s):
        if self.count is not None and self.n % self.count_every == 0:
            r = self.count.predict(rgb, frame_id=self.n); self.cinfo = dict(count=r["count"], density=r["density_map"], points=r["head_points"])
        self.n += 1; out = self.ex.push(rgb, count=self.cinfo["count"], density=self.cinfo["density"], points=self.cinfo["points"])
        if out is None: return None
        res = dict(flow=out["flow"], region=out["region"], tracks=out.get("tracks"), window=None, count_info=self.cinfo)
        if out["window"] is None: return res
        w = dict(out["window"]); w.update(scene="video", series="video", time_s=float(t_s), start_s=float(t_s) - self.cfg.window_len / self.eff_fps); prev = self.hist[-1] if self.hist else None
        for new, c in (("d_speed_prev", "mean_speed"), ("d_accel_prev", "mean_abs_acceleration"), ("d_density_prev", "density"), ("d_coherence_prev", "direction_coherence"), ("d_counter_prev", "counter_flow_ratio"), ("d_compression_prev", "compression_score")):
            w[new] = float(w[c] - prev[c]) if prev is not None and _fin(w.get(c)) and _fin(prev.get(c)) else 0.0             # identical to add_temporal_context used in training
        w.update(self.cm.compute(w, self.hist)); self.hist.append(dict(w)); abn = self.det.evaluate_df(pd.DataFrame([w])).iloc[0].to_dict(); w.update({k: (v.item() if hasattr(v, "item") else v) for k, v in abn.items()})
        pr = self.model.predict(pd.DataFrame([w])); raw = float(pr["score"][0]); sc, lv, ch = self.smoother.update(raw); dm = describe_motion(w, self.model); fl = self.model.factor_levels(w)
        a_ok = w["abnormality_level"] >= 2; r_ok = lv >= 1; flag = r_ok if self.fusion == "RISK_ONLY" else (r_ok or a_ok) if self.fusion == "RISK_OR_ABN" else (r_ok and a_ok); self.raw_flags.append(bool(flag)); f3 = self.raw_flags[-3:]
        self.online_on = (sum(f3) >= 2) if not self.online_on else not (len(f3) >= 2 and not any(f3[-2:]))             # persistence: ON after 2 of 3 windows, OFF after 2 clear windows
        cnt = w.get("crowd_count"); opct = self.cm.percentile("crowd_count", cnt) if _fin(cnt) else self.cm.percentile("occupancy", w.get("occupancy", np.nan)); occ_lvl = "n/a" if not _fin(opct) else "HIGH" if opct >= 85 else "MODERATE" if opct >= 60 else "LOW"
        state = str(pr["state"][0]) if (lv >= 1 or w["abnormality_level"] >= 2) else "NORMAL"; tags = [t for t in str(w["abn_tags"]).split("|")]
        w.update(risk_score_raw=raw, risk_score=sc, risk_change=ch, risk_level_idx=int(lv), risk_level=RISK_LEVELS[lv], p_high=float(pr["proba"][0][2:].sum()), motion_state=state, motion_state_model=str(pr["state"][0]), state_confidence=float(pr["state_proba"][0].max()),
                 motion_level=fl.get("speed", {}).get("level", "n/a"), motion=dm["motion"], direction=dm["direction"], acceleration_level=dm["acceleration"], scattering_level=dm["scattering"], counter_flow_level=dm["counter_flow"], density_level=dm["density"],
                 occupancy_level=occ_lvl, abnormal_online=bool(self.online_on), alarm_raw=bool(flag), tags=tags, evidence=explain_window(w, self.det), count=cnt)
        try: ex_ = self.model.explain(w); w["reasons"] = "; ".join(f"{r['phrase']} ({r['level'].lower()})" for r in ex_["reasons"]) or "no strong abnormal-motion factor"
        except Exception: w["reasons"] = ""
        self.rows.append(w); self.last_window = w; res["window"] = w; return res
    def timeline(self): return pd.DataFrame([{k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()} for r in self.rows])


# --- notebook 04, cell 47: group_events --------------------
def group_events(TL, rows, det, cm, rc, fusion_mode):
    """Consecutive alarm windows -> ONE event (gaps of 1 window are closed; shorter than min_event_windows is dropped)."""
    if len(TL) == 0: return []
    flag = fill_gaps(TL["alarm_raw"].values.astype(bool), 1); ev = []
    for a, b, _ in intervals_from_levels(flag.astype(int), rc.min_event_windows, 1):
        seg = [rows[i] for i in range(a, b + 1)]; S = TL.iloc[a:b + 1]; pk = int(np.argmax(0.5 * S["risk_score"].values + 50 * S["abnormality_score"].values)); pkw = seg[pk]
        before = [rows[i] for i in range(max(0, a - 4), a)] or [rows[a]]; base_speed = float(np.nanmedian([w["mean_speed"] for w in before])); tagc = {}
        for w in seg:
            for t in w["tags"]: tagc[t] = tagc.get(t, 0) + 1
        tagc_nn = {k: v for k, v in tagc.items() if k not in ("NORMAL",)}; dominant = max((k for k in tagc_nn if k != "MULTI_SIGNAL_ABNORMAL"), key=lambda k: tagc_nn[k], default=(max(tagc_nn, key=tagc_nn.get) if tagc_nn else "NORMAL"))
        sp_pct = [cm.percentile("mean_speed", w["mean_speed"]) for w in seg]; pre_pct = [cm.percentile("mean_speed", w["mean_speed"]) for w in before]; sudden = [i for i, w in enumerate(seg) if ("SUDDEN_ACCELERATION" in w["tags"] or "SUDDEN_SPEED_CHANGE" in w["tags"])]
        rush = bool(sudden and all(np.nan_to_num(p, nan=0) < det.sustained_pct for p in pre_pct) and any(np.nan_to_num(sp_pct[j], nan=0) >= det.sustained_pct for j in range(sudden[0], len(seg))) and (len(seg) - sudden[0]) >= 2)
        c0, c1, cpk = before[0].get("count", np.nan), seg[-1].get("count", np.nan), np.nanmax([w.get("count", np.nan) for w in seg]) if any(_fin(w.get("count", np.nan)) for w in seg) else np.nan
        e = dict(start_s=float(seg[0]["start_s"]), end_s=float(seg[-1]["time_s"]), peak_time_s=float(pkw["time_s"]), n_windows=len(seg), peak_risk_score=float(S["risk_score"].max()), peak_risk=RISK_LEVELS[int(S["risk_level_idx"].max())],
                 peak_abnormality=float(S["abnormality_score"].max()), max_independent_signals=int(S["n_active_groups"].max()), tags={k: v for k, v in sorted(tagc_nn.items(), key=lambda kv: -kv[1])}, dominant_behaviour=dominant, rush_pattern=rush,
                 count_before=float(c0) if _fin(c0) else np.nan, count_peak=float(cpk) if _fin(cpk) else np.nan, count_end=float(c1) if _fin(c1) else np.nan, density_before=float(before[0].get("density", np.nan)), density_peak=float(np.nanmax([w.get("density", np.nan) for w in seg])),
                 occupancy_before=float(before[0].get("occupancy", np.nan)), occupancy_peak=float(np.nanmax([w.get("occupancy", np.nan) for w in seg])), speed_before=base_speed, speed_peak=float(np.nanmax([w["mean_speed"] for w in seg])), evidence=list(pkw["evidence"]),
                 reasons_model=pkw.get("reasons", ""), confidence=float(np.clip(0.5 * S["abnormality_score"].max() + 0.5 * S["p_high"].max(), 0, 1)))
        e["explanation"] = event_text(e); ev.append(e)
    return ev


# --- notebook 04, cell 47: event_text --------------------
def event_text(e):
    lines = ["ABNORMAL BEHAVIOR DETECTED", "", f"Time: {e['start_s']:.1f}s - {e['end_s']:.1f}s   (peak at {e['peak_time_s']:.1f}s, {e['n_windows']} windows)", "", "Tags:"] + [f"- {k}" for k in e["tags"]] + (["- ABNORMAL_RUSH_PATTERN  (normal speed -> sudden acceleration -> sustained high speed)"] if e["rush_pattern"] else [])
    lines += ["", "Evidence:"] + [f"- {x}" for x in e["evidence"]] + [f"- {e['max_independent_signals']} independent signal group(s) active at the peak"]
    if np.isfinite(e["count_before"]) and np.isfinite(e["count_peak"]): lines += ["", "Crowd count:", f"{e['count_before']:.0f} -> {e['count_peak']:.0f}"]
    lines += ["", "Risk:", f"{e['peak_risk']} ({e['peak_risk_score']:.0f}/100)", "", "Confidence:", f"{e['confidence']:.2f}  (evidence strength; NOT a calibrated probability of a real stampede)"]; return "\n".join(lines)
