# CrowdVision AI - AI based stampede-risk estimation

React (Vite + Tailwind + Recharts) frontend + Flask backend. The backend runs the pipeline of the four notebooks:

| Notebook | Question | Writes (to `crowdrisk_outputs/models/`) | Backend module |
|---|---|---|---|
| `01_Crowd_Count_Model.ipynb` | how many people, where? | `crowd_count_model.pth` | `crowd_count.py` |
| `02_Motion_Model.ipynb` | how do they move? | `flow_selection.json` (`motion_model.pth` if MotionNetV2 wins) | `motion.py` |
| `03_Abnormal_Motion_Detection.ipynb` | is the motion abnormal? | `abnormal_motion.json`, `feature_config.json`, `label_rules.json` | `features.py`, `risk_model.py` (ChangeModel, AbnormalityDetector) |
| `04_Risk_Analysis_and_Video_Inference.ipynb` | is it risky, when, why? | `risk_model.pkl` | `risk_model.py`, `pipeline.py`, `predictor.py` |

## 1. Where to put what

```
project/
├── models/        <- COPY  crowdrisk_outputs/models/*   HERE   (see models/README.md for the exact file list)
├── backend/       Flask API + the notebook code (do not move files out of it)
├── frontend/      React UI
├── notebooks/     the four notebooks (training side)
└── outputs/       created automatically at run time (annotated media, JSON, CSV, reports, charts, .run_manifest.json); safe to delete
```

* Required: `crowd_count_model.pth`, `flow_selection.json`, `abnormal_motion.json`, `risk_model.pkl`  
* Only if `flow_selection.json` says `"motionnet_v2"`: `motion_model.pth`  
* NOT needed for serving: `crowdrisk_outputs/outputs/*` and `crowdrisk_outputs/work/*`

## 2. Run

```bash
# backend  (Python >= 3.10)
cd backend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # CPU torch is fine:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python app.py                                            # http://localhost:5000

# frontend (second terminal)
cd frontend
npm install
npm run dev                                              # http://localhost:5173
```

To try it on a phone in the same Wi-Fi: `npm run dev -- --host`, open `http://<your-PC-ip>:5173`, and start the backend with `CORS_ORIGINS` unset (any origin) or including that address.

Check the models: open <http://localhost:5000/models/status> - it lists every expected file (found / missing, which notebook writes it) and what was loaded
(count variant, flow estimator, risk configuration, alarm rule, warnings). If a file is missing the server still starts and the UI shows a banner.

Optional environment variables: `CROWDRISK_MODELS_DIR`, `CROWDRISK_OUTPUT_DIR`, `CROWDRISK_DEVICE` (`auto`|`cpu`), `CROWDRISK_COUNT_EVERY` (default 5),
`CROWDRISK_MAX_SECONDS` (analyse only the first N seconds), frontend: `VITE_API_BASE_URL`.

## 3. How the backend matches the notebooks

Every analysis definition in `backend/` was **extracted programmatically (verbatim) from the notebooks** - the 01/02/03/04 copies of each function are byte-identical, and
each block is tagged `# --- notebook NN, cell M: name` in the source. Nothing was re-implemented. In particular:

| Aspect | Where it is defined (identical in training and serving) |
|---|---|
| Count-model input: long side clamped to `[min_long, max_long]`, INTER_AREA / INTER_CUBIC resize, exposure normalisation (`normalize_exposure`), ImageNet mean/std, pad to a multiple of 32, tiling with overlap, anchors, softmax point head + density branch | `crowd_count.py` (`CrowdPredictor._prep/_run/_block`, `to_tensor`, `CrowdPointNet`) |
| Point post-processing: min candidate confidence, accept threshold, uncertain band, distance NMS, regional re-analysis (up-scale crop, merge policy) | `crowd_count.py` (`finalize_points`, `regional_discrepancy`, `CrowdPredictor.refine`) - all settings come from the checkpoint (`ck["infer"]`) |
| Layers | `crowd_count.py` `CrowdPointNet` / `build_model(spec)` - the spec stored in the checkpoint rebuilds the exact network; weights load with `strict` `load_state_dict` |
| Video frame handling: analysis stride `round(native_fps / cfg.fps)`, resize to the working width (`work_size`, INTER_AREA), BGR->RGB, count every `COUNT_EVERY` steps | `predictor.py::predict_video` (port of notebook 04 `run_video`) |
| Optical flow: estimator + width from `flow_selection.json` (DIS @ 640 for your run) | `motion.py` |
| Ego-motion compensation, crowd region, 36 per-step statistics, flow-advected trajectories, window features (window 20, hop 5) | `features.py` (`StreamExtractor`, `step_features`, `track_features`, `window_features`) |
| Change scores vs. the previous K windows, 10 evidence signals, noisy-OR, thresholds, tags | `risk_model.py` (`ChangeModel`, `signal_scores`, `AbnormalityDetector`) - parameters come from `abnormal_motion.json` |
| Risk model (LightGBM level + motion-state), feature scaling, validation-tuned cut-points, occlusion explanations, EMA + hysteresis | `risk_model.py` (`RiskModel`, `FeatureScaler`, `RiskSmoother`) - loaded from `risk_model.pkl` |
| Alarm rule (`RISK_AND_ABN` for your run), persistence (on 2 of 3, off after 2 clear), gap filling, event grouping, rush pattern, explanations | `pipeline.py` (`RiskStreamer`, `group_events`, `event_text`) |
| Video HUD and diagnostics figures | `visualization.py` (`draw_overlay_v2`, `plot_full_timeline`, 10 diagnostics of notebook 04 cell 54) |

Only deliberate differences: models load from `models/` instead of notebook globals; `run_video` reports progress / supports cancel / writes web-ready H.264; NaN values are turned
into `null` for JSON; a guard replaces an invalid (0/NaN) FPS reading by the FPS assumption; `risk_model.pkl` is unpickled with the classes registered as `__main__.*`
(it was pickled from a notebook).

Check it any time: `cd backend && python verify_notebook_sync.py` compares all 112 notebook-derived blocks with `notebooks/*.ipynb` (exit code 0 = identical).

**If you retrain** a notebook, copy the new files over the old ones (all four from the same run) and re-run notebooks 03 and 04 after changing 01/02. If you change a
function inside a notebook, apply the same change in the matching backend file - they must stay identical.

## 4. What the app returns

* **Video** - the full pipeline: per-window timeline (risk score / level, abnormality score + 10 evidence signals, tags, motion state, count, density, speed, acceleration,
  direction, compression, scattering, counter-flow), **abnormal events** (start/end, peak, tags, evidence, count/density/speed change, confidence), annotated MP4 (flow arrows,
  head points, HUD), timeline + 10-panel diagnostics PNG, JSON / CSV / TXT / PDF.
* **Image** - crowd count (accepted head points), uncertain points + count interval, density-map sum, density heat-map, 2x2 panel, crowd regime
  (sparse / medium / dense / extreme from the TRAIN-fitted edges). A single image has no motion, so **no stampede-risk score is produced for images** (the notebooks'
  risk model needs motion windows).

Window unit: one analysis step = one frame at `round(native_fps / 25)` stride, so a window of 20 steps ≈ 0.8 s of 25 fps video; speeds are in pixels per step at the
640 px working width. **Risk and abnormality labels are DERIVED (weak) from TUB CrowdFlow ground-truth motion + synthetic injections - they are not official stampede
labels**; `confidence` is evidence strength, not a probability (see the notebooks' limitations).

## 5. Speed

Counting runs the full model (tiling + regional refinement) every 5th step; on CPU the notebook processed about one analysis step per second at 640x360. Use
`CROWDRISK_MAX_SECONDS` for quick tests, a GPU (`CROWDRISK_DEVICE=auto`) for speed, or raise `CROWDRISK_COUNT_EVERY` (this changes the count refresh rate vs. the notebook).

## 6. Deployment (backend and frontend are deployed separately)

```
Browser ──► Vercel (static React build, VITE_API_BASE_URL) ──HTTPS + CORS──► Render web service (gunicorn + Flask + PyTorch + models on a persistent disk)
```

The ML backend (PyTorch, OpenCV, several hundred MB of models, minutes-long CPU analyses, in-process progress state) is **not** suitable for Vercel serverless functions - deploy it
on a host that runs a long-lived Python process (Render, Railway, Fly.io, a VM, ...). Only the static frontend goes to Vercel.

### 6.1 Backend on Render (blueprint: `render.yaml`)
1. Push the repository, then *New -> Blueprint* in Render and select it (or create a *Web Service* by hand with the values below). `render.yaml` defines:
   * root directory `backend`, Python 3.11, build = CPU-only PyTorch + `requirements.txt`, start = `gunicorn -c gunicorn.conf.py app:app`, health check `/health`
   * a **persistent disk** mounted at `/var/data` (models + results survive restarts / deploys)
   * env vars `CROWDRISK_MODELS_DIR=/var/data/models`, `CROWDRISK_OUTPUT_DIR=/var/data/outputs`, `CROWDRISK_DEVICE=cpu`, `CORS_ORIGINS` (you are asked for it)
2. **Put the trained models on the disk** (they are not in the repository): open a shell/SSH to the service (Render dashboard -> *Shell* / *SSH*), `mkdir -p /var/data/models`, then copy the four required files
   from `crowdrisk_outputs/models/` (`crowd_count_model.pth`, `flow_selection.json`, `abnormal_motion.json`, `risk_model.pkl`; plus `motion_model.pth` only if `flow_selection.json` says `motionnet_v2`)
   with `scp`/`rsync`/SFTP as described in Render's SSH documentation, or download them from your own private storage with `curl`/`wget`. Restart the service afterwards.
3. Open `https://<your-service>.onrender.com/models/status` - it must show `"ready": true`.
4. Set `CORS_ORIGINS` to the frontend origin(s), e.g. `https://crowdvision.vercel.app` (comma separated; regular expressions are allowed, e.g. `https://crowdvision-.*\.vercel\.app` for preview deployments).

Sizing / behaviour you must plan for:
* **RAM**: PyTorch + models need roughly 2 GB or more - small/free plans will be killed (out of memory). Verify on your plan.
* **Exactly one gunicorn worker** (set in `gunicorn.conf.py`): analysis lock, `/progress` and the loaded models live in the process. Scale up (CPU/RAM/GPU), not out.
* **Long requests**: `/predict/video` is one HTTP request that lasts as long as the analysis (about 1 analysis step per second on CPU at 640x360). Your host/proxy must allow requests of that length
  (check its request-timeout limit); otherwise analyse short clips, set `CROWDRISK_MAX_SECONDS`, or use a faster/GPU machine. `GUNICORN_TIMEOUT` only governs gunicorn itself.
* Models load once at start-up (tens of seconds); a sleeping/cold-started instance answers `/health` only after loading.

Other hosts: `Procfile` (`web: cd backend && gunicorn -c gunicorn.conf.py app:app`) works on Railway/Heroku-style platforms; on a plain VM run
`cd backend && PORT=8000 CORS_ORIGINS=https://your-frontend gunicorn -c gunicorn.conf.py app:app` behind nginx/Caddy (allow large uploads and long timeouts, e.g. `client_max_body_size 2g; proxy_read_timeout 1800;`).
gunicorn does not run on Windows - there use `python app.py` (development server) or another WSGI server such as waitress.

### 6.2 Frontend on Vercel (`frontend/vercel.json`)
1. *Add New Project* -> import the repository -> **Root Directory: `frontend`** (framework Vite is detected; build `npm run build`, output `dist`).
2. Environment variable (Production **and** Preview): `VITE_API_BASE_URL=https://<your-service>.onrender.com` (no trailing slash; it is baked in at build time - redeploy after changing it).
3. Deploy. The app header must show *Backend: Connected*. If it shows *Offline*, check `VITE_API_BASE_URL`, that the backend is awake, and `CORS_ORIGINS`. Both URLs must be HTTPS (browsers block HTTP API calls from an HTTPS page).

### 6.3 Environment variables
| Where | Variable | Meaning (default) |
|---|---|---|
| backend | `CROWDRISK_MODELS_DIR` | folder with the model files (`<project>/models`) |
| backend | `CROWDRISK_OUTPUT_DIR` | folder for generated results (`<project>/outputs`) |
| backend | `CROWDRISK_DEVICE` | `auto` (GPU if available) or `cpu` |
| backend | `CROWDRISK_COUNT_EVERY` / `CROWDRISK_MAX_SECONDS` | count-model refresh interval (5) / analyse only the first N seconds (all) |
| backend | `PORT` / `HOST` | listen port set by the platform (5000) / bind address for `python app.py` (0.0.0.0) |
| backend | `CORS_ORIGINS` | allowed frontend origin(s), comma separated, regex allowed (`*` = any, development only) |
| backend | `MAX_UPLOAD_MB` | upload limit (2048) |
| backend | `WEB_THREADS` / `GUNICORN_TIMEOUT` | gunicorn threads (8) / worker timeout in seconds (900) |
| frontend | `VITE_API_BASE_URL` | backend origin; dev default `http://localhost:5000`; **required** for production builds |

## 7. Result files and safe replacement of the previous result

Only the latest successful analysis is kept on disk (as before), but the replacement is now crash-safe and independent of server memory:

* Every file of a run carries the run id (32 hex characters): annotated image/video, density/panel images, JSON, CSV, events files, TXT/PDF reports, charts, diagnostics.
* `outputs/.run_manifest.json` (written atomically, never served by `/outputs`) records the **committed** run and the list of its files, plus a **pending** record while an analysis is running.
* A new run is committed in this order: **(1)** every new artifact is verified (exists, non-empty, image/video decodes, JSON parses, PDF header valid, all inside the generated-output folders) ->
  **(2)** the manifest is atomically switched to the new run -> **(3)** only then the previous run's files are deleted. Up to step 2 nothing old is touched, so the previous result stays downloadable
  (image->image, video->video, image->video, video->image all use the same path).
* Failed / cancelled / report-error runs delete **only their own** partial files (matched by run id); the previous result and manifest stay.
* A deletion that fails (file locked, permission) never fails the new prediction: the file is remembered in `stale` and retried on the next run / server start.
* Server start (`recover_outputs()`): a run still marked pending (crash, kill, redeploy) is discarded; stale deletions are retried.
* Safety rails: only regular files inside `images/ videos/ json/ csv/ reports/ charts/` whose name contains a run id can ever be deleted - never models, source files, `previews/`, symlinks, other files in `outputs/`, paths with `..`, or files of the current run.
  A missing/corrupt manifest is moved aside (`.run_manifest.json.corrupt`) and treated as "nothing known" (nothing is deleted, the next run starts a clean record).

## 8. Tests
```bash
cd backend
python -m unittest discover -s tests -v          # 16 unit tests of the artifact logic, no models needed
python tests/integration_replacement.py          # end-to-end through the API; needs the models (see models/README.md); takes a few minutes on CPU
```
The integration test covers all four media sequences, a corrupt upload, an injected report failure, cancel, the analysis lock, a hard-killed analysis followed by a restart, and (optionally,
`REFERENCE_VIDEO_JSON` / `REFERENCE_IMAGE_JSON`) a comparison of the response keys with an older response.
