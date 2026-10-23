# models/  - put the trained files here

Copy these files from the notebooks' output folder  `crowdrisk_outputs/models/`  into THIS folder
(flat - no sub-folders):

```
models/
├── crowd_count_model.pth     REQUIRED  notebook 01  CrowdPointNet weights + inference settings (refinement, tiling, thresholds ...)
├── flow_selection.json       REQUIRED  notebook 02  which optical-flow estimator was selected (your run: "dis", work_width 640)
├── abnormal_motion.json      REQUIRED  notebook 03  feature config, change (baseline) model, abnormality detector, reference scales
├── risk_model.pkl            REQUIRED  notebook 04  LightGBM risk + motion-state models, alarm rule (your run: RISK_AND_ABN)
│
├── motion_model.pth          only if flow_selection.json says "motionnet_v2"  (your run selected DIS, so it is NOT needed)
├── crowd_count_info.json     optional  notebook 01  count-model test metrics (stored in the API response if present)
├── feature_config.json       optional  notebook 03  copy of the config already embedded in abnormal_motion.json
├── label_rules.json          optional  notebook 03  training-time label thresholds (not used at inference)
└── motion_model_last.pth     optional  notebook 02  training resume file (not used at inference)
```

Files that are NOT needed by the backend (they stay in `crowdrisk_outputs/`): everything in `outputs/` (figures, metrics, predictions, demo videos)
and `work/` (checkpoints, caches).

The four REQUIRED files must come from the SAME run: notebook 03 stores the fingerprint of the crowd-count model it used, and the backend warns
(`GET /models/status`) if `crowd_count_model.pth` is a different file.

Use another location? Set `CROWDRISK_MODELS_DIR=/path/to/models` before starting the backend.
