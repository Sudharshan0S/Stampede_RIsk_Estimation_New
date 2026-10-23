"""End-to-end check of the safe-replacement behaviour through the real HTTP API.

REQUIRES the trained models (or any set of model files in the same formats) in CROWDRISK_MODELS_DIR (default: project/models).
Uses a temporary outputs folder and tiny synthetic media. Takes a few minutes on CPU.

    cd backend && python tests/integration_replacement.py
Optional:  REFERENCE_VIDEO_JSON / REFERENCE_IMAGE_JSON = an older API response saved as JSON, to confirm that the response keys did not change.
"""
import io, json, os, signal, subprocess, sys, tempfile, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = os.environ.get("CROWDRISK_IT_OUT") or tempfile.mkdtemp(prefix="crowdrisk_it_")      # child processes reuse the parent's folder
os.environ["CROWDRISK_IT_OUT"] = OUT
os.environ["CROWDRISK_OUTPUT_DIR"] = OUT
os.environ.setdefault("CROWDRISK_DEVICE", "cpu")
sys.path.insert(0, str(HERE))

import cv2, numpy as np

FAILS = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail and not cond else "")); 
    if not cond: FAILS.append(name)

def make_image(path, seed=1):
    rng = np.random.default_rng(seed); im = np.full((180, 320, 3), 70, np.uint8)
    for _ in range(40): cv2.circle(im, (int(rng.integers(10, 310)), int(rng.integers(10, 170))), 6, tuple(int(x) for x in rng.integers(60, 255, 3)), -1)
    cv2.imwrite(str(path), im)

def make_video(path, frames=70, seed=1):
    rng = np.random.default_rng(seed); n = 35; pos = rng.uniform([10, 10], [310, 170], (n, 2)); vel = rng.normal(0, 2.0, (n, 2)); vel[:, 0] += 1.5
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
    for _ in range(frames):
        im = np.full((180, 320, 3), 70, np.uint8); pos = (pos + vel) % [320, 180]
        for p in pos: cv2.circle(im, (int(p[0]), int(p[1])), 6, (200, 120, 80), -1)
        w.write(im)
    w.release()

GEN = ("images", "videos", "json", "csv", "reports", "charts")
def disk():
    return sorted(f"{d}/{f.name}" for d in GEN for f in (Path(OUT) / d).iterdir())
def run_ids():
    import re
    return {m.group(0) for f in disk() for m in [re.search(r"[0-9a-f]{32}", f)] if m}
def manifest():
    return json.loads((Path(OUT) / ".run_manifest.json").read_text())

if len(sys.argv) > 1 and sys.argv[1] == "child_video":        # a process that gets killed in the middle of a video analysis
    import app as A
    c = A.app.test_client(); A.app.test_client().post("/predict/video", data={"video": (open(sys.argv[2], "rb"), "v.mp4"), "area": "50"}, content_type="multipart/form-data"); sys.exit(0)
if len(sys.argv) > 1 and sys.argv[1] == "child_boot":         # a fresh process (= server restart)
    import app as A; print("BOOTED"); sys.exit(0)

import app as A
import predictor as P
import reporting
c = A.app.test_client(); tmp = Path(tempfile.mkdtemp())
img1, img2, vid1, vid2, junk = tmp / "a.jpg", tmp / "b.jpg", tmp / "a.mp4", tmp / "b.mp4", tmp / "junk.mp4"
make_image(img1, 1); make_image(img2, 2); make_video(vid1, 70, 1); make_video(vid2, 70, 2); junk.write_bytes(b"this is not a video")
def post_image(p): r = c.post("/predict/image", data={"image": (open(p, "rb"), Path(p).name), "area": "50"}, content_type="multipart/form-data"); return r.status_code, r.get_json()
def post_video(p): r = c.post("/predict/video", data={"video": (open(p, "rb"), Path(p).name), "area": "50"}, content_type="multipart/form-data"); return r.status_code, r.get_json()
def urls_ok(d):
    dl = d["downloads"]; flat = [v for k, v in dl.items() if k != "charts" and v] + list((dl.get("charts") or {}).values())
    return all(c.get(u).status_code == 200 for u in flat)

print("models ready:", A.models_status()["ready"], "| outputs:", OUT)
check("models are loaded (required for this test)", A.models_status()["ready"], str(A.models_status()["error"]))
prev_ids = set()
for label, fn, arg in (("image -> image (1st)", post_image, img1), ("image -> image (2nd)", post_image, img2), ("image -> video", post_video, vid1),
                       ("video -> image", post_image, img1), ("video -> video (1st)", post_video, vid1), ("video -> video (2nd)", post_video, vid2)):
    code, d = fn(arg); ok = code == 200 and d.get("success")
    check(f"{label}: success", ok, str(d.get("error"))[:200])
    if not ok: continue
    ids = run_ids(); m = manifest()
    check(f"{label}: exactly one run left on disk (the new one)", len(ids) == 1 and ids == {m["committed"]["run_id"]} and not (ids & prev_ids), f"{ids} vs {prev_ids}")
    check(f"{label}: every file on disk is tracked, nothing extra", set(disk()) == set(m["committed"]["files"]))
    check(f"{label}: all returned URLs are downloadable", urls_ok(d)); prev_ids = ids

# ---- failures keep the previous result ---------------------------------------------------------------------------------------------------------
before = disk(); mbefore = manifest()
code, d = post_video(junk); check("corrupt video upload -> reported as failure", code == 200 and d.get("success") is False)
check("corrupt video upload: previous result preserved, no partial files", disk() == before and manifest()["committed"] == mbefore["committed"] and manifest()["pending"] is None)
orig = reporting.generate_pdf_report
reporting.generate_pdf_report = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected report failure"))
code, d = post_image(img1); reporting.generate_pdf_report = orig
check("report-generation failure -> reported as failure", d.get("success") is False and "injected" in str(d.get("error")))
check("report-generation failure: previous result preserved, partial files removed", disk() == before and manifest()["committed"] == mbefore["committed"])

# ---- cancel ----------------------------------------------------------------------------------------------------------------------------------------
res = {}; th = threading.Thread(target=lambda: res.update(P.predict_video(str(vid1), 50))); th.start(); time.sleep(4)
try: P.predict_image(cv2.imread(str(img1)))
except P.Busy: check("analysis lock: second request while busy is rejected", True)
else: check("analysis lock: second request while busy is rejected", False)
P.request_cancel(); th.join()
check("cancelled run reported", res.get("cancelled") is True)
check("cancelled run: previous result preserved, no partial/raw files", disk() == before and manifest()["committed"] == mbefore["committed"], str(set(disk()) ^ set(before)))

# ---- hard kill + restart ------------------------------------------------------------------------------------------------------------------------
env = dict(os.environ); child = subprocess.Popen([sys.executable, __file__, "child_video", str(vid1)], env=env, cwd=str(HERE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
deadline = time.time() + 120
while time.time() < deadline and not any("_raw_" in f for f in disk()): time.sleep(0.5)
check("killed process had started writing a run", any("_raw_" in f for f in disk()) and manifest()["pending"] is not None)
child.send_signal(signal.SIGKILL); child.wait()
check("after SIGKILL the half-written run is still on disk (simulated crash)", len(run_ids()) == 2)
boot = subprocess.run([sys.executable, __file__, "child_boot"], env=env, cwd=str(HERE), capture_output=True, text=True, timeout=240)
check("server restart recovers: partial run removed", "BOOTED" in boot.stdout and disk() == before, boot.stdout[-300:] + boot.stderr[-300:])
check("server restart recovers: previous result and manifest intact", manifest()["committed"] == mbefore["committed"] and manifest()["pending"] is None)

# ---- response structure unchanged -------------------------------------------------------------------------------------------------------------
for name, env_key, post, arg in (("video", "REFERENCE_VIDEO_JSON", post_video, vid2), ("image", "REFERENCE_IMAGE_JSON", post_image, img2)):
    ref = os.environ.get(env_key)
    if ref and Path(ref).is_file():
        _, d = post(arg); old = json.load(open(ref)); check(f"{name} response keys identical to the reference response", set(d) == set(old), str(set(d) ^ set(old)))
        check(f"{name} downloads keys identical", set(d["downloads"]) == set(old["downloads"]), str(set(d["downloads"]) ^ set(old["downloads"])))
print("\nRESULT:", "ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
