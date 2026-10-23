"""Unit tests for the safe-replacement / artifact-tracking logic in predictor.py.

No trained models are needed: runs are simulated with small but VALID artifacts (real JPEG / MP4 / PDF / JSON files) written into a temporary outputs folder.
    cd backend && python -m unittest discover -s tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="crowdrisk_test_out_")
os.environ["CROWDRISK_OUTPUT_DIR"] = _TMP                       # must be set BEFORE importing config / predictor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import config
import predictor as P
from utils import to_output_url

OUT = config.OUTPUT_DIR


def _write_jpg(path):
    cv2.imwrite(str(path), np.full((48, 64, 3), 90, np.uint8))


def _write_mp4(path):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(12): w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()


def make_run(media_type, run_id=None, skip=(), corrupt=()):
    """Create the artifacts a finished run would have; returns (run_id, result, downloads)."""
    rid = run_id or uuid.uuid4().hex
    p = {}
    if media_type == "image":
        p["annotated_image"] = config.IMAGES_DIR / f"annotated_{rid}.jpg"; _write_jpg(p["annotated_image"])
        p["density_image"] = config.IMAGES_DIR / f"density_{rid}.jpg"; _write_jpg(p["density_image"])
        p["panel_image"] = config.IMAGES_DIR / f"panel_{rid}.jpg"; _write_jpg(p["panel_image"])
    else:
        p["annotated_video"] = config.VIDEOS_DIR / f"annotated_{rid}.mp4"; _write_mp4(p["annotated_video"])
        p["events_csv"] = config.CSV_DIR / f"video_{rid}_events.csv"; p["events_csv"].write_text("a,b\n1,2\n")
    p["json"] = config.JSON_DIR / f"{media_type}_{rid}.json"; p["json"].write_text(json.dumps({"ok": True}))
    p["csv"] = config.CSV_DIR / f"{media_type}_{rid}.csv"; p["csv"].write_text("a,b\n1,2\n")
    p["report"] = config.REPORTS_DIR / f"{media_type}_{rid}_report.txt"; p["report"].write_text("report")
    p["report_pdf"] = config.REPORTS_DIR / f"{media_type}_{rid}_report.pdf"; p["report_pdf"].write_bytes(b"%PDF-1.4\n%fake\n")
    chart = config.CHARTS_DIR / f"risk_chart_{rid}.png"; _write_jpg(chart)
    for k in corrupt:
        if k == "json": p["json"].write_text("{not json")
        if k == "report_pdf": p["report_pdf"].write_bytes(b"NOTAPDF")
    for k in skip: p[k].unlink()                                 # file listed in the response but missing on disk
    downloads = {k: to_output_url(v) for k, v in p.items()}; downloads["charts"] = {"risk_chart": to_output_url(chart)}
    result = {"output_image": str(p["annotated_image"])} if media_type == "image" else {"processed_video": str(p["annotated_video"])}
    return rid, result, downloads


def files_of(rid):
    return sorted(f.name for d in (config.IMAGES_DIR, config.VIDEOS_DIR, config.JSON_DIR, config.CSV_DIR, config.REPORTS_DIR, config.CHARTS_DIR) for f in d.iterdir() if rid in f.name)


def commit(media_type, **kw):
    rid, result, dl = make_run(media_type, **kw)
    P._begin_run(rid); P._commit_run(rid, media_type, result, dl)
    return rid


class ArtifactReplacementTests(unittest.TestCase):
    def setUp(self):
        for d in (config.IMAGES_DIR, config.VIDEOS_DIR, config.JSON_DIR, config.CSV_DIR, config.REPORTS_DIR, config.CHARTS_DIR, config.PREVIEW_DIR):
            for f in d.iterdir(): f.unlink()
        for f in OUT.glob(".run_manifest*"): f.unlink()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_TMP, ignore_errors=True)

    # ---- replacement across every media combination ----------------------------------------------------------------------------
    def test_cross_media_replacement_all_four_combinations(self):
        for first, second in (("image", "image"), ("video", "video"), ("image", "video"), ("video", "image")):
            with self.subTest(f"{first}->{second}"):
                self.setUp()
                a = commit(first); self.assertTrue(files_of(a))
                b = commit(second)
                self.assertEqual(files_of(a), [], "previous run must be fully removed")
                self.assertTrue(files_of(b), "new run must be intact")
                m = P._read_manifest()
                self.assertEqual(m["committed"]["run_id"], b); self.assertEqual(m["committed"]["media_type"], second); self.assertIsNone(m["pending"])
                on_disk = {f.relative_to(OUT).as_posix() for d in P._GENERATED_DIRS for f in d.iterdir()}
                self.assertEqual(on_disk, set(m["committed"]["files"]), "everything on disk is tracked and nothing else is left")

    def test_charts_and_all_artifact_kinds_are_tracked_and_removed(self):
        a = commit("video"); names = " ".join(files_of(a))
        for part in ("annotated_", ".json", ".csv", "_events.csv", "_report.txt", "_report.pdf", "risk_chart_"): self.assertIn(part, names)
        commit("image"); self.assertEqual(files_of(a), [])

    # ---- previous result survives every kind of failure -----------------------------------------------------------------------------
    def test_incomplete_new_run_keeps_previous_result(self):
        old = commit("video")
        for label, kw in (("missing report file", dict(skip=("report",))), ("corrupt json", dict(corrupt=("json",))), ("corrupt pdf", dict(corrupt=("report_pdf",))),
                          ("missing csv", dict(skip=("csv",)))):
            with self.subTest(label):
                rid, result, dl = make_run("image", **kw); P._begin_run(rid)
                with self.assertRaises(RuntimeError): P._commit_run(rid, "image", result, dl)
                P._discard_run(rid)                                                    # what predict_image's except-branch does
                self.assertEqual(files_of(rid), [], "partial files of the failed run are removed")
                self.assertTrue(files_of(old), "previous result must be preserved")
                self.assertEqual(P._read_manifest()["committed"]["run_id"], old)

    def test_unreadable_media_is_rejected(self):
        old = commit("image")
        rid, result, dl = make_run("video"); Path(result["processed_video"]).write_bytes(b"garbage")
        with self.assertRaises(RuntimeError): P._commit_run(rid, "video", result, dl)
        self.assertTrue(files_of(old))

    def test_cancelled_run_preserves_previous_result_and_removes_raw_file(self):
        old = commit("image"); rid = uuid.uuid4().hex; P._begin_run(rid)
        raw = config.VIDEOS_DIR / f"_raw_{rid}.mp4"; _write_mp4(raw)                  # what predict_video has written when the user cancels
        self.assertGreaterEqual(P._discard_run(rid), 1)
        self.assertFalse(raw.exists()); self.assertTrue(files_of(old)); self.assertIsNone(P._read_manifest()["pending"])

    # ---- restart recovery ------------------------------------------------------------------------------------------------------------------
    def test_restart_recovery_removes_interrupted_run_only(self):
        old = commit("video")
        crashed = uuid.uuid4().hex; P._begin_run(crashed); _write_mp4(config.VIDEOS_DIR / f"_raw_{crashed}.mp4"); (config.CSV_DIR / f"video_{crashed}.csv").write_text("x")
        # "restart": nothing is kept in memory, recovery works purely from the manifest on disk
        P.recover_outputs()
        self.assertEqual(files_of(crashed), []); self.assertTrue(files_of(old)); self.assertIsNone(P._read_manifest()["pending"])

    def test_cleanup_after_restart_uses_persisted_manifest(self):
        a = commit("image")
        # simulate a restart: the module-level state is gone, only outputs/.run_manifest.json remains
        self.assertTrue((OUT / ".run_manifest.json").exists())
        b = commit("video")
        self.assertEqual(files_of(a), []); self.assertTrue(files_of(b))

    def test_new_run_discards_stale_pending_record(self):
        crashed = uuid.uuid4().hex; P._begin_run(crashed); (config.JSON_DIR / f"image_{crashed}.json").write_text("{}")
        b = commit("image"); self.assertEqual(files_of(crashed), []); self.assertTrue(files_of(b))

    # ---- corrupt / missing metadata -------------------------------------------------------------------------------------------------------
    def test_corrupt_manifest_is_safe(self):
        a = commit("image"); (OUT / ".run_manifest.json").write_text("{ this is not json")
        b = commit("video")                                                                   # must not crash
        self.assertTrue(files_of(b)); self.assertTrue(files_of(a), "with no trustworthy record, nothing old is deleted")
        self.assertEqual(P._read_manifest()["committed"]["run_id"], b)
        self.assertTrue((OUT / ".run_manifest.json.corrupt").exists())

    def test_wrong_structure_manifest_is_treated_as_corrupt(self):
        a = commit("image"); (OUT / ".run_manifest.json").write_text(json.dumps({"committed": {"run_id": 5, "files": "nope"}}))
        self.assertIsNone(P._read_manifest()["committed"]); self.assertTrue(files_of(a))

    def test_missing_manifest_first_run(self):
        self.assertFalse((OUT / ".run_manifest.json").exists()); a = commit("image"); self.assertTrue(files_of(a))

    # ---- deletion errors never invalidate a successful run ---------------------------------------------------------------------------------
    def test_deletion_error_does_not_fail_run_and_is_retried(self):
        a = commit("image"); victim = next(config.IMAGES_DIR.glob(f"panel_{a}.jpg")); real_unlink = Path.unlink
        def flaky(self_, *args, **kw):
            if self_.resolve() == victim.resolve(): raise PermissionError("locked")
            return real_unlink(self_, *args, **kw)
        Path.unlink = flaky
        try: b = commit("video")                                                              # must not raise
        finally: Path.unlink = real_unlink
        m = P._read_manifest(); self.assertEqual(m["committed"]["run_id"], b); self.assertTrue(files_of(b))
        self.assertTrue(victim.exists()); self.assertIn(victim.relative_to(OUT).as_posix(), m["stale"], "failed deletion is remembered")
        P.recover_outputs()                                                                   # next start retries
        self.assertFalse(victim.exists()); self.assertEqual(P._read_manifest()["stale"], [])

    # ---- scope: nothing outside the generated-output folders is ever deleted ---------------------------------------------------------------
    def test_unrelated_and_out_of_scope_files_are_never_deleted(self):
        models = Path(_TMP).parent / f"models_{uuid.uuid4().hex[:6]}"; models.mkdir(); model_file = models / "risk_model.pkl"; model_file.write_text("m")
        note = config.REPORTS_DIR / "notes.txt"; note.write_text("user file"); other = OUT / "keep_me.txt"; other.write_text("x")
        preview = config.PREVIEW_DIR / "live_preview.mp4"; preview.write_bytes(b"p"); lookalike = config.JSON_DIR / "keepme_notahexid.json"; lookalike.write_text("{}")
        a = commit("image")
        m = P._read_manifest()                                                                # tamper: a hostile / buggy manifest pointing at protected paths
        m["committed"]["files"] += ["../" + model_file.relative_to(models.parent).as_posix(), str(model_file), "reports/notes.txt", "keep_me.txt", "previews/live_preview.mp4",
                                    "json/keepme_notahexid.json", "../../etc/passwd", "backend/app.py"]
        P._write_manifest(m); b = commit("video")
        for f in (model_file, note, other, preview, lookalike): self.assertTrue(f.exists(), f"{f} must survive")
        self.assertEqual(files_of(a), []); self.assertTrue(files_of(b)); shutil.rmtree(models)

    def test_current_run_files_are_protected_even_if_listed_as_old(self):
        a = commit("image"); rid, result, dl = make_run("video"); m = P._read_manifest()
        m["committed"]["files"] += [to_output_url(Path(result["processed_video"])).replace("/outputs/", "")]       # old record wrongly lists a NEW file
        P._write_manifest(m); P._begin_run(rid); P._commit_run(rid, "video", result, dl)
        self.assertTrue(files_of(rid)); self.assertTrue(Path(result["processed_video"]).exists())

    def test_symlink_is_not_followed(self):
        if os.name == "nt": self.skipTest("symlinks need privileges on Windows")
        a = commit("image"); target = Path(_TMP).parent / "outside_target.txt"; target.write_text("t")
        link = config.JSON_DIR / f"link_{uuid.uuid4().hex}.json"; link.symlink_to(target)
        m = P._read_manifest(); m["committed"]["files"].append(link.relative_to(OUT).as_posix()); P._write_manifest(m); commit("video")
        self.assertTrue(target.exists()); target.unlink(); link.unlink()

    def test_manifest_is_not_served_and_valid_json(self):
        commit("image"); json.loads((OUT / ".run_manifest.json").read_text())
        import app as A
        c = A.app.test_client(); self.assertEqual(c.get("/outputs/.run_manifest.json").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
