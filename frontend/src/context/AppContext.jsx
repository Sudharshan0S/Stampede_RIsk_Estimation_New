import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import {
  analyzeImage,
  analyzeVideo,
  checkHealth,
  cancelAnalysis as apiCancelAnalysis,
  previewVideo,
  resolveMediaUrl,
} from "../lib/api";

const AppContext = createContext(null);

const IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp"];
const VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv"];

export function detectMediaKind(file) {
  if (!file) return null;
  const ext = file.name.split(".").pop()?.toLowerCase();
  if (IMAGE_EXTENSIONS.includes(ext)) return "image";
  if (VIDEO_EXTENSIONS.includes(ext)) return "video";
  return null;
}

const STEP_ORDER = ["input", "analysis", "results"];

export function AppProvider({ children }) {
  const [step, setStep] = useState("input");
  const [tab, setTab] = useState("input"); // header nav tab: input | analysis | results | about

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewNote, setPreviewNote] = useState(null);
  const [mediaKind, setMediaKind] = useState(null);
  const [area, setArea] = useState(100);

  const [status, setStatus] = useState("idle"); // idle | analyzing | done | error | cancelled
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [analysisStartedAt, setAnalysisStartedAt] = useState(null);
  const [currentStageLabel, setCurrentStageLabel] = useState(null);

  const [backendOnline, setBackendOnline] = useState(null); // null = unknown
  const [modelsError, setModelsError] = useState(null); // set when the backend is up but the trained models are not loaded

  const abortRef = useRef(null);
  const selectionTokenRef = useRef(0);
  const blobUrlRef = useRef(null);

  const pingBackend = useCallback(async () => {
    try {
      const health = await checkHealth();
      setBackendOnline(true);
      setModelsError(health?.models_ready === false ? health.models_error || "Trained models are not loaded." : null);
      return true;
    } catch {
      setBackendOnline(false);
      return false;
    }
  }, []);

  const selectFile = useCallback((nextFile) => {
    const kind = detectMediaKind(nextFile);
    if (!kind) {
      setError("Unsupported file type. Use JPG, JPEG, PNG, BMP, MP4, AVI, MOV or MKV.");
      return;
    }

    const token = ++selectionTokenRef.current;

    setError(null);
    setFile(nextFile);
    setMediaKind(kind);
    setResult(null);
    setStatus("idle");
    setPreviewNote(null);

    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }

    const blobUrl = URL.createObjectURL(nextFile);
    blobUrlRef.current = blobUrl;
    setPreviewUrl(blobUrl);

    const extension = nextFile.name.split(".").pop()?.toLowerCase();

    // mp4 already plays natively in the browser — no backend round trip
    // needed, zero extra time. Only non-mp4 videos need the cheap remux.
    if (kind !== "video" || extension === "mp4") {
      setPreviewLoading(false);
      return;
    }

    setPreviewLoading(true);
    previewVideo(nextFile)
      .then((data) => {
        if (selectionTokenRef.current !== token) return; // a newer file was selected meanwhile
        if (data.success && data.url) {
          setPreviewUrl(resolveMediaUrl(data.url));
        } else if (data.success && data.no_conversion_needed) {
          // shouldn't happen (mp4 already short-circuited above), but safe no-op
        } else {
          setPreviewNote(
            data.error || "Preview isn't available for this format, but analysis will still run normally."
          );
        }
      })
      .catch(() => {
        if (selectionTokenRef.current !== token) return;
        setPreviewNote("Preview isn't available for this format, but analysis will still run normally.");
      })
      .finally(() => {
        if (selectionTokenRef.current === token) setPreviewLoading(false);
      });
  }, []);

  const clearFile = useCallback(() => {
    selectionTokenRef.current += 1; // invalidate any in-flight preview request
    setFile(null);
    setMediaKind(null);
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    setPreviewUrl(null);
    setPreviewLoading(false);
    setPreviewNote(null);
    setResult(null);
    setStatus("idle");
    setError(null);
    setAnalysisStartedAt(null);
    setCurrentStageLabel(null);
  }, []);

  const reset = useCallback(() => {
    clearFile();
    setArea(100);
    setStep("input");
    setTab("input");
  }, [clearFile]);

  const runAnalysis = useCallback(async () => {
    if (!file || !mediaKind) return;

    setStatus("analyzing");
    setError(null);
    setStep("analysis");
    setTab("analysis");
    setAnalysisStartedAt(Date.now());

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const data =
        mediaKind === "image"
          ? await analyzeImage(file, area, { signal: controller.signal })
          : await analyzeVideo(file, area, { signal: controller.signal });

      if (data.cancelled) {
        setStatus("cancelled");
        setError(null);
        setStep("input");
        setTab("input");
        return;
      }

      if (data.success === false) {
        throw new Error(data.error || "Analysis failed.");
      }

      setResult(data);
      setStatus("done");
      setStep("results");
      setTab("results");
    } catch (err) {
      if (err?.name === "CanceledError" || err?.name === "AbortError") return;
      setStatus("error");
      setError(err?.response?.data?.error || err.message || "Analysis failed.");
      setStep("results");
      setTab("results");
    }
  }, [file, mediaKind, area]);

  const cancelAnalysis = useCallback(async () => {
    try {
      await apiCancelAnalysis();
    } catch {
      // Backend may already be done/unreachable — abort the request locally regardless.
    }
    abortRef.current?.abort();
  }, []);

  const value = useMemo(
    () => ({
      step,
      setStep,
      tab,
      setTab,
      file,
      previewUrl,
      previewLoading,
      previewNote,
      mediaKind,
      area,
      setArea,
      status,
      error,
      result,
      analysisStartedAt,
      currentStageLabel,
      setCurrentStageLabel,
      backendOnline,
      modelsError,
      pingBackend,
      selectFile,
      clearFile,
      reset,
      runAnalysis,
      cancelAnalysis,
      STEP_ORDER,
    }),
    [
      step,
      tab,
      file,
      previewUrl,
      previewLoading,
      previewNote,
      mediaKind,
      area,
      status,
      error,
      result,
      analysisStartedAt,
      currentStageLabel,
      backendOnline,
      modelsError,
      pingBackend,
      selectFile,
      clearFile,
      reset,
      runAnalysis,
      cancelAnalysis,
    ]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside AppProvider");
  return ctx;
}
