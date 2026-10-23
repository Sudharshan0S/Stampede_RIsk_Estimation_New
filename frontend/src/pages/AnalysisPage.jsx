import { useEffect, useRef, useState } from "react";
import {
  Loader2,
  CheckCircle2,
  Circle,
  XCircle,
  Cpu,
  Server,
  Clock,
  Layers,
  Terminal,
  Gauge,
  Timer,
  Flag,
  Ban,
} from "lucide-react";
import { useApp } from "../context/AppContext";
import { fetchProgress } from "../lib/api";
import { Card, SectionHeading, Badge, Button } from "../components/common/Primitives";
import ConfirmDialog from "../components/common/ConfirmDialog";
import { STAGE_LABELS } from "../components/layout/TopStatusBar";

const WORKFLOW_STEPS = [
  { key: "PREPARING", label: "Preparing Environment" },
  { key: "CONNECTING", label: "Connecting Backend" },
  { key: "UPLOADING", label: "Uploading Media" },
  { key: "EXTRACTING", label: "Extracting Frames" },
  { key: "CROWD_COUNTING", label: "Crowd Counting" },
  { key: "MOTION_ANALYSIS", label: "Motion Analysis" },
  { key: "RISK_ESTIMATION", label: "Risk Estimation" },
  { key: "GENERATING_VIDEO", label: "Generating Visualizations" },
  { key: "EXPORTING", label: "Generating Reports" },
  { key: "COMPLETED", label: "Completed" },
];

function stepState(stepKey, currentStage, status) {
  if (status === "cancelled") return "cancelled";
  if (status === "error") return stepKey === "COMPLETED" ? "failed" : stepState_inner(stepKey, currentStage);
  if (status === "done") return "done";
  return stepState_inner(stepKey, currentStage);
}

function stepState_inner(stepKey, currentStage) {
  const currentIdx = WORKFLOW_STEPS.findIndex((s) => s.key === currentStage);
  const thisIdx = WORKFLOW_STEPS.findIndex((s) => s.key === stepKey);
  if (currentIdx === -1) return "pending";
  if (thisIdx < currentIdx) return "done";
  if (thisIdx === currentIdx) return "running";
  return "pending";
}

function fmtClock(date) {
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fmtMMSS(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "--:--";
  const s = Math.max(0, Math.round(seconds));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

export default function AnalysisPage() {
  const { status, mediaKind, error, cancelAnalysis, analysisStartedAt, setCurrentStageLabel, reset } = useApp();
  const [snapshot, setSnapshot] = useState(null);
  const [log, setLog] = useState([]);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [, setTick] = useState(0);
  const sinceRef = useRef(0);
  const consoleRef = useRef(null);
  const lastSnapshotAtRef = useRef(Date.now());

  // Ticks the component every 250ms so Elapsed/Remaining read as a smooth,
  // continuously running clock instead of only jumping forward whenever a
  // new backend frame update happens to arrive.
  useEffect(() => {
    if (status !== "analyzing") return undefined;
    const id = setInterval(() => setTick((t) => t + 1), 250);
    return () => clearInterval(id);
  }, [status]);

  useEffect(() => {
    if (status !== "analyzing") return undefined;

    setLog([]);
    sinceRef.current = 0;

    let cancelled = false;
    const poll = async () => {
      try {
        const data = await fetchProgress(sinceRef.current || undefined);
        if (cancelled) return;
        setSnapshot(data.current);
        lastSnapshotAtRef.current = Date.now();
        setCurrentStageLabel(data.current?.message || null);
        if (data.log?.length) {
          setLog((prev) => [...prev, ...data.log]);
        }
        if (data.latest_seq) sinceRef.current = data.latest_seq;
      } catch {
        // backend momentarily unreachable — keep polling silently
      }
    };

    poll();
    const id = setInterval(poll, 500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [status, setCurrentStageLabel]);

  // Auto-scroll the console as new lines arrive.
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [log]);

  const isDone = status === "done" || status === "error" || status === "cancelled";
  const pct = isDone ? (status === "done" ? 100 : snapshot?.progress ?? 0) : snapshot?.progress ?? 0;
  const currentStage = snapshot?.current_stage || "PREPARING";
  const processedFrames = snapshot?.processed_frames ?? 0;
  const totalFrames = snapshot?.total_frames ?? 0;

  const startedClock = analysisStartedAt ? new Date(analysisStartedAt) : new Date();

  // Smooth, continuously-ticking elapsed time: while analyzing, derive it
  // from the client-side start timestamp + "now" (updated every 250ms
  // above) rather than the backend snapshot, which only changes when a
  // new frame finishes processing and could otherwise look stepped/frozen
  // between updates. Once finished, show the backend's authoritative value.
  const elapsedSeconds = isDone
    ? snapshot?.elapsed_seconds ?? 0
    : analysisStartedAt
    ? (Date.now() - analysisStartedAt) / 1000
    : 0;

  // Same idea for the ETA countdown: extrapolate smoothly from the last
  // known eta_seconds using how much local time has passed since it arrived.
  const rawEta = snapshot?.eta_seconds;
  const etaSeconds =
    isDone || rawEta == null
      ? rawEta
      : Math.max(0, rawEta - (Date.now() - lastSnapshotAtRef.current) / 1000);
  const estimatedFinish = etaSeconds != null ? new Date(Date.now() + etaSeconds * 1000) : null;

  const completedCount = WORKFLOW_STEPS.filter(
    (s) => stepState(s.key, currentStage, status) === "done"
  ).length;

  const handleConfirmCancel = async () => {
    setConfirmCancel(false);
    setCancelling(true);
    await cancelAnalysis();
    setCancelling(false);
  };

  return (
    <div className="space-y-6">
      <SectionHeading
        eyebrow="Step 02"
        title="Live AI Analysis"
        description="Monitor every stage of the AI powered crowd analysis pipeline while the backend processes the uploaded media."
        action={
          status === "error" ? (
            <Badge tone="danger">
              <XCircle size={12} /> Analysis Failed
            </Badge>
          ) : status === "cancelled" ? (
            <Badge tone="neutral">
              <Ban size={12} /> Cancelled by User
            </Badge>
          ) : isDone ? (
            <Badge tone="success">
              <CheckCircle2 size={12} /> Processing Complete
            </Badge>
          ) : (
            <Badge tone="accent">
              <Loader2 size={12} className="animate-spin" /> Processing
            </Badge>
          )
        }
      />

      {status === "error" && (
        <Card className="p-4 border-risk-critical/30 bg-risk-critical/5 text-sm text-risk-critical">
          {error}
        </Card>
      )}
      {status === "cancelled" && (
        <Card className="p-4 border-white/10 bg-white/[0.03] text-sm text-slate-300 flex items-center justify-between gap-3 flex-wrap">
          <span>Analysis cancelled by user. Temporary files were removed and the backend returned to idle.</span>
          <Button variant="ghost" onClick={reset}>
            Back to Upload
          </Button>
        </Card>
      )}

      {/* Top metrics row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card className="p-4 min-w-0">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Layers size={13} /> Current Stage
          </div>
          <div className="text-sm font-semibold text-white break-words">
            {STAGE_LABELS[currentStage] || currentStage}
          </div>
        </Card>
        <Card className="p-4 min-w-0">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Server size={13} /> Backend Status
          </div>
          <div className="text-sm font-semibold text-risk-low">ONLINE</div>
        </Card>
        <Card className="p-4 min-w-0">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Clock size={13} /> Elapsed Time
          </div>
          <div className="text-sm font-semibold text-white tabular">{fmtMMSS(elapsedSeconds)}</div>
        </Card>
        <Card className="p-4 min-w-0">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Cpu size={13} /> Progress
          </div>
          <div className="text-sm font-semibold text-accent-soft tabular">{pct}%</div>
        </Card>
      </div>

      {/* Timing & throughput row (video especially benefits from this) */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card className="p-4 min-w-0">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Flag size={13} /> Started
          </div>
          <div className="text-sm font-semibold text-white tabular">{fmtClock(startedClock)}</div>
        </Card>
        <Card className="p-4 min-w-0">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Timer size={13} /> Remaining
          </div>
          <div className="text-sm font-semibold text-white tabular">
            {isDone ? "00:00" : fmtMMSS(etaSeconds)}
          </div>
        </Card>
        <Card className="p-4 min-w-0">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Clock size={13} /> ETA
          </div>
          <div className="text-sm font-semibold text-white tabular">
            {isDone || !estimatedFinish ? "—" : fmtClock(estimatedFinish)}
          </div>
        </Card>
        <Card className="p-4 min-w-0">
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-1">
            <Gauge size={13} /> FPS (current / avg)
          </div>
          <div className="text-sm font-semibold text-white tabular whitespace-nowrap">
            {(snapshot?.fps_current ?? 0).toFixed(1)} / {(snapshot?.fps_average ?? 0).toFixed(1)}
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* AI processing ring + stats */}
        <Card className="lg:col-span-2 p-4 sm:p-6 min-w-0">
          <h3 className="text-sm font-semibold text-white mb-6">AI Processing Engine</h3>
          <div className="flex flex-col items-center">
            <ProgressRing percent={pct} tone={status === "error" ? "#ff4d5e" : status === "cancelled" ? "#8b93b0" : "#7c5cff"} />
            {mediaKind === "video" && (
              <div className="w-full grid grid-cols-1 xs:grid-cols-2 gap-3 mt-6">
                <div className="rounded-lg bg-base-900/60 border border-white/[0.06] px-3 py-2.5 text-center">
                  <div className="text-[11px] text-slate-500">Frames Processed</div>
                  <div className="text-sm font-semibold text-white tabular">
                    {processedFrames} / {totalFrames || "—"}
                  </div>
                </div>
                <div className="rounded-lg bg-base-900/60 border border-white/[0.06] px-3 py-2.5 text-center">
                  <div className="text-[11px] text-slate-500">Frames Remaining</div>
                  <div className="text-sm font-semibold text-white tabular">
                    {isDone ? 0 : snapshot?.frames_remaining ?? Math.max(0, totalFrames - processedFrames)}
                  </div>
                </div>
              </div>
            )}
          </div>
        </Card>

        {/* Workflow checklist */}
        <Card className="lg:col-span-3 p-4 sm:p-6 min-w-0">
          <h3 className="text-sm font-semibold text-white mb-4">Backend Processing Workflow</h3>
          <div className="space-y-2 max-h-[320px] sm:max-h-[360px] overflow-y-auto overscroll-contain pr-1">
            {WORKFLOW_STEPS.map((s) => {
              const state = stepState(s.key, currentStage, status);
              return (
                <div
                  key={s.key}
                  className={`flex items-center gap-3 rounded-xl border px-3 sm:px-4 py-2.5 sm:py-3 transition-colors ${
                    state === "done"
                      ? "border-risk-low/25 bg-risk-low/[0.06]"
                      : state === "running"
                      ? "border-accent/40 bg-accent/[0.08]"
                      : state === "failed"
                      ? "border-risk-critical/30 bg-risk-critical/[0.06]"
                      : state === "cancelled"
                      ? "border-white/10 bg-white/[0.03]"
                      : "border-white/[0.06] bg-base-900/40"
                  }`}
                >
                  {state === "done" ? (
                    <CheckCircle2 size={16} className="text-risk-low shrink-0" />
                  ) : state === "running" ? (
                    <Loader2 size={16} className="text-accent-soft shrink-0 animate-spin" />
                  ) : state === "failed" ? (
                    <XCircle size={16} className="text-risk-critical shrink-0" />
                  ) : state === "cancelled" ? (
                    <Ban size={16} className="text-slate-500 shrink-0" />
                  ) : (
                    <Circle size={16} className="text-slate-600 shrink-0" />
                  )}
                  <div className="min-w-0">
                    <div
                      className={`text-sm font-medium truncate ${
                        state === "pending" ? "text-slate-500" : "text-white"
                      }`}
                    >
                      {s.label}
                    </div>
                    <div className="text-[11px] text-slate-500 capitalize">{state}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      {/* Live console */}
      <Card className="p-4 sm:p-6 min-w-0">
        <div className="flex items-center justify-between gap-2 mb-3">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Terminal size={15} className="text-accent-soft" /> Live AI Console
          </h3>
          <Badge tone={isDone ? "success" : "accent"}>{isDone ? "Idle" : "Processing"}</Badge>
        </div>
        <div
          ref={consoleRef}
          className="rounded-xl bg-black/50 border border-white/[0.06] p-3 sm:p-4 h-48 sm:h-64 overflow-y-auto overscroll-contain font-mono text-[11px] sm:text-xs space-y-1"
        >
          {log.length === 0 && <p className="text-slate-600">Waiting for backend updates…</p>}
          {log.map((entry) => (
            <div key={entry.seq} className="flex gap-2 sm:gap-3 text-slate-400">
              <span className="text-slate-600 shrink-0">{entry.time}</span>
              <span className="text-slate-300 min-w-0 break-words">{entry.message}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Cancel Analysis */}
      {!isDone && (
        <div className="flex justify-center">
          <Button
            variant="danger"
            icon={Ban}
            onClick={() => setConfirmCancel(true)}
            disabled={cancelling}
          >
            {cancelling ? "Cancelling…" : "Cancel Analysis"}
          </Button>
        </div>
      )}

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel this analysis?"
        description="The backend will stop processing, delete temporary files, and return to idle. This cannot be undone."
        confirmLabel="Cancel Analysis"
        cancelLabel="Keep Running"
        onConfirm={handleConfirmCancel}
        onClose={() => setConfirmCancel(false)}
      />
    </div>
  );
}

function ProgressRing({ percent, tone = "#7c5cff" }) {
  const radius = 72;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (Math.min(100, Math.max(0, percent)) / 100) * circumference;

  return (
    <div className="relative h-40 w-40 xs:h-48 xs:w-48">
      <svg viewBox="0 0 180 180" className="h-full w-full -rotate-90">
        <circle cx="90" cy="90" r={radius} fill="none" stroke="#1a2036" strokeWidth="12" />
        <circle
          cx="90"
          cy="90"
          r={radius}
          fill="none"
          stroke={tone}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.4s ease" }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center flex-col text-center">
        <div>
          <div className="text-3xl font-display font-bold text-white tabular">{percent}%</div>
          <div className="text-xs text-slate-500 mt-1">AI Progress</div>
        </div>
      </div>
    </div>
  );
}
