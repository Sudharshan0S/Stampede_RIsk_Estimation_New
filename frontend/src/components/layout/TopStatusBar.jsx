import { useEffect, useState } from "react";
import { Calendar, Clock, Database, Activity } from "lucide-react";
import { useApp } from "../../context/AppContext";

const STAGE_LABELS = {
  IDLE: "Ready",
  PREPARING: "Preparing Workspace",
  CONNECTING: "Connecting Backend",
  UPLOADING: "Uploading Media",
  EXTRACTING: "Extracting Frames",
  CROWD_COUNTING: "Crowd Counting Running",
  MOTION_ANALYSIS: "Motion Analysis Running",
  RISK_ESTIMATION: "Estimating Risk",
  GENERATING_VIDEO: "Generating Annotated Video",
  GENERATING_IMAGE: "Generating Annotated Image",
  EXPORTING: "Generating Reports",
  COMPLETED: "Completed Successfully",
  FAILED: "Backend Error",
  CANCELLED: "Cancelled by User",
};

function InfoCard({ icon: Icon, label, value, tone = "text-white" }) {
  return (
    <div className="flex items-center gap-2.5 sm:gap-3 rounded-xl border border-white/[0.06] bg-base-800/60 px-3 sm:px-4 py-2.5 sm:py-3.5 min-w-0 sm:flex-1">
      <div className="hidden xs:grid place-items-center h-8 w-8 sm:h-9 sm:w-9 shrink-0 rounded-lg bg-accent/10 text-accent-soft">
        <Icon size={16} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-[10px] sm:text-[11px] uppercase tracking-wide text-slate-500 whitespace-nowrap truncate">{label}</div>
        <div className={`text-sm font-semibold ${tone} leading-snug break-words`}>{value}</div>
      </div>
    </div>
  );
}

export default function TopStatusBar() {
  const { backendOnline, modelsError, status, currentStageLabel } = useApp();
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const dateStr = now.toLocaleDateString(undefined, {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });
  const dateStrShort = now.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" });
  const timeStr = now.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  const stageLabel =
    currentStageLabel ||
    (status === "analyzing"
      ? "Processing"
      : status === "done"
      ? "Completed Successfully"
      : status === "error"
      ? "Backend Error"
      : status === "cancelled"
      ? "Cancelled by User"
      : "Ready");

  return (
    <div className="flex flex-col gap-2 sm:gap-3 w-full">
      <div className="grid grid-cols-2 sm:flex sm:flex-row gap-2 sm:gap-3">
        <InfoCard icon={Calendar} label="Analysis Date" value={<><span className="sm:hidden">{dateStrShort}</span><span className="hidden sm:inline">{dateStr}</span></>} />
        <InfoCard icon={Clock} label="Current Time" value={timeStr} />
      </div>
      <div className="grid grid-cols-2 sm:flex sm:flex-row gap-2 sm:gap-3">
        <InfoCard
          icon={Database}
          label="Backend"
          value={backendOnline === null ? "Checking…" : backendOnline ? (modelsError ? "Connected · models not loaded" : "Connected") : "Offline"}
          tone={backendOnline ? (modelsError ? "text-risk-medium" : "text-risk-low") : backendOnline === false ? "text-risk-critical" : "text-slate-300"}
        />
        <InfoCard icon={Activity} label="Status" value={stageLabel} />
      </div>
    </div>
  );
}

export { STAGE_LABELS };
