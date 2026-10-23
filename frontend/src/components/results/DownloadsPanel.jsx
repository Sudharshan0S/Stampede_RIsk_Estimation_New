import { useState } from "react";
import {
  Download,
  FileImage,
  FileVideo,
  FileJson,
  FileSpreadsheet,
  FileText,
  FileBadge,
  BarChart3,
  CheckCircle2,
  Loader2,
  ShieldAlert,
} from "lucide-react";
import { Card, Badge, riskTone } from "../common/Primitives";
import { downloadOutput } from "../../lib/api";
import { fmtNumber, parseReasons } from "../../lib/format";

function DownloadRow({ icon: Icon, label, hint, onClick, disabled }) {
  const [state, setState] = useState("idle"); // idle | loading | done

  const handleClick = async () => {
    if (disabled || state === "loading") return;
    setState("loading");
    try {
      await onClick();
      setState("done");
      setTimeout(() => setState("idle"), 1800);
    } catch {
      setState("idle");
    }
  };

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className="w-full flex items-center gap-3 rounded-xl border border-white/[0.06] bg-base-900/50 px-3.5 py-3 text-left transition-colors hover:border-accent/40 hover:bg-accent/[0.06] disabled:opacity-35 disabled:cursor-not-allowed disabled:hover:bg-base-900/50 disabled:hover:border-white/[0.06]"
    >
      <div className="grid place-items-center h-9 w-9 shrink-0 rounded-lg bg-accent/10 text-accent-soft">
        <Icon size={16} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium text-white truncate">{label}</div>
        <div className="text-[11px] text-slate-500 truncate">{hint}</div>
      </div>
      {state === "loading" ? (
        <Loader2 size={16} className="text-accent-soft animate-spin shrink-0" />
      ) : state === "done" ? (
        <CheckCircle2 size={16} className="text-risk-low shrink-0" />
      ) : (
        <Download size={15} className="text-slate-500 shrink-0" />
      )}
    </button>
  );
}

export default function DownloadsPanel({ result }) {
  const isVideo = result.media_type === "video";
  const downloads = result.downloads || {};
  const reasons = parseReasons(result.risk_reason);

  return (
    <div className="space-y-5 lg:sticky lg:top-20">
      <Card className="p-4 sm:p-5 min-w-0">
        <div className="flex items-center justify-between gap-2 mb-1">
          <h3 className="text-sm font-semibold text-white">{isVideo ? "Peak Risk" : "Crowd Regime"}</h3>
          <Badge tone={isVideo ? riskTone(result.risk_level) : "accent"}>{isVideo ? result.risk_level : (result.crowd_regime || "—").toUpperCase()}</Badge>
        </div>
        <div className="text-4xl font-display font-bold text-white mt-2 tabular">
          {isVideo ? `${fmtNumber(result.peak_risk_score, 0)} / 100` : fmtNumber(result.crowd_count)}
        </div>
        <p className="text-xs text-slate-500 mt-1">
          {isVideo
            ? `${result.n_events} abnormal event(s) · risk labels are derived (weak), not official stampede labels.`
            : "Estimated people (accepted head points). Motion risk needs a video."}
        </p>

        {reasons.length > 0 && (
          <div className="mt-4 pt-4 border-t border-white/[0.06] space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wide">
              <ShieldAlert size={13} /> {isVideo ? "Why (peak window)" : "Note"}
            </div>
            <ul className="space-y-1.5">
              {reasons.map((r, i) => (
                <li key={i} className="text-sm text-slate-300 flex gap-2">
                  <span className="text-accent-soft">•</span>
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>

      <Card className="p-5">
        <h3 className="text-sm font-semibold text-white mb-3">Downloads</h3>
        <div className="space-y-2.5">
          <DownloadRow
            icon={isVideo ? FileVideo : FileImage}
            label={isVideo ? "Annotated Video" : "Annotated Image"}
            hint={isVideo ? "MP4: flow arrows, heads, risk HUD" : "JPG with head-point markers"}
            disabled={!(isVideo ? downloads.annotated_video : downloads.annotated_image)}
            onClick={() =>
              downloadOutput(
                isVideo ? downloads.annotated_video : downloads.annotated_image,
                isVideo ? "annotated_video.mp4" : "annotated_image.jpg"
              )
            }
          />
          <DownloadRow
            icon={FileJson}
            label="Full JSON Output"
            hint={isVideo ? "Summary, events and per-window data" : "All computed metrics + head points"}
            disabled={!downloads.json}
            onClick={() => downloadOutput(downloads.json, "analysis_result.json")}
          />
          <DownloadRow
            icon={FileSpreadsheet}
            label="CSV Export"
            hint={isVideo ? "Full feature table, one row per window" : "Single-row metric export"}
            disabled={!downloads.csv}
            onClick={() => downloadOutput(downloads.csv, "analysis_result.csv")}
          />
          {isVideo && (
            <DownloadRow
              icon={FileSpreadsheet}
              label="Events CSV"
              hint="One row per detected abnormal event"
              disabled={!downloads.events_csv}
              onClick={() => downloadOutput(downloads.events_csv, "detected_events.csv")}
            />
          )}
          {isVideo && (
            <DownloadRow
              icon={FileJson}
              label="Events JSON"
              hint="Events with explanations"
              disabled={!downloads.events_json}
              onClick={() => downloadOutput(downloads.events_json, "events.json")}
            />
          )}
          <DownloadRow
            icon={FileText}
            label="Summary Report (TXT)"
            hint="Plain-text analysis report"
            disabled={!downloads.report}
            onClick={() => downloadOutput(downloads.report, "risk_report.txt")}
          />
          <DownloadRow
            icon={FileBadge}
            label="Summary Report (PDF)"
            hint="Formatted analysis report"
            disabled={!downloads.report_pdf}
            onClick={() => downloadOutput(downloads.report_pdf, "risk_report.pdf")}
          />
        </div>
      </Card>

      {downloads.charts && Object.keys(downloads.charts).length > 0 && (
        <Card className="p-5">
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <BarChart3 size={15} className="text-accent-soft" /> Chart Images
          </h3>
          <div className="space-y-2.5">
            {Object.entries(downloads.charts).map(([name, url]) => (
              <DownloadRow
                key={name}
                icon={BarChart3}
                label={name
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, (c) => c.toUpperCase())}
                hint="PNG chart generated by the backend"
                disabled={!url}
                onClick={() => downloadOutput(url, `${name}.png`)}
              />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
