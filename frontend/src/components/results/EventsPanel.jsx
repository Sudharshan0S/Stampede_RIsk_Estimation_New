import { Siren, Clock, Users, Gauge, Wind, PlayCircle, ShieldCheck } from "lucide-react";
import { Card, Badge, riskTone } from "../common/Primitives";
import { fmtNumber, fmtSec, prettyTag } from "../../lib/format";

function Metric({ icon: Icon, label, value }) {
  return (
    <div className="rounded-lg border border-white/[0.06] bg-base-900/50 px-3 py-2 min-w-0">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-slate-500">
        <Icon size={12} /> {label}
      </div>
      <div className="text-sm font-medium text-white tabular truncate">{value}</div>
    </div>
  );
}

/** Abnormal events found by the notebook-04 pipeline (consecutive alarm windows merged into one event). */
export default function EventsPanel({ result, onSeek }) {
  const events = result.events || [];

  if (events.length === 0) {
    return (
      <Card className="p-8 flex flex-col items-center text-center gap-3">
        <ShieldCheck size={28} className="text-risk-low" />
        <p className="text-sm text-slate-300 max-w-md">
          No abnormal event was detected. An alarm needs persistence (on after 2 of 3 windows) and, with the selected
          alarm rule ({result.summary?.fusion_mode}), agreement between the risk model and the abnormality detector.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {events.map((e, i) => (
        <Card key={i} className="p-4 sm:p-5 space-y-4 min-w-0">
          <div className="flex items-start justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-3">
              <div className="grid place-items-center h-10 w-10 rounded-xl bg-risk-critical/10 text-risk-critical">
                <Siren size={18} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">Event {i + 1}</h3>
                <p className="text-xs text-slate-500">
                  {fmtSec(e.start_s)} – {fmtSec(e.end_s)} · peak at {fmtSec(e.peak_time_s)} · {e.n_windows} windows
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <Badge tone={riskTone(e.peak_risk)}>
                {e.peak_risk} · {fmtNumber(e.peak_risk_score, 0)}/100
              </Badge>
              <Badge tone="accent">confidence {fmtNumber(e.confidence, 2)}</Badge>
              {onSeek && (
                <button
                  onClick={() => onSeek(e.peak_time_s)}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 coarse:min-h-[44px] text-xs text-slate-300 hover:text-white hover:border-accent/40 transition-colors"
                >
                  <PlayCircle size={13} /> Jump to peak
                </button>
              )}
            </div>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {Object.entries(e.tags || {}).map(([t, n]) => (
              <Badge key={t} tone={t === e.dominant_behaviour ? "danger" : "neutral"}>
                {prettyTag(t)} <span className="text-slate-500">×{n}</span>
              </Badge>
            ))}
            {e.rush_pattern && <Badge tone="danger">Abnormal rush pattern</Badge>}
          </div>

          <div className="grid grid-cols-1 xs:grid-cols-2 lg:grid-cols-4 gap-2">
            <Metric icon={Users} label="Crowd count" value={`${fmtNumber(e.count_before)} → ${fmtNumber(e.count_peak)}`} />
            <Metric icon={Gauge} label="Density" value={`${fmtNumber(e.density_before, 2)} → ${fmtNumber(e.density_peak, 2)}`} />
            <Metric icon={Wind} label="Speed (px/step)" value={`${fmtNumber(e.speed_before, 2)} → ${fmtNumber(e.speed_peak, 2)}`} />
            <Metric icon={Clock} label="Independent signal groups" value={fmtNumber(e.max_independent_signals)} />
          </div>

          {(e.evidence || []).length > 0 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-1.5">Evidence</div>
              <ul className="space-y-1">
                {e.evidence.map((x, k) => (
                  <li key={k} className="text-sm text-slate-300 flex gap-2 break-words">
                    <span className="text-accent-soft">•</span>
                    <span>{x}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="text-[11px] text-slate-600">
            Confidence is evidence strength, not a calibrated probability of a real stampede.
          </p>
        </Card>
      ))}
    </div>
  );
}
