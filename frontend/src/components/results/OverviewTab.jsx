import {
  Users,
  Gauge,
  LayoutGrid,
  Wind,
  Timer,
  TrendingUp,
  AlertOctagon,
  Activity,
  Zap,
  Shuffle,
  Shapes,
  Server,
  CheckCircle2,
  Clock,
  Siren,
  Layers,
  HelpCircle,
  Sparkles,
} from "lucide-react";
import { StatTile } from "../common/Primitives";
import { fmtNumber, fmtPercent, fmtSeconds, fmtSec } from "../../lib/format";

function Section({ title, children }) {
  return (
    <div>
      <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2.5">{title}</h4>
      <div className="grid grid-cols-1 xs:grid-cols-2 lg:grid-cols-4 gap-3">{children}</div>
    </div>
  );
}

const cap = (s) => (s ? String(s).charAt(0).toUpperCase() + String(s).slice(1).toLowerCase() : "—");

export default function OverviewTab({ result }) {
  const isVideo = result.media_type === "video";

  if (!isVideo) {
    return (
      <div className="space-y-6">
        <Section title="Crowd Count">
          <StatTile icon={Users} label="Crowd Count" value={fmtNumber(result.crowd_count)} sub="Accepted head points" />
          <StatTile
            icon={HelpCircle}
            label="Uncertain Heads"
            value={fmtNumber(result.n_uncertain)}
            sub={`Interval ${result.count_interval?.[0]} – ${result.count_interval?.[1]}`}
          />
          <StatTile icon={Layers} label="Density-map Sum" value={fmtNumber(result.density_map_sum, 1)} sub="Support signal only" />
          <StatTile icon={Sparkles} label="Count Uncertainty" value={result.uncertainty?.level || "—"} sub="Heuristic, not calibrated" />
        </Section>
        <Section title="Crowd Density">
          <StatTile icon={Shapes} label="Crowd Regime" value={cap(result.crowd_regime)} sub="sparse / medium / dense / extreme" />
          <StatTile icon={Gauge} label="Density" value={fmtNumber(result.density, 2)} sub={`people / 10k px (${result.models?.work_width ?? 640} px work width)`} />
          <StatTile icon={Users} label="People per m²" value={fmtNumber(result.people_per_m2, 2)} sub={`count ÷ ${result.area} m² (your input)`} />
          <StatTile
            icon={LayoutGrid}
            label="Occupied Area"
            value={result.occupied_area_ratio == null ? "—" : fmtPercent(result.occupied_area_ratio * 100, 0)}
            sub="density-map cells with people"
          />
        </Section>
        <Section title="Processing">
          <StatTile icon={AlertOctagon} label="Motion Risk" value="N/A" sub="needs video (motion)" />
          <StatTile icon={Clock} label="Processing Time" value={fmtSeconds(result.processing_time)} sub={`${fmtNumber(result.inference_ms, 0)} ms inference`} />
          <StatTile icon={Server} label="Backend Status" value="ONLINE" accentClass="text-risk-low" />
          <StatTile icon={CheckCircle2} label="Analysis Status" value="Completed" accentClass="text-risk-low" />
        </Section>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Section title="Abnormal Behaviour">
        <StatTile
          icon={Siren}
          label="Abnormal Events"
          value={fmtNumber(result.n_events)}
          sub={result.n_events ? "see the Events tab" : "none detected"}
          accentClass={result.n_events ? "text-risk-critical" : "text-risk-low"}
        />
        <StatTile icon={Timer} label="Time Abnormal" value={fmtSec(result.time_abnormal_s)} sub="alarm active (persistence rule)" />
        <StatTile icon={Activity} label="Peak Abnormality" value={fmtNumber(result.maximum_abnormality, 2)} sub="noisy-OR evidence 0–1" />
        <StatTile icon={Shapes} label="Dominant Motion State" value={(result.dominant_motion_state || "—").replace(/_/g, " ")} />
      </Section>

      <Section title="Risk">
        <StatTile icon={AlertOctagon} label="Peak Risk" value={`${fmtNumber(result.peak_risk_score, 0)} / 100`} sub={`${result.risk_level} at ${fmtSec(result.peak_risk_time_s)}`} />
        <StatTile icon={AlertOctagon} label="Average Risk" value={`${fmtNumber(result.average_risk_score, 1)} / 100`} />
        <StatTile icon={TrendingUp} label="Risk Trend" value={result.risk_trend} sub={`${fmtNumber(result.current_risk_score, 0)} / 100 at the end`} />
        <StatTile icon={Timer} label="Time at HIGH+" value={fmtSec(result.time_high_risk_s)} sub="HIGH or CRITICAL windows" />
      </Section>

      <Section title="Crowd">
        <StatTile icon={Users} label="Avg Crowd Count" value={fmtNumber(result.average_crowd_count)} sub={`Peak ${fmtNumber(result.maximum_crowd_count)}`} />
        <StatTile icon={Gauge} label="Avg Density" value={fmtNumber(result.average_density, 2)} sub={`Peak ${fmtNumber(result.maximum_density, 2)} /10k px`} />
        <StatTile icon={Users} label="Avg People per m²" value={fmtNumber(result.average_people_per_m2, 2)} sub={`÷ ${result.area} m² (your input)`} />
        <StatTile icon={LayoutGrid} label="Avg Occupancy" value={result.average_occupancy == null ? "—" : fmtPercent(result.average_occupancy * 100, 0)} sub="crowd-region fraction" />
      </Section>

      <Section title="Motion">
        <StatTile icon={Wind} label="Avg Speed" value={fmtNumber(result.average_speed, 2)} sub={`Peak ${fmtNumber(result.maximum_speed, 2)} px/step`} />
        <StatTile icon={Zap} label="Acceleration" value={fmtNumber(result.average_acceleration, 3)} sub={`Peak ${fmtNumber(result.maximum_acceleration, 3)}`} />
        <StatTile icon={Shuffle} label="Direction Change" value={fmtNumber(result.average_direction_change_rate, 2)} sub="avg rate" />
        <StatTile icon={Shuffle} label="Counter-flow" value={fmtPercent((result.average_counter_flow_ratio || 0) * 100, 0)} sub="avg share of opposing motion" />
      </Section>

      <Section title="Processing">
        <StatTile icon={Activity} label="Windows Analysed" value={fmtNumber(result.windows)} sub={`${fmtNumber(result.window_seconds, 1)} s each`} />
        <StatTile icon={Activity} label="Frames Read" value={fmtNumber(result.total_frames)} sub={`${fmtNumber(result.fps, 1)} fps source → ${fmtNumber(result.analysis_fps, 1)} steps/s`} />
        <StatTile icon={Clock} label="Processing Time" value={fmtSeconds(result.processing_time)} sub={`video ${fmtSec(result.duration_s)}`} />
        <StatTile icon={CheckCircle2} label="Analysis Status" value="Completed" accentClass="text-risk-low" />
      </Section>
    </div>
  );
}
