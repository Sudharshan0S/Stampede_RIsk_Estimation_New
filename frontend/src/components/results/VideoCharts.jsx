import { useMemo } from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
  Cell,
} from "recharts";
import { Card } from "../common/Primitives";
import { riskColor, prettyTag } from "../../lib/format";
import { useMediaQuery } from "../../lib/useMediaQuery";

const C = {
  accent: "#7c5cff",
  cyan: "#3ee6de",
  amber: "#f2c94c",
  orange: "#ff9a3c",
  red: "#ff4d5e",
  soft: "#9b82ff",
  grey: "#8b93b0",
};
const AXIS = { fill: "#8b93b0", fontSize: 11 };

function ChartCard({ title, description, children, height = "h-60 sm:h-64" }) {
  return (
    <Card className="p-4 sm:p-6 min-w-0">
      <h3 className="text-sm font-semibold text-white mb-1">{title}</h3>
      {description && <p className="text-xs text-slate-500 mb-4">{description}</p>}
      <div className={height}>{children}</div>
    </Card>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-white/10 bg-base-900/95 px-3 py-2 text-xs shadow-panel max-w-[240px]">
      <div className="text-slate-400 mb-1">t = {Number(label).toFixed(1)} s</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="text-white font-medium tabular flex justify-between gap-3">
          <span className="text-slate-400">{p.name}</span>
          <span>{typeof p.value === "number" ? p.value.toFixed(2) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

export default function VideoCharts({ result, onTimeClick, selectedTime }) {
  const narrow = useMediaQuery("(max-width: 639px)"); // phones: shorter axis labels, narrower axes
  const yW = narrow ? 40 : 46;
  const rows = useMemo(() => result.timeline || [], [result.timeline]);
  const events = useMemo(() => result.events || [], [result.events]);

  const handleClick = (e) => {
    const t = e?.activeLabel;
    if (t !== undefined && onTimeClick) onTimeClick(Number(t));
  };
  const click = { onClick: handleClick, style: { cursor: onTimeClick ? "pointer" : "default" } };
  const margin = { top: 8, right: narrow ? 8 : 16, left: narrow ? 0 : -16, bottom: 0 };

  const shading = events.map((e, i) => <ReferenceArea key={i} x1={e.start_s} x2={e.end_s} fill="#ff4d5e" fillOpacity={0.12} />);
  const cursor = selectedTime != null && <ReferenceLine x={selectedTime} stroke="#ffffff" strokeDasharray="4 4" strokeWidth={1.5} />;
  const spanS = rows.length ? rows[rows.length - 1].time_s - rows[0].time_s : 0; // short clips need 1-decimal ticks, otherwise '1s 1s 2s 2s'
  const xAxis = <XAxis dataKey="time_s" type="number" domain={["dataMin", "dataMax"]} tick={AXIS} axisLine={{ stroke: "#232a45" }} tickLine={false} tickFormatter={(v) => `${Number(v).toFixed(spanS < 12 ? 1 : 0)}s`} />;

  const edges = result.models?.level_edges || [25, 50, 75];

  const stateData = useMemo(
    () => Object.entries(result.motion_state_counts || {}).map(([name, count]) => ({ name: name.replace(/_/g, " "), count })).sort((a, b) => b.count - a.count),
    [result.motion_state_counts]
  );
  const tagData = useMemo(
    () => Object.entries(result.tag_counts || {}).map(([name, count]) => ({ name: prettyTag(name), count })).sort((a, b) => b.count - a.count).slice(0, 10),
    [result.tag_counts]
  );
  const levelData = ["LOW", "MEDIUM", "HIGH", "CRITICAL"].map((n) => ({ name: n, value: result.windows_by_level?.[n] ?? 0 }));

  const evidenceData = useMemo(() => {
    const signals = ["speed", "acceleration", "direction", "occupancy", "density", "compression", "scattering", "counterflow", "convergence", "temporal_change"];
    return signals.map((s) => ({ name: s.replace("_", " "), active: rows.filter((r) => r[`active_${s}`]).length }));
  }, [rows]);

  if (rows.length === 0) return <Card className="p-8 text-center text-sm text-slate-500">No timeline data was returned for this video.</Card>;

  return (
    <div className="space-y-4 sm:space-y-6">
      <ChartCard
        title="Risk Score Over Time"
        description="Raw model score and the smoothed score (EMA + hysteresis) used for the level. Red bands = detected abnormal events. Click to jump the video."
      >
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={rows} margin={margin} {...click}>
            <defs>
              <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={C.accent} stopOpacity={0.55} />
                <stop offset="100%" stopColor={C.accent} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <ReferenceArea y1={edges[2]} y2={100} fill="#ff4d5e" fillOpacity={0.06} />
            <ReferenceArea y1={edges[1]} y2={edges[2]} fill="#ff9a3c" fillOpacity={0.05} />
            <ReferenceArea y1={edges[0]} y2={edges[1]} fill="#f2c94c" fillOpacity={0.04} />
            <CartesianGrid stroke="#1a2036" vertical={false} />
            {xAxis}
            <YAxis width={yW} domain={[0, 100]} tick={AXIS} axisLine={false} tickLine={false} />
            <Tooltip content={<ChartTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11, color: C.grey }} />
            {shading}
            <Area type="monotone" dataKey="risk_score" name="Risk (smoothed)" stroke={C.accent} fill="url(#riskFill)" strokeWidth={2} />
            <Line type="monotone" dataKey="risk_score_raw" name="Risk (raw)" stroke={C.grey} dot={false} strokeWidth={1} strokeDasharray="3 3" />
            {cursor}
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Abnormality Evidence" description="Noisy-OR combination of the ten evidence signals (0–1) and the number of independent signal groups that fired.">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={margin} {...click}>
            <CartesianGrid stroke="#1a2036" vertical={false} />
            {xAxis}
            <YAxis width={yW} yAxisId="l" domain={[0, 1]} tick={AXIS} axisLine={false} tickLine={false} />
            <YAxis width={yW} yAxisId="r" orientation="right" domain={[0, 5]} allowDecimals={false} tick={AXIS} axisLine={false} tickLine={false} />
            <Tooltip content={<ChartTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11, color: C.grey }} />
            {shading}
            <Line yAxisId="l" type="monotone" dataKey="abnormality_score" name="Abnormality score" stroke={C.red} dot={false} strokeWidth={2} />
            <Line yAxisId="r" type="stepAfter" dataKey="n_active_groups" name="Independent groups" stroke={C.amber} dot={false} strokeWidth={1.5} />
            {cursor}
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        <ChartCard title="Crowd Count & Density" description="Head count from the crowd-count model and density (people per 10k px at the working resolution).">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={margin} {...click}>
              <CartesianGrid stroke="#1a2036" vertical={false} />
              {xAxis}
              <YAxis width={yW} yAxisId="l" tick={AXIS} axisLine={false} tickLine={false} />
              <YAxis width={yW} yAxisId="r" orientation="right" tick={AXIS} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.grey }} />
              {shading}
              <Line yAxisId="l" type="monotone" dataKey="count" name="Crowd count" stroke={C.accent} dot={false} strokeWidth={2} />
              <Line yAxisId="r" type="monotone" dataKey="density" name="Density" stroke={C.cyan} dot={false} strokeWidth={2} />
              {cursor}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Crowd Speed" description="Mean and 95th-percentile optical-flow speed (pixels per analysis step at the working width).">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={margin} {...click}>
              <CartesianGrid stroke="#1a2036" vertical={false} />
              {xAxis}
              <YAxis width={yW} tick={AXIS} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.grey }} />
              {shading}
              <Line type="monotone" dataKey="mean_speed" name="Mean speed" stroke={C.soft} dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="p95_speed" name="p95 speed" stroke={C.amber} dot={false} strokeWidth={2} />
              {cursor}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Acceleration & Direction" description="Mean |acceleration| and the direction-change rate.">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={margin} {...click}>
              <CartesianGrid stroke="#1a2036" vertical={false} />
              {xAxis}
              <YAxis width={yW} tick={AXIS} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.grey }} />
              {shading}
              <Line type="monotone" dataKey="mean_abs_acceleration" name="|Acceleration|" stroke={C.orange} dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="direction_change_rate" name="Direction change" stroke={C.cyan} dot={false} strokeWidth={2} />
              {cursor}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Compression, Scattering & Counter-flow" description="Spatial motion patterns derived from the predicted flow field.">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={rows} margin={margin} {...click}>
              <CartesianGrid stroke="#1a2036" vertical={false} />
              {xAxis}
              <YAxis width={yW} tick={AXIS} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: C.grey }} />
              {shading}
              <Area type="monotone" dataKey="compression_score" name="Compression" stroke={C.red} fill={C.red} fillOpacity={0.1} strokeWidth={2} />
              <Area type="monotone" dataKey="scattering_score" name="Scattering" stroke={C.accent} fill={C.accent} fillOpacity={0.1} strokeWidth={2} />
              <Area type="monotone" dataKey="counter_flow_ratio" name="Counter-flow" stroke={C.amber} fill={C.amber} fillOpacity={0.08} strokeWidth={2} />
              {cursor}
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Windows per Risk Level" description="How many temporal windows fell into each smoothed risk level.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={levelData} margin={margin}>
              <CartesianGrid stroke="#1a2036" vertical={false} />
              <XAxis dataKey="name" tick={AXIS} axisLine={{ stroke: "#232a45" }} tickLine={false} />
              <YAxis width={yW} allowDecimals={false} tick={AXIS} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="value" name="Windows" radius={[6, 6, 0, 0]}>
                {levelData.map((d) => (
                  <Cell key={d.name} fill={riskColor(d.name)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Active Evidence Signals" description="Windows in which each abnormality signal crossed its validated threshold.">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={evidenceData} layout="vertical" margin={{ top: 8, right: 12, left: narrow ? 0 : 24, bottom: 0 }}>
              <CartesianGrid stroke="#1a2036" horizontal={false} />
              <XAxis type="number" allowDecimals={false} tick={AXIS} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={narrow ? 70 : 92} tick={AXIS} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="active" name="Windows" fill={C.accent} radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Motion States" description="Motion-state classifier output per window (only reported while risk or abnormality is raised; otherwise NORMAL).">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stateData} layout="vertical" margin={{ top: 8, right: 12, left: narrow ? 0 : 40, bottom: 0 }}>
              <CartesianGrid stroke="#1a2036" horizontal={false} />
              <XAxis type="number" allowDecimals={false} tick={AXIS} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={narrow ? 86 : 140} tick={AXIS} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="count" name="Windows" fill={C.cyan} radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Most Frequent Abnormal Tags" description="Tags assigned by the abnormality detector (a window can carry several).">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={tagData} layout="vertical" margin={{ top: 8, right: 12, left: narrow ? 0 : 40, bottom: 0 }}>
              <CartesianGrid stroke="#1a2036" horizontal={false} />
              <XAxis type="number" allowDecimals={false} tick={AXIS} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={narrow ? 93 : 150} tick={AXIS} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="count" name="Windows" fill={C.orange} radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}
