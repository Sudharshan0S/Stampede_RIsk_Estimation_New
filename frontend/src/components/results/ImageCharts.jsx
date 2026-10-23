import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Card } from "../common/Primitives";
import { fmtNumber } from "../../lib/format";

const AXIS = { fill: "#8b93b0", fontSize: 11 };

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-white/10 bg-base-900/95 px-3 py-2 text-xs shadow-panel">
      <div className="text-slate-400 mb-1">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="text-white font-medium tabular">
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
        </div>
      ))}
    </div>
  );
}

const REGIMES = ["sparse", "medium", "dense", "extreme"];
const REGIME_COLORS = ["#2fd67c", "#f2c94c", "#ff9a3c", "#ff4d5e"];

export default function ImageCharts({ result }) {
  const countData = [
    { name: "Accepted heads", value: result.crowd_count ?? 0, color: "#2fd67c" },
    { name: "Uncertain", value: result.n_uncertain ?? 0, color: "#f2c94c" },
    { name: "Density-map sum", value: result.density_map_sum ?? 0, color: "#7c5cff" },
  ];
  const edges = result.regime_edges || [];
  const idx = REGIMES.indexOf(result.crowd_regime);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
      <Card className="p-4 sm:p-6 min-w-0">
        <h3 className="text-sm font-semibold text-white mb-1">Crowd Count Estimate</h3>
        <p className="text-xs text-slate-500 mb-4">
          Final count = accepted predicted head points. The density-map sum is only a consistency check; uncertain points are not counted.
        </p>
        <div className="h-56 sm:h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={countData} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
              <CartesianGrid stroke="#1a2036" vertical={false} />
              <XAxis dataKey="name" interval={0} tick={{ ...AXIS, fontSize: 10 }} axisLine={{ stroke: "#232a45" }} tickLine={false} />
              <YAxis tick={AXIS} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="value" name="People" radius={[6, 6, 0, 0]}>
                {countData.map((d) => (
                  <Cell key={d.name} fill={d.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card className="p-4 sm:p-6 min-w-0">
        <h3 className="text-sm font-semibold text-white mb-1">Crowd Regime</h3>
        <p className="text-xs text-slate-500 mb-5">
          Density = {fmtNumber(result.density, 2)} people per 10k px at the {result.models?.work_width ?? 640} px working width, placed against the regime edges fitted on the training scenes.
        </p>
        <div className="flex gap-1.5">
          {REGIMES.map((r, i) => (
            <div key={r} className="flex-1">
              <div
                className="h-3 rounded-full"
                style={{ background: REGIME_COLORS[i], opacity: i === idx ? 1 : 0.22, boxShadow: i === idx ? `0 0 14px ${REGIME_COLORS[i]}` : "none" }}
              />
              <div className={`mt-2 text-[11px] uppercase tracking-wide ${i === idx ? "text-white font-semibold" : "text-slate-500"}`}>{r}</div>
              <div className="text-[9px] xs:text-[10px] text-slate-600 tabular break-words">
                {i === 0 ? `< ${fmtNumber(edges[0], 2)}` : i === 3 ? `≥ ${fmtNumber(edges[2], 2)}` : `${fmtNumber(edges[i - 1], 2)} – ${fmtNumber(edges[i], 2)}`}
              </div>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-500 mt-6 leading-relaxed">
          The motion-risk model needs movement, so a single image is not given a stampede-risk score. Upload a video for risk, abnormal events and motion analysis.
        </p>
      </Card>
    </div>
  );
}
