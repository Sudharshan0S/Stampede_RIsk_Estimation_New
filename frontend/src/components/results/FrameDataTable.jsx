import { useMemo, useState } from "react";
import { Search, ArrowUpDown, ArrowUp, ArrowDown, Download, Rows3 } from "lucide-react";
import { Card, Badge, riskTone, Button } from "../common/Primitives";
import { fmtNumber } from "../../lib/format";

// One row per temporal window (window_len steps, hop steps apart) - the unit the risk model works on.
const COLUMNS = [
  { key: "time_s", label: "Time (s)", digits: 1 },
  { key: "count", label: "Crowd", digits: 0 },
  { key: "density", label: "Density", digits: 2 },
  { key: "mean_speed", label: "Speed", digits: 2 },
  { key: "mean_abs_acceleration", label: "Accel", digits: 3 },
  { key: "direction_change_rate", label: "Dir. change", digits: 2 },
  { key: "counter_flow_ratio", label: "Counter-flow", digits: 2 },
  { key: "abnormality_score", label: "Abnormality", digits: 2 },
  { key: "risk_score", label: "Risk", digits: 1 },
  { key: "risk_level", label: "Level" },
  { key: "motion_state", label: "Motion state" },
  { key: "tags", label: "Tags" },
];

function toCsv(rows) {
  if (!rows.length) return "";
  const keys = Object.keys(rows[0]);
  const esc = (v) => {
    const s = Array.isArray(v) ? v.join(";") : String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return `${keys.join(",")}\n${rows.map((r) => keys.map((k) => esc(r[k])).join(",")).join("\n")}`;
}

function downloadClientCsv(rows, filename) {
  const url = URL.createObjectURL(new Blob([toCsv(rows)], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

const tagText = (r) => (r.tags || []).filter((t) => t !== "NORMAL").map((t) => t.replace(/_/g, " ").toLowerCase()).join(", ");

export default function FrameDataTable({ result, selectedTime, onTimeClick }) {
  const all = useMemo(() => result.timeline || [], [result.timeline]);
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState("time_s");
  const [sortDir, setSortDir] = useState("asc");
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return all;
    return all.filter(
      (r) =>
        String(r.time_s).includes(q) ||
        (r.risk_level || "").toLowerCase().includes(q) ||
        (r.motion_state || "").toLowerCase().includes(q) ||
        tagText(r).includes(q)
    );
  }, [all, search]);

  const sorted = useMemo(() => {
    const dir = sortDir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const av = sortKey === "tags" ? tagText(a) : a[sortKey];
      const bv = sortKey === "tags" ? tagText(b) : b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });
  }, [filtered, sortKey, sortDir]);

  const pages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const cur = Math.min(page, pages - 1);
  const slice = sorted.slice(cur * pageSize, cur * pageSize + pageSize);

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  if (all.length === 0) return <Card className="p-8 text-center text-sm text-slate-500">No window data was returned.</Card>;

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between gap-3 flex-wrap px-4 sm:px-5 py-4 border-b border-white/[0.06]">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Rows3 size={15} className="text-accent-soft" /> Window Data
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            {all.length} temporal windows ({result.window_seconds?.toFixed?.(1)} s each). Click a row to jump the video to that moment.
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap w-full sm:w-auto">
          <label className="flex flex-1 sm:flex-none items-center gap-2 rounded-lg border border-white/10 bg-base-900/60 px-2.5 py-1.5 coarse:min-h-[44px] focus-within:border-accent cursor-text">
            <Search size={13} className="text-slate-500" />
            <input
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(0);
              }}
              placeholder="Search level, state, tag…"
              className="bg-transparent text-xs text-white outline-none w-full sm:w-44 min-w-0"
            />
          </label>
          <Button variant="ghost" icon={Download} onClick={() => downloadClientCsv(sorted, "windows.csv")}>
            CSV
          </Button>
        </div>
      </div>

      <div className="overflow-x-auto overscroll-x-contain">
        <table className="w-full min-w-[820px] text-xs">
          <thead>
            <tr className="text-left text-slate-500 border-b border-white/[0.06]">
              {COLUMNS.map((c) => (
                <th key={c.key} className="px-3 py-2.5 font-medium whitespace-nowrap">
                  <button onClick={() => toggleSort(c.key)} className="inline-flex items-center gap-1 coarse:min-h-[44px] coarse:min-w-[44px] hover:text-white">
                    {c.label}
                    {sortKey === c.key ? sortDir === "asc" ? <ArrowUp size={11} /> : <ArrowDown size={11} /> : <ArrowUpDown size={11} className="opacity-40" />}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {slice.map((r) => {
              const active = selectedTime != null && Math.abs(r.time_s - selectedTime) < 1e-6;
              return (
                <tr
                  key={r.window_index}
                  onClick={() => onTimeClick?.(r.time_s)}
                  className={`border-b border-white/[0.04] cursor-pointer hover:bg-white/[0.04] ${active ? "bg-accent/10" : ""} ${r.abnormal_event ? "border-l-2 border-l-risk-critical" : ""}`}
                >
                  {COLUMNS.map((c) => (
                    <td key={c.key} className="px-3 py-2.5 coarse:py-3 whitespace-nowrap tabular text-slate-300">
                      {c.key === "risk_level" ? (
                        <Badge tone={riskTone(r.risk_level)}>{r.risk_level}</Badge>
                      ) : c.key === "motion_state" ? (
                        (r.motion_state || "").replace(/_/g, " ")
                      ) : c.key === "tags" ? (
                        <span className="text-slate-400 max-w-[260px] truncate inline-block align-bottom">{tagText(r) || "—"}</span>
                      ) : (
                        fmtNumber(r[c.key], c.digits)
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between gap-3 px-4 sm:px-5 py-3 text-xs text-slate-400 flex-wrap">
        <span>
          {sorted.length} rows · page {cur + 1} / {pages}
        </span>
        <div className="flex items-center gap-2">
          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(0);
            }}
            className="rounded-md border border-white/10 bg-base-900/60 px-2 py-1 coarse:min-h-[44px] text-xs text-white"
          >
            {[25, 50, 100].map((n) => (
              <option key={n} value={n}>
                {n} / page
              </option>
            ))}
          </select>
          <Button variant="ghost" onClick={() => setPage(Math.max(0, cur - 1))} disabled={cur === 0}>
            Prev
          </Button>
          <Button variant="ghost" onClick={() => setPage(Math.min(pages - 1, cur + 1))} disabled={cur >= pages - 1}>
            Next
          </Button>
        </div>
      </div>
    </Card>
  );
}
