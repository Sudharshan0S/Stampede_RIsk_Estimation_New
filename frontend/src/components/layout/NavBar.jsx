import { Upload, Activity, BarChart3, Info, RotateCcw, Bell } from "lucide-react";
import { useApp } from "../../context/AppContext";

const TABS = [
  { key: "input", label: "Input", icon: Upload },
  { key: "analysis", label: "Analysis", icon: Activity },
  { key: "results", label: "Results", icon: BarChart3 },
  { key: "about", label: "About", icon: Info },
];

export default function NavBar() {
  const { tab, setTab, status, mediaKind, result, reset, step } = useApp();

  const canGoAnalysis = status === "analyzing" || status === "done" || status === "error";
  const canGoResults = status === "done" || status === "error";

  const disabledFor = (key) => {
    if (key === "analysis") return !canGoAnalysis;
    if (key === "results") return !canGoResults;
    return false;
  };

  const badge =
    status === "analyzing"
      ? { text: "Analysis Running", tone: "text-cyan-glow border-cyan-glow/30 bg-cyan-glow/10" }
      : status === "done"
      ? { text: "Analysis Complete", tone: "text-risk-low border-risk-low/30 bg-risk-low/10" }
      : status === "error"
      ? { text: "Analysis Failed", tone: "text-risk-critical border-risk-critical/30 bg-risk-critical/10" }
      : status === "cancelled"
      ? { text: "Analysis Cancelled", tone: "text-slate-400 border-white/10 bg-white/5" }
      : { text: "Waiting for Upload", tone: "text-slate-400 border-white/10 bg-white/5" };

  return (
    <div className="safe-top sticky top-0 z-30 border-b border-white/[0.06] bg-base-950/85 backdrop-blur-md">
      <div className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-10">
        <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 py-2.5 sm:py-3">
          <div className="flex items-center gap-3 min-w-0 flex-1 sm:flex-none">
            <div className="grid place-items-center h-9 w-9 shrink-0 rounded-xl bg-gradient-to-br from-accent to-cyan-glow text-base-950 font-display font-bold">
              CV
            </div>
            <div className="min-w-0">
              <div className="text-sm font-display font-semibold text-white truncate">CrowdVision AI</div>
              <div className="text-[11px] text-slate-500 truncate">Smart Crowd Intelligence</div>
            </div>
          </div>

          <nav className="order-last sm:order-none w-full sm:w-auto flex items-center gap-1 rounded-xl border border-white/[0.06] bg-base-800/60 p-1 overflow-x-auto">
            {TABS.map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                disabled={disabledFor(key)}
                onClick={() => setTab(key)}
                className={`flex-1 sm:flex-none justify-center inline-flex items-center gap-1.5 whitespace-nowrap rounded-lg px-2.5 sm:px-3 py-2 sm:py-1.5 coarse:min-h-[44px] text-xs xs:text-sm font-medium transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${
                  tab === key ? "bg-accent text-white shadow-glow" : "text-slate-300 hover:bg-white/[0.06]"
                }`}
              >
                <Icon size={14} /> {label}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-2 shrink-0">
            <span className={`hidden sm:inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium ${badge.tone}`}>
              <Bell size={12} /> {badge.text}
            </span>
            <button
              onClick={reset}
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 sm:py-1.5 coarse:min-h-[44px] text-xs font-medium text-slate-300 hover:bg-white/[0.08] transition-colors"
            >
              <RotateCcw size={13} /> Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
