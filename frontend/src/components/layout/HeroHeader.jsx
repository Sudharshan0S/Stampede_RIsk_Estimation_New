import { Sparkles, Wifi, Cpu, Check } from "lucide-react";
import { useApp } from "../../context/AppContext";
import TopStatusBar from "./TopStatusBar";

const STEPS = [
  { key: "input", label: "Upload", index: "01" },
  { key: "analysis", label: "Analysis", index: "02" },
  { key: "results", label: "Results", index: "03" },
];

export default function HeroHeader() {
  const { step, backendOnline } = useApp();
  const activeIndex = STEPS.findIndex((s) => s.key === step);

  return (
    <header className="px-4 sm:px-6 lg:px-10 pt-6 pb-5">
      <div className="mx-auto max-w-[1400px]">
        <div className="flex flex-col xl:flex-row xl:items-start gap-6 xl:gap-8">
          {/* Brand block */}
          <div className="flex-1 min-w-0">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs font-medium text-accent-soft mb-3">
              <Sparkles size={12} /> AI Analytics Platform
            </span>
            <h1 className="text-2xl sm:text-3xl lg:text-[2.15rem] font-display font-semibold text-white leading-tight">
              Crowd Analysis &amp; Stampede Risk Estimation
            </h1>
            <p className="text-slate-400 text-sm sm:text-base mt-2 max-w-xl">
              Intelligent crowd density, motion behaviour and stampede risk assessment dashboard.
            </p>
          </div>

          {/* Live status tiles */}
          <div className="w-full xl:w-[620px] shrink-0">
            <TopStatusBar />
          </div>
        </div>

        {/* Connection pills + step progress */}
        <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${
                backendOnline
                  ? "border-risk-low/30 bg-risk-low/10 text-risk-low"
                  : backendOnline === false
                  ? "border-risk-critical/30 bg-risk-critical/10 text-risk-critical"
                  : "border-white/10 bg-white/5 text-slate-400"
              }`}
            >
              <Wifi size={13} /> Backend {backendOnline ? "Connected" : backendOnline === false ? "Offline" : "Checking"}
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-cyan-glow/30 bg-cyan-glow/10 px-3 py-1.5 text-xs font-medium text-cyan-glow">
              <Cpu size={13} /> AI Engine Ready
            </span>
          </div>

          <ol className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
            {STEPS.map((s, i) => {
              const isActive = i === activeIndex;
              const isDone = i < activeIndex;
              return (
                <li key={s.key} className="flex items-center gap-1.5 sm:gap-2">
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold border transition-colors ${
                      isActive
                        ? "bg-accent text-white border-accent shadow-glow"
                        : isDone
                        ? "bg-risk-low/15 text-risk-low border-risk-low/30"
                        : "bg-white/[0.03] text-slate-500 border-white/10"
                    }`}
                  >
                    {isDone ? <Check size={12} /> : s.index} {s.label}
                  </span>
                  {i < STEPS.length - 1 && <span className="h-px w-3 sm:w-5 bg-white/15" />}
                </li>
              );
            })}
          </ol>
        </div>
      </div>
    </header>
  );
}
