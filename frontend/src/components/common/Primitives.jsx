export function Card({ children, className = "", ...props }) {
  return (
    <div
      className={`glass-panel rounded-2xl shadow-panel ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function SectionHeading({ eyebrow, title, description, action }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        {eyebrow && (
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-accent-soft/80 mb-1">
            {eyebrow}
          </p>
        )}
        <h2 className="text-xl sm:text-2xl font-display font-semibold text-white break-words sm:truncate">
          {title}
        </h2>
        {description && (
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">{description}</p>
        )}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

export function StatTile({ icon: Icon, label, value, sub, accentClass = "text-white" }) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-base-800/60 p-4 min-w-0">
      <div className="flex items-center gap-2 text-slate-400 text-xs font-medium uppercase tracking-wide mb-2">
        {Icon && <Icon size={14} className="shrink-0" />}
        <span className="truncate">{label}</span>
      </div>
      <div title={typeof value === "string" ? value : undefined} className={`text-xl xs:text-2xl font-display font-semibold tabular ${accentClass} truncate`}>
        {value}
      </div>
      {sub && <div className="text-xs text-slate-500 mt-1 truncate">{sub}</div>}
    </div>
  );
}

export function Badge({ children, tone = "neutral", className = "" }) {
  const tones = {
    neutral: "bg-white/5 text-slate-300 border-white/10",
    accent: "bg-accent/15 text-accent-soft border-accent/30",
    success: "bg-risk-low/15 text-risk-low border-risk-low/30",
    warning: "bg-risk-medium/15 text-risk-medium border-risk-medium/30",
    danger: "bg-risk-critical/15 text-risk-critical border-risk-critical/30",
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium whitespace-nowrap ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function Button({
  children,
  variant = "primary",
  className = "",
  icon: Icon,
  ...props
}) {
  const variants = {
    primary:
      "bg-accent hover:bg-accent-soft text-white shadow-glow disabled:opacity-40 disabled:cursor-not-allowed",
    ghost:
      "bg-white/[0.04] hover:bg-white/[0.08] text-slate-200 border border-white/10 disabled:opacity-40",
    subtle: "bg-transparent hover:bg-white/[0.06] text-slate-300",
    danger: "bg-risk-critical/15 hover:bg-risk-critical/25 text-risk-critical border border-risk-critical/30",
  };
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 coarse:min-h-[44px] text-sm font-semibold transition-colors ${variants[variant]} ${className}`}
      {...props}
    >
      {Icon && <Icon size={16} className="shrink-0" />}
      <span className="truncate">{children}</span>
    </button>
  );
}

export function riskTone(level) {
  switch (String(level || "").toUpperCase()) {
    case "LOW":
      return "success";
    case "MEDIUM":
      return "warning";
    case "HIGH":
      return "warning";
    case "CRITICAL":
      return "danger";
    default:
      return "neutral";
  }
}
