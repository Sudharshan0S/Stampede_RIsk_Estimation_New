export function fmtNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(digits)}%`;
}

export function fmtBytes(bytes) {
  if (!bytes && bytes !== 0) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let val = bytes;
  while (val >= 1024 && i < units.length - 1) {
    val /= 1024;
    i += 1;
  }
  return `${val.toFixed(val < 10 && i > 0 ? 2 : 0)} ${units[i]}`;
}

export function fmtSeconds(seconds) {
  if (!seconds && seconds !== 0) return "—";
  const s = Math.round(seconds);
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

/** 12.3 -> "12.3 s" */
export function fmtSec(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(digits)} s`;
}

/** "SUDDEN_SPEED_CHANGE" -> "Sudden speed change" */
export function prettyTag(tag) {
  const s = String(tag || "").replace(/_/g, " ").toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export const RISK_COLORS = {
  LOW: "#2fd67c",
  MEDIUM: "#f2c94c",
  HIGH: "#ff9a3c",
  CRITICAL: "#ff4d5e",
  "N/A": "#8b93b0",
};

export function riskColor(level) {
  return RISK_COLORS[String(level || "").toUpperCase()] || "#7c5cff";
}

/** Parses the "risk_reason" string ("• A \n • B") into a clean array. */
export function parseReasons(reasonString) {
  if (!reasonString) return [];
  return reasonString
    .split("•")
    .map((r) => r.trim())
    .filter(Boolean);
}

export function nowClock() {
  return new Date().toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
