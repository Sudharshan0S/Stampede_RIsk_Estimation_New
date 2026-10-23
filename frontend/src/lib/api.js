import axios from "axios";

// Flask backend origin.
//   development (npm run dev):  VITE_API_BASE_URL, default http://localhost:5000
//   production build (Vercel):  VITE_API_BASE_URL MUST be set at build time (e.g. https://crowdvision-backend.onrender.com);
//                               if it is empty the app talks to its own origin (only useful behind a reverse proxy that forwards to the backend).
function resolveApiBase() {
  let base = (import.meta.env.VITE_API_BASE_URL || "").trim();
  if (!base) base = import.meta.env.DEV ? "http://localhost:5000" : "";
  if (base && !/^https?:\/\//i.test(base) && !base.startsWith("/")) base = `https://${base}`; // "my-api.onrender.com" -> https://my-api.onrender.com
  return base.replace(/\/+$/, ""); // no trailing slash
}
export const API_BASE_URL = resolveApiBase();

const client = axios.create({
  baseURL: API_BASE_URL,
});

/** Resolves a backend-relative URL (e.g. "/outputs/visualizations/x.jpg")
 * into a fully qualified URL the browser/img/video tag can load. */
export function resolveMediaUrl(path) {
  if (!path) return null;
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? "" : "/"}${path}`;
}

export async function checkHealth() {
  const { data } = await client.get("/health");
  return data;
}

export async function fetchProgress(since) {
  const params = since !== undefined && since !== null ? { since } : {};
  const { data } = await client.get("/progress", { params });
  return data;
}

export async function previewVideo(file) {
  const form = new FormData();
  form.append("video", file);
  const { data } = await client.post("/preview/video", form);
  return data;
}

export async function cancelAnalysis() {
  const { data } = await client.post("/cancel");
  return data;
}

export async function analyzeImage(file, area, { signal } = {}) {
  const form = new FormData();
  form.append("image", file);
  form.append("area", area);
  const { data } = await client.post("/predict/image", form, { signal });
  return data;
}

export async function analyzeVideo(file, area, { signal } = {}) {
  const form = new FormData();
  form.append("video", file);
  form.append("area", area);
  const { data } = await client.post("/predict/video", form, { signal });
  return data;
}

/** Fetches a backend output file as a Blob and triggers a native browser
 * download with the given filename, regardless of response headers. */
export async function downloadOutput(path, filename) {
  const url = resolveMediaUrl(path);
  if (!url) throw new Error("Nothing to download yet.");

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Download failed (${response.status})`);

  const blob = await response.blob();
  const objectUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename || path.split("/").pop();
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(objectUrl);
}
