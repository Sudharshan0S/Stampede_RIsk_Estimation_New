import { useCallback, useRef, useState } from "react";
import { UploadCloud, FileImage, FileVideo, X, PlayCircle, Info, Users, AlertTriangle, Loader2 } from "lucide-react";
import { useApp } from "../context/AppContext";
import { Card, SectionHeading, Button } from "../components/common/Primitives";
import { fmtBytes } from "../lib/format";

const RECOMMENDATIONS = [
  "Clear CCTV footage",
  "Stable camera angle",
  "Visible crowd region",
  "Enter accurate crowd area",
];

export default function InputPage() {
  const {
    file,
    previewUrl,
    previewLoading,
    previewNote,
    mediaKind,
    area,
    setArea,
    selectFile,
    clearFile,
    runAnalysis,
    error,
    status,
    modelsError,
  } = useApp();
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragging(false);
      const dropped = e.dataTransfer.files?.[0];
      if (dropped) selectFile(dropped);
    },
    [selectFile]
  );

  const onPick = (e) => {
    const picked = e.target.files?.[0];
    if (picked) selectFile(picked);
    e.target.value = "";
  };

  return (
    <div className="space-y-6">
      <SectionHeading
        eyebrow="Step 01"
        title="Upload Media"
        description="Upload an image or surveillance video for AI powered crowd density and stampede risk estimation."
      />

      {modelsError && (
        <div className="rounded-xl border border-risk-medium/30 bg-risk-medium/10 px-4 py-3 text-sm text-risk-medium whitespace-pre-line">
          <span className="font-semibold">Trained models are not loaded.</span> {modelsError}
          {"\n"}Copy the files from crowdrisk_outputs/models into the project's models/ folder and restart the backend (see README).
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Left column: dropzone + recommendations */}
        <div className="space-y-6 min-w-0">
          <Card
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`p-6 sm:p-10 flex flex-col items-center justify-center text-center cursor-pointer border-dashed transition-colors min-h-[220px] sm:min-h-[280px] ${
              dragging ? "border-accent bg-accent/5" : "border-white/10"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.bmp,.mp4,.avi,.mov,.mkv"
              className="hidden"
              onChange={onPick}
            />
            <div className="grid place-items-center h-16 w-16 rounded-2xl bg-accent/10 text-accent-soft mb-5">
              <UploadCloud size={30} />
            </div>
            <h3 className="text-lg font-display font-semibold text-white">Drag &amp; Drop Files</h3>
            <p className="text-sm text-slate-400 mt-1">Click here or drag your media file.</p>
            <p className="text-xs text-slate-500 mt-3 tracking-wide uppercase">
              JPG · JPEG · PNG · BMP · MP4 · AVI · MOV · MKV
            </p>
          </Card>

          <Card className="p-5">
            <div className="flex items-center gap-2 text-sm font-semibold text-white mb-3">
              <Info size={15} className="text-accent-soft" /> Recommended
            </div>
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {RECOMMENDATIONS.map((r) => (
                <li key={r} className="flex items-center gap-2 text-sm text-slate-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-accent shrink-0" /> {r}
                </li>
              ))}
            </ul>
          </Card>

          {error && (
            <div className="flex items-start gap-2 rounded-xl border border-risk-critical/30 bg-risk-critical/10 px-4 py-3 text-sm text-risk-critical">
              <AlertTriangle size={16} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right column: preview + config */}
        <div className="space-y-6 min-w-0">
          {!file ? (
            <Card className="p-6 sm:p-10 min-h-[160px] sm:min-h-[280px] flex flex-col items-center justify-center text-center text-slate-500">
              <Users size={32} className="mb-3 opacity-40" />
              <p className="text-sm">No media selected yet.</p>
              <p className="text-xs mt-1">Your selected image or video preview will appear here.</p>
            </Card>
          ) : (
            <Card className="overflow-hidden">
              <div className="relative bg-black/40">
                {mediaKind === "image" ? (
                  <img src={previewUrl} alt="Selected preview" className="w-full preview-max object-contain" />
                ) : previewNote ? (
                  <div className="w-full h-[180px] sm:h-[220px] flex flex-col items-center justify-center gap-2 text-center px-6">
                    <FileVideo size={28} className="text-slate-600" />
                    <p className="text-xs text-slate-500 max-w-sm">{previewNote}</p>
                  </div>
                ) : (
                  <video src={previewUrl} controls playsInline preload="metadata" className="w-full preview-max object-contain" />
                )}

                {previewLoading && (
                  <div className="absolute inset-0 bg-black/50 backdrop-blur-[1px] flex flex-col items-center justify-center gap-2">
                    <Loader2 size={22} className="text-accent-soft animate-spin" />
                    <p className="text-xs text-slate-300">Preparing preview…</p>
                  </div>
                )}

                <button
                  onClick={clearFile}
                  className="absolute top-2 right-2 sm:top-3 sm:right-3 grid place-items-center h-10 w-10 coarse:h-11 coarse:w-11 rounded-full bg-black/60 text-white hover:bg-black/80 transition-colors"
                  aria-label="Remove file"
                >
                  <X size={16} />
                </button>
              </div>
              <div className="p-4 flex items-center gap-3 border-t border-white/[0.06]">
                {mediaKind === "image" ? (
                  <FileImage size={18} className="text-accent-soft shrink-0" />
                ) : (
                  <FileVideo size={18} className="text-accent-soft shrink-0" />
                )}
                <div className="min-w-0">
                  <p className="text-sm font-medium text-white break-all line-clamp-2">{file.name}</p>
                  <p className="text-xs text-slate-500">
                    {fmtBytes(file.size)} · {mediaKind === "image" ? "Image" : "Video"}
                  </p>
                </div>
              </div>
            </Card>
          )}

          <Card className="p-5 space-y-4">
            <h3 className="text-sm font-semibold text-white">Analysis Configuration</h3>
            <div>
              <label className="text-xs text-slate-400 mb-1.5 block">Crowd Area (m²)</label>
              <label className="flex items-center gap-2 rounded-lg border border-white/10 bg-base-900/60 px-3 py-2.5 coarse:min-h-[44px] focus-within:border-accent cursor-text">
                <Users size={16} className="text-slate-500 shrink-0" />
                <input
                  type="number"
                  inputMode="decimal"
                  min={1}
                  step="0.5"
                  value={area}
                  onChange={(e) => setArea(Math.max(1, Number(e.target.value) || 1))}
                  className="w-full min-w-0 bg-transparent text-sm text-white outline-none tabular"
                />
              </label>
              <p className="text-xs text-slate-500 mt-1.5">
                Optional. The real-world floor area the camera covers; only used to display people per m². The risk model itself works in image space and does not use it.
              </p>
            </div>

            {file && (
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="rounded-lg bg-base-900/60 border border-white/[0.06] px-3 py-2">
                  <div className="text-[11px] text-slate-500">File Type</div>
                  <div className="text-sm font-medium text-white capitalize">{mediaKind}</div>
                </div>
                <div className="rounded-lg bg-base-900/60 border border-white/[0.06] px-3 py-2">
                  <div className="text-[11px] text-slate-500">Size</div>
                  <div className="text-sm font-medium text-white">{fmtBytes(file.size)}</div>
                </div>
              </div>
            )}

            <div className="flex flex-col sm:flex-row gap-3 pt-2">
              <Button variant="ghost" className="flex-1" onClick={clearFile} disabled={!file}>
                Remove
              </Button>
              <Button
                variant="primary"
                className="flex-1"
                icon={PlayCircle}
                onClick={runAnalysis}
                disabled={!file || status === "analyzing"}
              >
                Analyze Media
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
