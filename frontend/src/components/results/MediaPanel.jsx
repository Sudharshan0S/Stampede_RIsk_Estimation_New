import { useState, forwardRef } from "react";
import { Maximize2, X, Waves, LayoutGrid, ExternalLink, Download, RefreshCcw, Users, Siren } from "lucide-react";
import { Card, Badge, riskTone, Button } from "../common/Primitives";
import { resolveMediaUrl, downloadOutput } from "../../lib/api";
import { useApp } from "../../context/AppContext";

function InfoChip({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/[0.06] bg-base-900/50 px-3 py-2 min-w-0">
      <Icon size={14} className="text-accent-soft shrink-0" />
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
        <div className="text-xs font-medium text-white truncate">{value}</div>
      </div>
    </div>
  );
}

const MediaPanel = forwardRef(function MediaPanel({ result }, videoRef) {
  const { previewUrl } = useApp();
  const [expanded, setExpanded] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);
  const isVideo = result.media_type === "video";
  // images: the backend writes three renderings (predictions, density heat-map, 2x2 diagnostic panel)
  const [variant, setVariant] = useState("annotated_image");
  const imageVariants = [
    { key: "annotated_image", label: "Head points" },
    { key: "density_image", label: "Density map" },
    { key: "panel_image", label: "Panel" },
  ].filter((v) => result.downloads?.[v.key]);
  const imageUrl = result.downloads?.[variant] || result.output_image_url;
  const annotatedUrl = resolveMediaUrl(isVideo ? result.output_video_url : imageUrl);
  const activeUrl = showOriginal && previewUrl ? previewUrl : annotatedUrl;

  return (
    <>
      <Card className="overflow-hidden">
        <div className="flex items-center justify-between gap-3 px-4 sm:px-5 py-4 border-b border-white/[0.06] flex-wrap">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-white">Annotated {isVideo ? "Video" : "Image"}</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              {isVideo
                ? "Optical-flow arrows, detected heads and the live risk HUD, rendered by the backend."
                : "Green = accepted head points, yellow = uncertain. Switch to the density heat-map or the 2×2 panel."}
            </p>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {isVideo ? <Badge tone={riskTone(result.risk_level)}>Peak {result.risk_level}</Badge> : <Badge tone="accent">{result.crowd_count} people</Badge>}
            {!isVideo && imageVariants.length > 1 && !showOriginal && (
              <div className="inline-flex max-w-full rounded-lg border border-white/10 overflow-hidden">
                {imageVariants.map((v) => (
                  <button
                    key={v.key}
                    onClick={() => setVariant(v.key)}
                    className={`px-2.5 py-1.5 coarse:min-h-[44px] text-xs transition-colors ${variant === v.key ? "bg-accent text-white" : "text-slate-300 hover:bg-white/[0.06]"}`}
                  >
                    {v.label}
                  </button>
                ))}
              </div>
            )}
            {previewUrl && (
              <button
                onClick={() => setShowOriginal((v) => !v)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 coarse:min-h-[44px] text-xs text-slate-300 hover:text-white hover:border-accent/40 transition-colors"
              >
                <RefreshCcw size={13} /> {showOriginal ? "Show Annotated" : "Show Original"}
              </button>
            )}
            {annotatedUrl && (
              <>
                <a
                  href={annotatedUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="grid place-items-center h-8 w-8 coarse:h-11 coarse:w-11 rounded-lg border border-white/10 text-slate-400 hover:text-white hover:border-accent/40 transition-colors"
                  aria-label="Open original resolution"
                  title="Open original resolution in a new tab"
                >
                  <ExternalLink size={14} />
                </a>
                <button
                  onClick={() => setExpanded(true)}
                  className="grid place-items-center h-8 w-8 coarse:h-11 coarse:w-11 rounded-lg border border-white/10 text-slate-400 hover:text-white hover:border-accent/40 transition-colors"
                  aria-label="Fullscreen / zoom"
                  title="Fullscreen / zoom"
                >
                  <Maximize2 size={14} />
                </button>
              </>
            )}
          </div>
        </div>

        {/* No max-height cap here — object-contain never crops, it only
            letterboxes, so the full annotated frame is always visible. */}
        <div className="bg-black/40 flex items-center justify-center">
          {!activeUrl ? (
            <div className="h-48 sm:h-72 grid place-items-center text-sm text-slate-600">Annotated media unavailable.</div>
          ) : isVideo && !showOriginal ? (
            <video
              ref={videoRef}
              src={activeUrl}
              controls
              playsInline
              preload="metadata"
              className="block w-full h-auto media-max object-contain bg-black"
            />
          ) : isVideo && showOriginal ? (
            <video src={activeUrl} controls playsInline preload="metadata" className="block w-full h-auto media-max object-contain bg-black" />
          ) : (
            <img
              src={activeUrl}
              alt={showOriginal ? "Original upload" : "Annotated result"}
              className="block w-full h-auto media-max object-contain cursor-zoom-in"
              onClick={() => setExpanded(true)}
            />
          )}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 p-4">
          {isVideo ? (
            <>
              <InfoChip icon={Waves} label="Dominant motion state" value={(result.dominant_motion_state || "—").replace(/_/g, " ")} />
              <InfoChip icon={Siren} label="Abnormal events" value={result.n_events ?? 0} />
              <InfoChip icon={LayoutGrid} label="Flow estimator" value={result.models?.flow_estimator || "—"} />
            </>
          ) : (
            <>
              <InfoChip icon={Users} label="Crowd count" value={`${result.crowd_count} (+${result.n_uncertain} uncertain)`} />
              <InfoChip icon={LayoutGrid} label="Crowd regime" value={result.crowd_regime || "—"} />
              <InfoChip icon={Waves} label="Count uncertainty" value={result.uncertainty?.level || "—"} />
            </>
          )}
        </div>

        {annotatedUrl && (
          <div className="px-4 pb-4">
            <Button className="w-full xs:w-auto"
              variant="ghost"
              icon={Download}
              onClick={() =>
                downloadOutput(
                  isVideo ? result.output_video_url : imageUrl,
                  isVideo ? "annotated_video.mp4" : "annotated_image.jpg"
                )
              }
            >
              Download {isVideo ? "Video" : "Image"}
            </Button>
          </div>
        )}
      </Card>

      {expanded && annotatedUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-3 sm:p-6"
          onClick={() => setExpanded(false)}
        >
          <button
            className="absolute top-[max(0.75rem,env(safe-area-inset-top))] right-[max(0.75rem,env(safe-area-inset-right))] z-10 grid place-items-center h-11 w-11 rounded-full bg-white/15 text-white hover:bg-white/25"
            onClick={() => setExpanded(false)}
            aria-label="Close"
          >
            <X size={18} />
          </button>
          {isVideo ? (
            <video
              src={annotatedUrl}
              controls
              autoPlay
              playsInline
              className="modal-media-max max-w-full"
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <img
              src={annotatedUrl}
              alt="Annotated result full size"
              className="modal-media-max max-w-full object-contain"
              onClick={(e) => e.stopPropagation()}
            />
          )}
        </div>
      )}
    </>
  );
});

export default MediaPanel;
