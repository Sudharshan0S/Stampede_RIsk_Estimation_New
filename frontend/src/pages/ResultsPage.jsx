import { useEffect, useRef, useState } from "react";
import { LayoutDashboard, ImagePlay, BarChart3, Table2, AlertTriangle, Siren } from "lucide-react";
import { useApp } from "../context/AppContext";
import { Card, SectionHeading, Button } from "../components/common/Primitives";
import DownloadsPanel from "../components/results/DownloadsPanel";
import MediaPanel from "../components/results/MediaPanel";
import OverviewTab from "../components/results/OverviewTab";
import ImageCharts from "../components/results/ImageCharts";
import VideoCharts from "../components/results/VideoCharts";
import FrameDataTable from "../components/results/FrameDataTable";
import EventsPanel from "../components/results/EventsPanel";

const TABS = [
  { key: "overview", label: "Overview", icon: LayoutDashboard },
  { key: "media", label: "Annotated Media", icon: ImagePlay },
  { key: "events", label: "Events", icon: Siren, videoOnly: true },
  { key: "charts", label: "Charts", icon: BarChart3 },
  { key: "data", label: "Window Data", icon: Table2, videoOnly: true },
];

export default function ResultsPage() {
  const { result, status, error, reset } = useApp();
  const [subTab, setSubTab] = useState("overview");
  const [selectedTime, setSelectedTime] = useState(null); // seconds into the video
  const videoRef = useRef(null);

  const isVideo = result?.media_type === "video";

  // Seeks the annotated video whenever a moment is picked from a chart, the window table or an event.
  // The annotated video is written at the analysis rate and spans the same timeline (time_s), so currentTime = time_s.
  useEffect(() => {
    if (!isVideo || selectedTime == null) return;
    const raf = requestAnimationFrame(() => {
      if (videoRef.current) {
        videoRef.current.currentTime = selectedTime;
        videoRef.current.scrollIntoView({ block: "center", behavior: "smooth" }); // on phones the tab switch can leave the player off-screen
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [selectedTime, subTab, isVideo]);

  const handleTimeClick = (t) => {
    setSelectedTime(t);
    setSubTab("media");
  };

  if (status === "error" || !result) {
    return (
      <div className="space-y-6">
        <SectionHeading eyebrow="Step 03" title="Results" description="Something went wrong during analysis." />
        <Card className="p-6 sm:p-8 flex flex-col items-center text-center gap-3">
          <AlertTriangle size={28} className="text-risk-critical" />
          <p className="text-sm text-slate-300 max-w-md whitespace-pre-line break-words">{error || "No result available yet."}</p>
          <Button variant="ghost" onClick={reset}>
            Start Over
          </Button>
        </Card>
      </div>
    );
  }

  const visibleTabs = TABS.filter((t) => !t.videoOnly || isVideo);

  return (
    <div className="space-y-6">
      <SectionHeading
        eyebrow="Step 03"
        title="Analysis Results"
        description={
          isVideo
            ? `Crowd-risk assessment of the uploaded video: ${result.windows} temporal windows, ${result.n_events} abnormal event(s).`
            : `Crowd-count analysis of the uploaded image (${result.image_width}×${result.image_height}).`
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 space-y-6 min-w-0">
          <nav className="flex items-center gap-1 rounded-xl border border-white/[0.06] bg-base-800/60 p-1 overflow-x-auto overscroll-x-contain">
            {visibleTabs.map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={(e) => {
                  setSubTab(key);
                  e.currentTarget.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" }); // scrollable tab strip on narrow screens
                }}
                className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-lg px-3.5 py-2 coarse:min-h-[44px] text-sm font-medium transition-colors ${
                  subTab === key ? "bg-accent text-white shadow-glow" : "text-slate-300 hover:bg-white/[0.06]"
                }`}
              >
                <Icon size={14} /> {label}
                {key === "events" && result.n_events > 0 && (
                  <span className="ml-1 rounded-full bg-risk-critical/20 text-risk-critical px-1.5 text-[10px] font-semibold">{result.n_events}</span>
                )}
              </button>
            ))}
          </nav>

          {subTab === "overview" && (
            <div className="space-y-6">
              <OverviewTab result={result} />
              <MediaPanel result={result} ref={videoRef} />
            </div>
          )}
          {subTab === "media" && <MediaPanel result={result} ref={videoRef} />}
          {subTab === "events" && isVideo && <EventsPanel result={result} onSeek={handleTimeClick} />}
          {subTab === "charts" &&
            (isVideo ? (
              <VideoCharts result={result} onTimeClick={handleTimeClick} selectedTime={selectedTime} />
            ) : (
              <ImageCharts result={result} />
            ))}
          {subTab === "data" && isVideo && <FrameDataTable result={result} selectedTime={selectedTime} onTimeClick={handleTimeClick} />}
        </div>

        <div className="lg:col-span-4 min-w-0">
          <DownloadsPanel result={result} />
        </div>
      </div>
    </div>
  );
}
