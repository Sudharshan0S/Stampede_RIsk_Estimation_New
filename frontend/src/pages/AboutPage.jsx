import { ScanEye, Waves, Gauge, ShieldAlert, Cpu, Server } from "lucide-react";
import { Card, SectionHeading } from "../components/common/Primitives";

const PIPELINE = [
  {
    icon: ScanEye,
    title: "Crowd Counting",
    body: "CrowdPointNet localises every head as a point (shared ResNet-18 trunk with a stride-4 point head and a density branch). Large frames are tiled, doubtful regions are re-analysed at higher resolution, and the count is the number of accepted head points.",
  },
  {
    icon: Waves,
    title: "Motion Estimation",
    body: "Dense optical flow between analysis steps. The estimator is the one that won validation in notebook 02 (DIS at 640 px width for the shipped model); camera ego-motion is compensated and the crowd region comes from the density map.",
  },
  {
    icon: Gauge,
    title: "Temporal Features",
    body: "Per step: speed, acceleration, direction, coherence, divergence, convergence, counter-flow and density statistics. Per window of 20 steps (hop 5): trends, flow-advected trajectories and change scores against the crowd's own recent baseline.",
  },
  {
    icon: ShieldAlert,
    title: "Abnormality & Risk",
    body: "A noisy-OR abnormality detector (10 evidence signals) and a LightGBM risk model give LOW / MEDIUM / HIGH / CRITICAL. Scores are smoothed, an alarm needs persistence, and consecutive alarm windows become one explained event.",
  },
];

export default function AboutPage() {
  return (
    <div className="space-y-8">
      <SectionHeading
        eyebrow="About"
        title="How CrowdVision AI works"
        description="A short walkthrough of the analysis pipeline running behind the Input, Analysis and Results tabs."
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {PIPELINE.map((step, i) => (
          <Card key={step.title} className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="grid place-items-center h-10 w-10 rounded-xl bg-accent/10 text-accent-soft">
                <step.icon size={18} />
              </div>
              <div>
                <div className="text-xs text-slate-500">Stage 0{i + 1}</div>
                <h3 className="text-sm font-semibold text-white">{step.title}</h3>
              </div>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed">{step.body}</p>
          </Card>
        ))}
      </div>

      <Card className="p-6">
        <div className="flex items-center gap-2 mb-3">
          <Cpu size={16} className="text-accent-soft" />
          <h3 className="text-sm font-semibold text-white">Output artifacts</h3>
        </div>
        <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-slate-400">
          <li>• Annotated image (head points, density map) or video (flow arrows, risk HUD)</li>
          <li>• Abnormal events with tags, evidence and explanations</li>
          <li>• Full JSON export of every computed value</li>
          <li>• CSV exports (per window for video, single row for images)</li>
          <li>• TXT / PDF reports, timeline and diagnostics charts</li>
        </ul>
      </Card>

      <p className="text-xs text-slate-600 flex items-center gap-1.5">
        <Server size={13} /> CrowdVision AI — Flask backend + React frontend, all data is generated locally by your own trained models. Risk labels are derived (weak) from TUB CrowdFlow ground-truth motion + synthetic injections - they are not official stampede labels.
      </p>
    </div>
  );
}
