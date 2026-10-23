import { AlertTriangle } from "lucide-react";
import { Button } from "./Primitives";

export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Keep Going",
  tone = "danger",
  onConfirm,
  onClose,
}) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="glass-panel rounded-2xl shadow-panel max-w-sm w-full p-5 sm:p-6 my-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3 mb-4">
          <div className="grid place-items-center h-10 w-10 shrink-0 rounded-xl bg-risk-critical/15 text-risk-critical">
            <AlertTriangle size={18} />
          </div>
          <div>
            <h3 className="text-base font-display font-semibold text-white">{title}</h3>
            {description && <p className="text-sm text-slate-400 mt-1">{description}</p>}
          </div>
        </div>
        <div className="flex flex-col-reverse xs:flex-row gap-3 xs:justify-end">
          <Button variant="ghost" onClick={onClose}>
            {cancelLabel}
          </Button>
          <Button variant={tone} onClick={onConfirm}>
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
