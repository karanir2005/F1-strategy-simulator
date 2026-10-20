import { AlertTriangle } from "lucide-react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";

/** Inline error block with an optional retry. role=alert so it is announced. */
export function ErrorBlock({
  title,
  message,
  hint,
  onRetry,
}: {
  title: string;
  message: string;
  hint?: ReactNode;
  onRetry?: () => void;
}) {
  return (
    <div role="alert" className="flex gap-3 rounded-panel border border-bad/40 bg-bad/10 p-4 text-sm">
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-bad" aria-hidden />
      <div className="min-w-0 space-y-1">
        <p className="font-medium text-fg">{title}</p>
        <p className="break-words text-muted-fg">{message}</p>
        {hint && <p className="text-[13px] text-muted-fg">{hint}</p>}
        {onRetry && (
          <Button variant="outline" size="sm" className="mt-2" onClick={onRetry}>
            Try again
          </Button>
        )}
      </div>
    </div>
  );
}

export function EmptyBlock({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-panel border border-dashed border-line-strong p-6 text-sm">
      <p className="font-medium text-fg">{title}</p>
      {children && <div className="mt-1 text-muted-fg">{children}</div>}
    </div>
  );
}

export function Sk({ className = "" }: { className?: string }) {
  return <div aria-hidden className={`skeleton ${className}`} />;
}
