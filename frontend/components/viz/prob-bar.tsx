"use client";

import * as m from "motion/react-m";

/**
 * Horizontal probability bar. Neutral fill (team colour lives on the row's lead strip, not here).
 * Animates scaleX (transform only) from the left edge. `domain` is the value that fills the track.
 */
export function ProbBar({
  value,
  domain,
  revealed = true,
  delay = 0,
  className = "",
}: {
  value: number;
  domain: number;
  revealed?: boolean;
  delay?: number;
  className?: string;
}) {
  const frac = domain > 0 ? Math.min(1, Math.max(0, value / domain)) : 0;
  return (
    <div className={`h-1.5 w-full overflow-hidden rounded-sm bg-line ${className}`} aria-hidden>
      <m.span
        className="block h-full w-full origin-left rounded-sm bg-fg/90"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: revealed ? frac : 0 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1], delay }}
      />
    </div>
  );
}
