"use client";

import { animate, useMotionValue, useReducedMotion, useTransform } from "motion/react";
import * as m from "motion/react-m";
import { useEffect } from "react";
import { fmtPct } from "@/lib/format";

/** Mid-count frames: plain rounding (no "<1%" flicker). The settled value uses the real formatter. */
function live(v: number): string {
  const pct = v * 100;
  return pct >= 10 ? `${Math.round(pct)}%` : `${pct.toFixed(1)}%`;
}

/**
 * Counts from the previous value to `value` (a 0-1 fraction). The text is a motion value, so
 * the count does not re-render React each frame. Skips the count for reduced-motion users.
 */
export function PctTicker({
  value,
  duration = 0.9,
  className,
}: {
  value: number;
  duration?: number;
  className?: string;
}) {
  const reduced = useReducedMotion();
  const mv = useMotionValue(0);
  const text = useTransform(mv, (v) => (Math.abs(v - value) < 1e-6 ? fmtPct(value) : live(v)));

  useEffect(() => {
    if (reduced) {
      mv.set(value);
      return;
    }
    const controls = animate(mv, value, { duration, ease: [0.22, 1, 0.36, 1] });
    return () => controls.stop();
  }, [value, reduced, duration, mv]);

  return <m.span className={`tnum ${className ?? ""}`}>{text}</m.span>;
}
