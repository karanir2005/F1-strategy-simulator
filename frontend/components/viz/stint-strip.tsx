"use client";

import * as m from "motion/react-m";
import { COMPOUND_VAR } from "@/lib/format";

export interface StripStint {
  compound: string;
  laps: number;
}

/** Stint widths are proportional to laps; fill is the tyre colour. Segments grow left to right. */
export function StintStrip({
  stints,
  size = "md",
  playKey,
  labels,
}: {
  stints: StripStint[];
  size?: "sm" | "md" | "lg";
  /** change to replay the reveal (e.g. a plan hash) */
  playKey?: string;
  /** optional compound -> tyre label (e.g. "C3") */
  labels?: Record<string, string | null | undefined>;
}) {
  const h = size === "lg" ? "h-11 text-sm" : size === "md" ? "h-8 text-[13px]" : "h-6 text-xs";
  const total = stints.reduce((s, x) => s + x.laps, 0);
  return (
    <div
      key={playKey}
      className={`flex w-full gap-px overflow-hidden rounded-sm ${h}`}
      role="img"
      aria-label={stints.map((s) => `${s.compound.toLowerCase()} ${s.laps} laps`).join(", then ")}
    >
      {stints.map((s, i) => {
        const tyre = labels?.[s.compound];
        const wide = total > 0 && s.laps / total > 0.14;
        return (
          <m.div
            key={i}
            style={{ flexGrow: s.laps, flexBasis: 0, background: COMPOUND_VAR[s.compound] ?? "#888", originX: 0 }}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1], delay: i * 0.08 }}
            className="flex min-w-0 items-center justify-center font-mono font-semibold text-[#14161a] tnum"
            title={`${s.compound}${tyre ? ` (${tyre})` : ""}, ${s.laps} laps`}
          >
            <span className="truncate px-1">
              {wide ? `${s.compound[0]}${tyre ? ` ${tyre}` : ""} ` : ""}
              {s.laps}
            </span>
          </m.div>
        );
      })}
    </div>
  );
}
