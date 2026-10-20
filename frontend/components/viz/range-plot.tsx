"use client";

import * as m from "motion/react-m";
import type { ReactNode } from "react";

export interface RangeRow {
  id: string;
  label: ReactNode;
  sub?: ReactNode;
  p10: number;
  mean: number;
  p90: number;
  /** drawn in the "best" (purple) timing-screen colour */
  best?: boolean;
  right?: ReactNode;
}

/** Round tick positions covering [min, max] with roughly `count` ticks. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  const span = Math.max(max - min, 1e-9);
  const raw = span / Math.max(count - 1, 1);
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((k) => k * pow).find((s) => s >= raw) ?? raw;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

/**
 * p10-p90 whiskers with a mean dot, all on one shared axis. Plain divs: the whisker grows
 * outward from the mean with a scaleX transform. `rule` draws a vertical reference line.
 */
export function RangePlot({
  rows,
  domain,
  rule,
  ruleLabel,
  tickFormat,
  axisLabel,
}: {
  rows: RangeRow[];
  domain: [number, number];
  rule?: number;
  ruleLabel?: string;
  tickFormat: (v: number) => string;
  axisLabel?: string;
}) {
  const [lo, hi] = domain;
  const pos = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  const ticks = niceTicks(lo, hi, 5);

  const grid = (
    <>
      {ticks.map((t) => (
        <span key={t} aria-hidden className="absolute top-0 bottom-0 w-px bg-line" style={{ left: pos(t) }} />
      ))}
      {rule !== undefined && rule >= lo && rule <= hi && (
        <span
          aria-hidden
          className="absolute top-0 bottom-0 w-px bg-best/70"
          style={{ left: pos(rule) }}
          title={ruleLabel}
        />
      )}
    </>
  );

  return (
    <div className="w-full">
      <ul className="m-0 list-none p-0">
        {rows.map((r, i) => {
          const span = Math.max(r.p90 - r.p10, 1e-9);
          const color = r.best ? "bg-best" : "bg-fg";
          return (
            <li
              key={r.id}
              className="grid grid-cols-[minmax(5.5rem,9rem)_1fr] items-center gap-x-3 sm:grid-cols-[minmax(6rem,10rem)_1fr_auto]"
            >
              <div className="min-w-0 py-1.5 text-[13px] leading-4">
                <div className="truncate font-medium text-fg">{r.label}</div>
                {r.sub && <div className="truncate font-mono text-[11px] text-muted-fg tnum">{r.sub}</div>}
              </div>
              <div className="relative h-10">
                {grid}
                <span
                  className="absolute top-0 bottom-0 flex items-center"
                  style={{ left: pos(r.p10), width: `${(span / (hi - lo)) * 100}%` }}
                >
                  <m.span
                    className={`block h-0.5 w-full rounded-sm ${color} opacity-70`}
                    style={{ originX: (r.mean - r.p10) / span }}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1], delay: i * 0.04 }}
                  />
                </span>
                {[r.p10, r.p90].map((v, k) => (
                  <span
                    key={k}
                    aria-hidden
                    className={`absolute top-1/2 h-3 w-0.5 -translate-x-1/2 -translate-y-1/2 ${color} opacity-70`}
                    style={{ left: pos(v) }}
                  />
                ))}
                <span
                  aria-hidden
                  className={`absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-raised ${color}`}
                  style={{ left: pos(r.mean) }}
                />
              </div>
              {r.right !== undefined && (
                <div className="hidden min-w-[4.5rem] text-right font-mono text-[13px] text-fg tnum sm:block">
                  {r.right}
                </div>
              )}
            </li>
          );
        })}
      </ul>
      <div className="grid grid-cols-[minmax(5.5rem,9rem)_1fr] gap-x-3 sm:grid-cols-[minmax(6rem,10rem)_1fr_auto]">
        <div className="text-[11px] text-faint">{axisLabel}</div>
        <div className="relative h-5 border-t border-line-strong">
          {ticks.map((t) => (
            <span
              key={t}
              className="absolute top-1 -translate-x-1/2 font-mono text-[11px] text-muted-fg tnum"
              style={{ left: pos(t) }}
            >
              {tickFormat(t)}
            </span>
          ))}
        </div>
        <div className="hidden min-w-[4.5rem] sm:block" />
      </div>
    </div>
  );
}
