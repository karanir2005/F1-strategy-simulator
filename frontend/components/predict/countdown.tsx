"use client";

import { useEffect, useState } from "react";
import type { PredictionResponse } from "@/lib/api";
import { daysUntil } from "@/lib/format";

export interface CountdownTarget {
  label: string;
  /** exact start (ms) when the API gives a session time; null = date only, whole days */
  at: number | null;
  /** whole local calendar days away (date-only mode); same logic as the status line */
  days: number;
  iso: string | null;
}

/** Qualifying first (unless the grid is already set), then the race. Uses real start times when the API has them. */
export function countdownTarget(p: PredictionResponse, now: number): CountdownTarget | null {
  const ms = (iso: string | null | undefined) => (iso ? Date.parse(iso) : NaN);
  const qStart = ms(p.race.quali_start_utc);
  const rStart = ms(p.race.race_start_utc);
  if (p.mode !== "grid" && Number.isFinite(qStart) && qStart > now)
    return { label: "Qualifying", at: qStart, days: 0, iso: p.race.quali_start_utc ?? null };
  if (Number.isFinite(rStart) && rStart > now)
    return { label: "Race", at: rStart, days: 0, iso: p.race.race_start_utc ?? null };
  if (Number.isFinite(rStart)) return null; // race start is known and has passed

  // date-only fallback: same calendar-day arithmetic as weekendStatus
  const d = new Date(now);
  if (p.mode !== "grid" && p.race.quali_date) {
    const dq = daysUntil(p.race.quali_date, d);
    if (dq > 0) return { label: "Qualifying", at: null, days: dq, iso: null };
  }
  const dr = daysUntil(p.race.date, d);
  if (Number.isFinite(dr) && dr > 0) return { label: "Lights out", at: null, days: dr, iso: null };
  return null;
}

function localSession(iso: string | null): string | null {
  if (!iso) return null;
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return null;
  return t.toLocaleString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

function Unit({ n, unit }: { n: number; unit: string }) {
  return (
    <div className="flex min-w-[3.25rem] flex-col">
      <span className="display text-3xl leading-none text-fg tnum">{String(n).padStart(2, "0")}</span>
      <span className="label-mono mt-1">{unit}</span>
    </div>
  );
}

/** Live countdown. Renders nothing on the server and first client pass (no hydration mismatch). */
export function Countdown({ data }: { data: PredictionResponse }) {
  const [now, setNow] = useState<number | null>(null);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    setNow(Date.now());
    // reduced motion: no per-second ticking digits, refresh once a minute
    const id = window.setInterval(() => setNow(Date.now()), reduced ? 60_000 : 1000);
    return () => window.clearInterval(id);
  }, [reduced]);

  if (now === null) return <div className="min-h-[3.25rem]" aria-hidden />;
  const target = countdownTarget(data, now);
  if (!target) return null;

  // the other session's local time, shown beside the countdown when known
  const sessions = [
    ["Qualifying", localSession(data.race.quali_start_utc ?? null)],
    ["Race", localSession(data.race.race_start_utc ?? null)],
  ].filter(([, t]) => t) as [string, string][];
  const times = (
    <div className="mt-2 space-y-0.5 font-mono text-[11px] text-faint">
      {sessions.map(([name, t]) => (
        <p key={name}>
          {name}: {t}
        </p>
      ))}
    </div>
  );

  if (target.at === null) {
    return (
      <div className="min-h-[3.25rem]">
        <p className="label-mono mb-2">{target.label} in (exact start time not available)</p>
        <div className="flex gap-4">
          <Unit n={target.days} unit={target.days === 1 ? "day" : "days"} />
        </div>
      </div>
    );
  }

  const total = Math.max(0, Math.floor((target.at - now) / 1000));
  const d = Math.floor(total / 86400);
  const h = Math.floor((total % 86400) / 3600);
  const mi = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const spoken = `${d} days, ${h} hours, ${mi} minutes`;

  return (
    <div className="min-h-[3.25rem]">
      <p className="label-mono mb-2">{target.label} starts in</p>
      <p className="sr-only">{`${target.label} in ${spoken}`}</p>
      <div aria-hidden className="flex flex-wrap gap-x-4 gap-y-2">
        <Unit n={d} unit="days" />
        <Unit n={h} unit="hrs" />
        <Unit n={mi} unit="min" />
        {!reduced && <Unit n={s} unit="sec" />}
      </div>
      {times}
    </div>
  );
}
