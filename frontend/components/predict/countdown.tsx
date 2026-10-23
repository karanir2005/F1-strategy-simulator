"use client";

import { useEffect, useState } from "react";
import type { PredictionResponse } from "@/lib/api";

/** What we count down to. The API gives dates only (no session times), so the target is 00:00 UTC of that day. */
export function countdownTarget(p: PredictionResponse, now: number): { label: string; at: number } | null {
  const day = (iso: string | null) => (iso ? Date.parse(`${iso}T00:00:00Z`) : NaN);
  const quali = day(p.race.quali_date);
  const race = day(p.race.date);
  if (p.mode !== "grid" && Number.isFinite(quali) && quali > now) return { label: "Qualifying day", at: quali };
  if (Number.isFinite(race) && race > now) return { label: "Race day", at: race };
  return null;
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

  const total = Math.max(0, Math.floor((target.at - now) / 1000));
  const d = Math.floor(total / 86400);
  const h = Math.floor((total % 86400) / 3600);
  const mi = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const spoken = `${d} days, ${h} hours, ${mi} minutes`;

  return (
    <div className="min-h-[3.25rem]">
      <p className="label-mono mb-2">{target.label} starts in (UTC midnight)</p>
      <p className="sr-only">{`${target.label} in ${spoken}`}</p>
      <div aria-hidden className="flex flex-wrap gap-x-4 gap-y-2">
        <Unit n={d} unit="days" />
        <Unit n={h} unit="hrs" />
        <Unit n={mi} unit="min" />
        {!reduced && <Unit n={s} unit="sec" />}
      </div>
    </div>
  );
}
