"use client";

import { useEffect, useState } from "react";
import type { PredictionResponse } from "@/lib/api";
import { daysUntil, fmtDay, fmtUpdated } from "@/lib/format";
import { Badge } from "@/components/ui/badge";
import { Countdown } from "@/components/predict/countdown";

function plural(n: number, word: string) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** Plain-language state of the weekend, from the race / quali dates and the API mode. */
export function weekendStatus(p: PredictionResponse, now: Date): string {
  const toRace = daysUntil(p.race.date, now);
  const toQuali = p.race.quali_date ? daysUntil(p.race.quali_date, now) : null;
  const race = toRace > 0 ? `lights out in ${plural(toRace, "day")}` : toRace === 0 ? "race day" : "race finished";
  if (p.mode === "grid") return `Qualifying done, grid set. ${race.charAt(0).toUpperCase()}${race.slice(1)}.`;
  if (toQuali !== null && toQuali > 0) return `Qualifying in ${plural(toQuali, "day")}. ${race.charAt(0).toUpperCase()}${race.slice(1)}.`;
  if (toQuali === 0) return "Qualifying today. The prediction will switch to grid mode once the result is published.";
  return `Qualifying has run but the grid is not published yet. ${race.charAt(0).toUpperCase()}${race.slice(1)}.`;
}

/** Note under the mode badge; null when there is nothing to say (pre-quali, or an old saved result). */
export function gridSourceNote(src: PredictionResponse["grid_source"]): string | null {
  if (src === "override_file") return "Grid adjusted for penalties";
  if (src === "qualifying_classification") return "Grid is the qualifying classification; penalties not applied";
  return null;
}

export function ModeBadge({ mode }: { mode: string }) {
  const grid = mode === "grid";
  return (
    <div
      className="inline-flex items-center overflow-hidden rounded-sm border border-line-strong font-mono text-[11px] tracking-wide uppercase"
      role="group"
      aria-label={`Prediction mode: ${grid ? "grid" : "pre-qualifying"}`}
    >
      <span className={`px-2 py-1 ${!grid ? "bg-signal text-signal-fg" : "text-faint"}`}>Pre-quali</span>
      <span className={`px-2 py-1 ${grid ? "bg-signal text-signal-fg" : "text-faint"}`}>Grid</span>
    </div>
  );
}

export function RaceHeader({ data }: { data: PredictionResponse }) {
  const [status, setStatus] = useState<string | null>(null);
  useEffect(() => setStatus(weekendStatus(data, new Date())), [data]);
  const grid = data.mode === "grid";

  return (
    <header className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div className="min-w-0">
        <p className="label-mono">
          Round {data.race.round} · {fmtDay(data.race.date)}
        </p>
        <h1 className="display mt-1 text-[34px] leading-[1.02] text-fg sm:text-5xl">{data.race.name}</h1>
        <p className="mt-1 text-[15px] text-muted-fg">{data.race.circuit || "Circuit to be confirmed"}</p>
        <p className="mt-3 min-h-6 text-[15px] text-fg">{status}</p>
        <div className="mt-4">
          <Countdown data={data} />
        </div>
      </div>
      <div className="flex flex-col gap-2 md:items-end">
        <div className="flex items-center gap-2">
          <ModeBadge mode={data.mode} />
          {data.stale && <Badge variant="warn">Saved</Badge>}
        </div>
        <p className="max-w-xs text-[13px] leading-5 text-muted-fg md:text-right">
          {grid
            ? gridSourceNote(data.grid_source) ?? "Uses the grid published after qualifying."
            : "Before qualifying: based on form and results history only. It will sharpen once the grid is set."}
        </p>
        <p className="font-mono text-[11px] text-faint tnum">Updated {fmtUpdated(data.generated_at)}</p>
      </div>
    </header>
  );
}
