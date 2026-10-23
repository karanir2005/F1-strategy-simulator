"use client";

import * as m from "motion/react-m";
import { useMemo } from "react";
import type { DriverPrediction, PredictionResponse } from "@/lib/api";
import { fmtOneIn, fmtPct } from "@/lib/format";
import { teamColor, teamShort } from "@/lib/teams";
import { PctTicker } from "@/components/viz/pct-ticker";
import { ProbBar } from "@/components/viz/prob-bar";

/** Largest-remainder rounding of win probabilities into `n` equal dots (each dot = 1 of n simulated races). */
export function quantileDots(drivers: DriverPrediction[], n = 100): DriverPrediction[] {
  const total = drivers.reduce((s, d) => s + d.p_win, 0) || 1;
  const raw = drivers.map((d) => ({ d, x: (d.p_win / total) * n }));
  const base = raw.map((r) => ({ ...r, c: Math.floor(r.x) }));
  let left = n - base.reduce((s, r) => s + r.c, 0);
  [...base]
    .sort((a, b) => b.x - b.c - (a.x - a.c))
    .forEach((r) => {
      if (left > 0) {
        r.c += 1;
        left -= 1;
      }
    });
  const out: DriverPrediction[] = [];
  for (const r of base) for (let i = 0; i < r.c; i++) out.push(r.d);
  return out;
}

function Headline({ data, top, runnerUp }: { data: PredictionResponse; top: DriverPrediction; runnerUp: DriverPrediction | null }) {
  const dots = useMemo(() => quantileDots(data.drivers, 100), [data.drivers]);
  const counts = useMemo(() => {
    const c = new Map<string, number>();
    dots.forEach((d) => c.set(d.code, (c.get(d.code) ?? 0) + 1));
    return c;
  }, [dots]);
  // teammates share a colour, so the second driver of each team is drawn as a hollow ring
  const secondCodes = useMemo(() => {
    const seen = new Map<string, string[]>();
    [...data.drivers]
      .sort((a, b) => a.code.localeCompare(b.code))
      .forEach((d) => seen.set(d.team, [...(seen.get(d.team) ?? []), d.code]));
    return new Set(Array.from(seen.values()).flatMap((codes) => codes.slice(1)));
  }, [data.drivers]);
  const swatch = (d: DriverPrediction): React.CSSProperties =>
    secondCodes.has(d.code)
      ? { border: `2px solid ${teamColor(d.team)}`, background: "transparent" }
      : { background: teamColor(d.team) };
  const lead = runnerUp ? (top.p_win - runnerUp.p_win) * 100 : null;

  return (
    <section aria-labelledby="hl-winner" className="min-w-0">
      <p id="hl-winner" className="label-mono">
        Most likely winner
      </p>
      <div className="mt-2 flex items-stretch gap-3">
        <span aria-hidden className="w-[3px] shrink-0 rounded-sm" style={{ background: teamColor(top.team) }} />
        <div className="min-w-0">
          <p className="truncate text-xl font-semibold leading-7 text-fg">
            <span className="font-mono text-sm font-medium text-muted-fg">{top.code}</span> {top.name}
          </p>
          <p className="text-[13px] leading-5 text-muted-fg">{teamShort(top.team)}</p>
        </div>
      </div>
      <p className="display mt-3 text-[88px] leading-[0.9] text-fg sm:text-[112px]">
        <PctTicker value={top.p_win} />
      </p>
      <div className="mt-4 max-w-md">
        <ProbBar value={top.p_win} domain={1} />
        <p className="mt-3 text-[15px] leading-6 text-fg">
          Wins {fmtOneIn(top.p_win)} simulated races. The rest of the field wins the other{" "}
          {fmtPct(1 - top.p_win)}.
        </p>
        {lead !== null && runnerUp && (
          <p className="mt-1 text-[13px] leading-5 text-muted-fg">
            {lead.toFixed(1)} points ahead of {runnerUp.code} ({fmtPct(runnerUp.p_win)}). A favourite at this level
            still loses most of the time.
          </p>
        )}
      </div>

      <div className="mt-6">
        <p className="label-mono">100 simulated races</p>
        <div
          role="img"
          aria-label={`Of 100 simulated races: ${Array.from(counts.entries()).map(([c, n]) => `${c} wins ${n}`).join(", ")}`}
          className="mt-2 grid max-w-[360px] grid-cols-[repeat(20,minmax(0,1fr))] gap-[5px]"
        >
          {dots.map((d, i) => (
            <m.span
              key={i}
              title={`${d.name}: wins in the simulation`}
              className="aspect-square rounded-full"
              style={swatch(d)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2, delay: Math.min(i * 0.005, 0.5) }}
            />
          ))}
        </div>
        <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-muted-fg">
          {Array.from(counts.entries())
            .filter(([, n]) => n >= 3)
            .map(([code, n]) => {
              const d = data.drivers.find((x) => x.code === code)!;
              return (
                <li key={code} className="flex items-center gap-1.5">
                  <span aria-hidden className="box-border size-2.5 rounded-full" style={swatch(d)} />
                  <span className="font-mono text-fg">{code}</span>
                  <span className="tnum">{n}</span>
                </li>
              );
            })}
        </ul>
        <p className="mt-2 max-w-md text-[13px] leading-5 text-muted-fg">
          Each dot is 1 of 100 simulated races, coloured by the winner&apos;s team (solid for one driver, hollow ring for the teammate). Drivers below half a percent get no
          dot.
        </p>
      </div>
    </section>
  );
}

const PODIUM_H = 200;

function Podium({ data }: { data: PredictionResponse }) {
  const picks = data.headline.podium
    .map((c) => data.drivers.find((d) => d.code === c))
    .filter((d): d is DriverPrediction => !!d);
  // highest podium chance in the centre, then left, then right
  const order = [picks[1], picks[0], picks[2]].filter((d): d is DriverPrediction => !!d);
  const delays: Record<string, number> = {};
  if (picks[1]) delays[picks[1].code] = 0.1;
  if (picks[0]) delays[picks[0].code] = 0.2;
  if (picks[2]) delays[picks[2].code] = 0;

  return (
    <section aria-labelledby="hl-podium" className="min-w-0">
      <p id="hl-podium" className="label-mono">
        Podium: three highest podium chances
      </p>
      <ol className="mt-3 flex items-end gap-2 sm:gap-3" style={{ height: Math.max(...picks.map((d) => d.p_podium), 0.2) * PODIUM_H + 96 }}>
        {order.map((d) => (
          <li key={d.code} className="flex min-w-0 flex-1 flex-col justify-end">
            <p className="font-mono text-sm font-semibold text-fg">{d.code}</p>
            <p className="display text-3xl leading-none text-fg sm:text-4xl">
              <PctTicker value={d.p_podium} />
            </p>
            <p className="mb-2 truncate text-[13px] leading-5 text-muted-fg">{d.name.split(" ").slice(-1)[0]}</p>
            <div className="relative w-full" style={{ height: Math.max(24, d.p_podium * PODIUM_H) }}>
              <m.div
                className="absolute inset-0 origin-bottom rounded-t-sm border border-b-0 border-line-strong bg-hover"
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1], delay: delays[d.code] ?? 0 }}
              >
                <span className="block h-[3px] w-full rounded-t-sm" style={{ background: teamColor(d.team) }} />
              </m.div>
            </div>
          </li>
        ))}
      </ol>
      <div className="h-px w-full bg-line-strong" aria-hidden />
      <p className="mt-3 max-w-md text-[13px] leading-5 text-muted-fg">
        Block height is each driver&apos;s own chance of a top-three finish. It is not the chance that exactly these
        three share the podium, which is much lower.
      </p>
    </section>
  );
}

export function Hero({ data }: { data: PredictionResponse }) {
  const top = data.drivers.find((d) => d.code === data.headline.winner) ?? data.drivers[0];
  const runnerUp = data.drivers.filter((d) => d.code !== top.code).sort((a, b) => b.p_win - a.p_win)[0] ?? null;
  return (
    <div className="grid gap-10 border-t border-line pt-8 lg:grid-cols-12 lg:gap-8">
      <div className="lg:col-span-7">
        <Headline data={data} top={top} runnerUp={runnerUp} />
      </div>
      <div className="lg:col-span-5">
        <Podium data={data} />
      </div>
    </div>
  );
}
