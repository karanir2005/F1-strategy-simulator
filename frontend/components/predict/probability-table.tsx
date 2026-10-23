"use client";

import { useInView } from "motion/react";
import * as m from "motion/react-m";
import { useId, useMemo, useRef, useState } from "react";
import type { DriverPrediction } from "@/lib/api";
import { fmtOneIn, fmtPct } from "@/lib/format";
import { teamColor, teamShort } from "@/lib/teams";
import { PctTicker } from "@/components/viz/pct-ticker";
import { ProbBar } from "@/components/viz/prob-bar";

type Metric = "p_win" | "p_podium" | "p_top10";

const METRICS: { key: Metric; label: string; long: string }[] = [
  { key: "p_win", label: "Win", long: "win the race" },
  { key: "p_podium", label: "Podium", long: "finish on the podium" },
  { key: "p_top10", label: "Top 10", long: "finish in the top ten" },
];

/** Bar scale: the metric's maximum rounded up to the next 5%, so the longest bar nearly fills the track. */
function barDomain(drivers: DriverPrediction[], key: Metric): number {
  const max = Math.max(...drivers.map((d) => d[key]), 0.05);
  return Math.min(1, Math.ceil(max * 20) / 20);
}

const COLS =
  "grid items-center gap-x-3 md:grid-cols-[2rem_minmax(0,1.5fr)_minmax(0,2fr)_4.25rem_4.25rem_4.25rem_4rem_3.5rem] grid-cols-[1.75rem_minmax(0,1fr)_3.75rem]";

function Detail({ d, open, id }: { d: DriverPrediction; open: boolean; id: string }) {
  return (
    <div
      id={id}
      className={`grid transition-[grid-template-rows] duration-200 ease-out motion-reduce:transition-none ${
        open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
      }`}
    >
      <div className={`overflow-hidden ${open ? "visible" : "invisible"}`}>
        <div className="grid gap-x-8 gap-y-3 px-3 pt-1 pb-4 sm:grid-cols-3 md:pl-[3.25rem]">
          {METRICS.map((mt) => (
            <div key={mt.key}>
              <div className="flex items-baseline justify-between text-[13px]">
                <span className="text-muted-fg">{mt.label}</span>
                <span className="font-mono text-fg tnum">{fmtPct(d[mt.key])}</span>
              </div>
              <ProbBar value={d[mt.key]} domain={1} revealed={open} className="mt-1.5" />
              <p className="mt-1 text-xs leading-4 text-muted-fg">
                {d[mt.key] < 0.005
                  ? "Almost never in a simulated race."
                  : `${fmtOneIn(d[mt.key]).replace(/^about/, "About")} simulated races end this way.`}
              </p>
            </div>
          ))}
          <div className="sm:col-span-3 sm:max-w-[calc(33.333%-1.4rem)]">
            <div className="flex items-baseline justify-between text-[13px]">
              <span className="text-muted-fg">Not finishing</span>
              <span className="font-mono text-fg tnum">{fmtPct(d.p_dnf)}</span>
            </div>
            <ProbBar value={d.p_dnf} domain={1} revealed={open} className="mt-1.5" />
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[13px] sm:col-span-3 sm:max-w-md">
            <dt className="text-muted-fg">Expected finish</dt>
            <dd className="font-mono text-fg tnum">P{d.exp_pos.toFixed(1)}</dd>
            <dt className="text-muted-fg">Starting position</dt>
            <dd className="font-mono text-fg tnum">
              {d.grid === null ? "not set (pre-quali)" : d.grid === 0 ? "pit lane" : `P${d.grid}`}
            </dd>
          </dl>
        </div>
      </div>
    </div>
  );
}

export function ProbabilityTable({ drivers }: { drivers: DriverPrediction[] }) {
  const [metric, setMetric] = useState<Metric>("p_win");
  const [openCode, setOpenCode] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -80px 0px" });
  const uid = useId();

  const sorted = useMemo(
    () => [...drivers].sort((a, b) => b[metric] - a[metric] || b.p_win - a.p_win),
    [drivers, metric],
  );
  const domain = barDomain(drivers, metric);
  const active = METRICS.find((x) => x.key === metric)!;

  return (
    <section aria-labelledby="table-h" ref={ref}>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 id="table-h" className="display text-3xl text-fg">
            Driver probabilities
          </h2>
          <p className="mt-1 text-[13px] text-muted-fg">
            Bars show the chance to {active.long}, scaled 0 to {Math.round(domain * 100)}%. Select a row for detail.
          </p>
        </div>
        <div role="radiogroup" aria-label="Ranking metric" className="inline-flex rounded-lg border border-line-strong p-0.5">
          {METRICS.map((mt) => (
            <button
              key={mt.key}
              type="button"
              role="radio"
              aria-checked={metric === mt.key}
              onClick={() => setMetric(mt.key)}
              className={`h-8 rounded-md px-3 text-[13px] font-medium transition-colors ${
                metric === mt.key ? "bg-signal text-signal-fg" : "text-muted-fg hover:text-fg"
              }`}
            >
              {mt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4 overflow-hidden rounded-panel border border-line">
        <div className={`${COLS} border-b border-line bg-sunken px-3 py-2`}>
          <span className="label-mono">#</span>
          <span className="label-mono">Driver</span>
          <span className="label-mono hidden md:block">{active.label} chance</span>
          <span className="label-mono text-right md:hidden">{active.label}</span>
          {METRICS.map((mt) => (
            <button
              key={mt.key}
              type="button"
              onClick={() => setMetric(mt.key)}
              className={`label-mono hidden text-right md:block ${metric === mt.key ? "!text-fg" : ""}`}
              aria-label={`Rank by ${mt.label}`}
            >
              {mt.label}
            </button>
          ))}
          <span className="label-mono hidden text-right md:block">Exp pos</span>
          <span className="label-mono hidden text-right md:block">DNF</span>
        </div>

        <ol className="m-0 list-none p-0">
          {sorted.map((d, i) => {
            const open = openCode === d.code;
            const detailId = `${uid}-${d.code}`;
            return (
              <m.li
                key={d.code}
                layout="position"
                transition={{ type: "spring", stiffness: 400, damping: 40 }}
                className="group border-b border-line last:border-b-0"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <button
                  type="button"
                  aria-expanded={open}
                  aria-controls={detailId}
                  onClick={() => setOpenCode(open ? null : d.code)}
                  className={`${COLS} relative w-full px-3 py-2.5 text-left transition-colors hover:bg-hover ${open ? "bg-hover" : ""}`}
                >
                  <span className="font-mono text-[13px] text-faint tnum">{i + 1}</span>
                  <span className="flex min-w-0 items-center gap-2.5">
                    <span
                      aria-hidden
                      className="h-8 w-[3px] shrink-0 rounded-sm transition-[width] duration-150 group-hover:w-1.5"
                      style={{ background: teamColor(d.team) }}
                    />
                    <span className="min-w-0">
                      <span className="block truncate text-[15px] leading-5 text-fg">
                        <span className="font-mono text-[13px] font-semibold">{d.code}</span>{" "}
                        <span className="hidden sm:inline">{d.name}</span>
                      </span>
                      <span className="block truncate text-xs leading-4 text-muted-fg">{teamShort(d.team)}</span>
                    </span>
                  </span>
                  <span className="hidden pr-2 md:block">
                    <ProbBar value={d[metric]} domain={domain} revealed={inView} delay={Math.min(i, 11) * 0.025} />
                  </span>
                  {/* active metric value; on md+ it sits in its own column below */}
                  <span className="text-right font-mono text-[15px] font-semibold text-fg md:hidden">
                    <PctTicker value={d[metric]} duration={0.6} />
                  </span>
                  {METRICS.map((mt) => (
                    <span
                      key={mt.key}
                      className={`hidden text-right font-mono text-[13px] tnum md:block ${
                        metric === mt.key ? "font-semibold text-fg" : "text-muted-fg"
                      }`}
                    >
                      {metric === mt.key ? <PctTicker value={d[mt.key]} duration={0.6} /> : fmtPct(d[mt.key])}
                    </span>
                  ))}
                  <span className="hidden text-right font-mono text-[13px] text-muted-fg tnum md:block">
                    {d.exp_pos.toFixed(1)}
                  </span>
                  <span className="hidden text-right font-mono text-[13px] text-muted-fg tnum md:block">
                    {fmtPct(d.p_dnf)}
                  </span>
                  {/* bar under the name on small screens */}
                  <span className="col-span-3 mt-2 block pl-[calc(1.75rem+0.75rem+13px)] md:hidden">
                    <ProbBar value={d[metric]} domain={domain} revealed={inView} delay={Math.min(i, 11) * 0.025} />
                  </span>
                </button>
                <Detail d={d} open={open} id={detailId} />
              </m.li>
            );
          })}
        </ol>
      </div>
      <p className="mt-3 text-xs leading-4 text-muted-fg">
        Exp pos is the expected finishing position from the simulation. DNF is the chance of not finishing.
      </p>
    </section>
  );
}
