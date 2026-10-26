"use client";

import { useInView } from "motion/react";
import * as m from "motion/react-m";
import { useId, useMemo, useRef, useState, type ReactNode } from "react";
import type { DriverPrediction, QualifyingBlock, QualifyingDriver } from "@/lib/api";
import { fmtPct } from "@/lib/format";
import { teamColor, teamShort } from "@/lib/teams";
import { PctTicker } from "@/components/viz/pct-ticker";
import { ProbBar } from "@/components/viz/prob-bar";

type Session = "race" | "quali";
type RaceMetric = "p_win" | "p_podium" | "p_top10";
type QualiMetric = "p_pole" | "p_front_row" | "p_q3";
type Metric = RaceMetric | QualiMetric;

/** One table row: race prediction plus, when available, the driver's qualifying prediction. */
type Row = DriverPrediction & { q: QualifyingDriver | null };

const METRICS: Record<Session, { key: Metric; label: string; long: string }[]> = {
  race: [
    { key: "p_win", label: "Win", long: "win the race" },
    { key: "p_podium", label: "Podium", long: "finish on the podium" },
    { key: "p_top10", label: "Top 10", long: "finish in the top ten" },
  ],
  quali: [
    { key: "p_pole", label: "Pole", long: "take pole position" },
    { key: "p_front_row", label: "Front row", long: "start from the front row" },
    { key: "p_q3", label: "Q3", long: "reach Q3 (top ten)" },
  ],
};

function val(r: Row, key: Metric): number {
  if (key === "p_win" || key === "p_podium" || key === "p_top10") return r[key];
  return r.q ? r.q[key] : 0;
}

/** Bar scale: the metric's maximum rounded up to the next 5%, so the longest bar nearly fills the track. */
function barDomain(rows: Row[], key: Metric): number {
  const max = Math.max(...rows.map((r) => val(r, key)), 0.05);
  return Math.min(1, Math.ceil(max * 20) / 20);
}

/** Places the model expects the driver to gain (+) or lose (-) versus the starting grid; grid mode only. */
function DeltaVsGrid({ r }: { r: Row }) {
  if (r.grid == null || r.grid === 0) return null;
  const d = Math.round(r.grid - r.exp_pos);
  if (d === 0) return <span className="ml-1.5 text-faint">=</span>;
  return (
    <span
      className={`ml-1.5 ${d > 0 ? "text-good" : "text-muted-fg"}`}
      title={`Expected finish vs starting position P${r.grid}`}
      aria-label={`${d > 0 ? "gains" : "loses"} ${Math.abs(d)} places versus the grid`}
    >
      {d > 0 ? "+" : "-"}
      {Math.abs(d)}
    </span>
  );
}

function gridLabel(g: number | null): string {
  return g == null ? "not set" : g === 0 ? "pit lane" : `P${g}`;
}

const COLS_BASE = "grid items-center gap-x-3 grid-cols-[1.75rem_minmax(0,1fr)_3.75rem]";
const COLS_QUALI = `${COLS_BASE} md:grid-cols-[2rem_minmax(0,1.5fr)_minmax(0,2fr)_4rem_5.5rem_4rem_5.5rem_4.5rem]`;
// race view has a third trailing column (starting grid)
const COLS_RACE = `${COLS_BASE} md:grid-cols-[2rem_minmax(0,1.5fr)_minmax(0,2fr)_4rem_4.5rem_4.5rem_3.5rem_6rem_3.5rem]`;

function Detail({ r, session, open, id }: { r: Row; session: Session; open: boolean; id: string }) {
  const metrics = METRICS[session];
  const extra =
    session === "race"
      ? { label: "Not finishing", value: r.p_dnf }
      : r.q
        ? { label: "Out in Q1", value: r.q.p_q1_exit }
        : null;
  return (
    <div
      id={id}
      className={`grid transition-[grid-template-rows] duration-200 ease-out motion-reduce:transition-none ${
        open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
      }`}
    >
      <div className={`overflow-hidden ${open ? "visible" : "invisible"}`}>
        <div className="grid gap-x-8 gap-y-3 px-3 pt-1 pb-4 sm:grid-cols-3 md:pl-[3.25rem]">
          {metrics.map((mt) => {
            const v = val(r, mt.key);
            return (
              <div key={mt.key}>
                <div className="flex items-baseline justify-between text-[13px]">
                  <span className="text-muted-fg">{mt.label}</span>
                  <span className="font-mono text-fg tnum">{fmtPct(v)}</span>
                </div>
                <ProbBar value={v} domain={1} revealed={open} className="mt-1.5" />
              </div>
            );
          })}
          {extra && (
            <div className="sm:col-span-3 sm:max-w-[calc(33.333%-1.4rem)]">
              <div className="flex items-baseline justify-between text-[13px]">
                <span className="text-muted-fg">{extra.label}</span>
                <span className="font-mono text-fg tnum">{fmtPct(extra.value)}</span>
              </div>
              <ProbBar value={extra.value} domain={1} revealed={open} className="mt-1.5" />
            </div>
          )}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[13px] sm:col-span-3 sm:max-w-md">
            {session === "race" ? (
              <>
                <dt className="text-muted-fg">Expected finish</dt>
                <dd className="font-mono text-fg tnum">P{r.exp_pos.toFixed(1)}</dd>
              </>
            ) : (
              r.q && (
                <>
                  <dt className="text-muted-fg">Expected qualifying</dt>
                  <dd className="font-mono text-fg tnum">P{r.q.exp_grid.toFixed(1)}</dd>
                </>
              )
            )}
            <dt className="text-muted-fg">Starting position</dt>
            <dd className="font-mono text-fg tnum">
              {r.grid == null ? "not set (pre-quali)" : gridLabel(r.grid)}
            </dd>
          </dl>
        </div>
      </div>
    </div>
  );
}

export function ProbabilityTable({
  drivers,
  qualifying,
}: {
  drivers: DriverPrediction[];
  qualifying?: QualifyingBlock | null;
}) {
  const hasQuali = !!qualifying?.available && qualifying.drivers.length > 0;
  const hasActual = hasQuali && qualifying!.drivers.some((d) => d.actual_grid != null);
  const [session, setSession] = useState<Session>("race");
  // "exp" = order by expected finish (race) / expected grid (qualifying); the default view
  const [sortBy, setSortBy] = useState<"exp" | Metric>("exp");
  const [openCode, setOpenCode] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -80px 0px" });
  const uid = useId();

  const rows: Row[] = useMemo(() => {
    const byCode = new Map((qualifying?.drivers ?? []).map((q) => [q.code, q]));
    return drivers.map((d) => ({ ...d, q: byCode.get(d.code) ?? null }));
  }, [drivers, qualifying]);

  const view: Session = hasQuali ? session : "race";
  const metrics = METRICS[view];
  const expMode = sortBy === "exp";
  // in expected-order mode the bars and highlighted column show the headline probability (win / pole)
  const metric: Metric = expMode ? metrics[0].key : (metrics.find((x) => x.key === sortBy)?.key ?? metrics[0].key);
  const expOf = (r: Row) => (view === "race" ? r.exp_pos : (r.q?.exp_grid ?? 99));
  const sorted = useMemo(
    () =>
      [...rows].sort((a, b) =>
        expMode ? expOf(a) - expOf(b) || val(b, metrics[0].key) - val(a, metrics[0].key) : val(b, metric) - val(a, metric) || b.p_win - a.p_win,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rows, metric, expMode, view],
  );
  const domain = barDomain(rows, metric);
  const active = metrics.find((x) => x.key === metric) ?? metrics[0];
  const expLabel = view === "race" ? "finish" : "qualifying";

  function pickSession(s: Session) {
    setSession(s);
    setSortBy("exp");
  }

  // two trailing columns differ per session
  const COLS = view === "race" ? COLS_RACE : COLS_QUALI;
  const tailHeads = view === "race" ? ["Grid", "Exp finish", "DNF"] : ["Exp quali", hasActual ? "Actual" : "Q1 out"];
  const tail = (r: Row): ReactNode[] =>
    view === "race"
      ? [
          r.grid == null ? "–" : r.grid === 0 ? "pit" : `P${r.grid}`,
          <span key="e">
            {r.exp_pos.toFixed(1)}
            <DeltaVsGrid r={r} />
          </span>,
          fmtPct(r.p_dnf),
        ]
      : r.q
        ? [r.q.exp_grid.toFixed(1), hasActual ? gridLabel(r.q.actual_grid) : fmtPct(r.q.p_q1_exit)]
        : ["–", "–"];

  return (
    <section aria-labelledby="table-h" ref={ref}>
      <div className="flex flex-col gap-3">
        <div>
          <h2 id="table-h" className="display text-3xl text-fg">
            Driver probabilities
          </h2>
          <p className="mt-1 text-[13px] text-muted-fg">
            {expMode ? `Ordered by expected ${expLabel === "finish" ? "finishing position" : "qualifying position"}. ` : ""}
            Bars show the chance to {active.long}, scaled 0 to {Math.round(domain * 100)}%. Select a row for detail.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {hasQuali && (
            <div role="radiogroup" aria-label="Session" className="inline-flex rounded-lg border border-line-strong p-0.5">
              {(["race", "quali"] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  role="radio"
                  aria-checked={view === s}
                  onClick={() => pickSession(s)}
                  className={`h-8 rounded-md px-3 text-[13px] font-medium transition-colors ${
                    view === s ? "bg-fg text-bg" : "text-muted-fg hover:text-fg"
                  }`}
                >
                  {s === "race" ? "Race" : hasActual ? "Qualifying (pre-quali prediction)" : "Qualifying"}
                </button>
              ))}
            </div>
          )}
          <div role="radiogroup" aria-label="Ranking metric" className="inline-flex rounded-lg border border-line-strong p-0.5">
            <button
              type="button"
              role="radio"
              aria-checked={expMode}
              onClick={() => setSortBy("exp")}
              className={`h-8 rounded-md px-2.5 text-[13px] font-medium transition-colors sm:px-3 ${
                expMode ? "bg-signal text-signal-fg" : "text-muted-fg hover:text-fg"
              }`}
            >
              <span className="sm:hidden">{view === "race" ? "Finish" : "Quali"}</span>
              <span className="hidden sm:inline">Expected {expLabel}</span>
            </button>
            {metrics.map((mt) => (
              <button
                key={mt.key}
                type="button"
                role="radio"
                aria-checked={!expMode && active.key === mt.key}
                onClick={() => setSortBy(mt.key)}
                className={`h-8 rounded-md px-2.5 text-[13px] font-medium transition-colors sm:px-3 ${
                  !expMode && active.key === mt.key ? "bg-signal text-signal-fg" : "text-muted-fg hover:text-fg"
                }`}
              >
                {mt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 overflow-hidden rounded-panel border border-line">
        <div className={`${COLS} border-b border-line bg-sunken px-3 py-2`}>
          <span className="label-mono">#</span>
          <span className="label-mono">Driver</span>
          <span className="label-mono hidden whitespace-nowrap md:block">{active.label} chance</span>
          <span className="label-mono text-right md:hidden">{expMode ? (view === "race" ? "Exp finish" : "Exp quali") : active.label}</span>
          {metrics.map((mt) => (
            <button
              key={mt.key}
              type="button"
              onClick={() => setSortBy(mt.key)}
              className={`label-mono hidden whitespace-nowrap text-right md:block ${!expMode && active.key === mt.key ? "!text-fg" : ""}`}
              aria-label={`Rank by ${mt.label}`}
            >
              {mt.label}
            </button>
          ))}
          {tailHeads.map((h) => (
            <span key={h} className="label-mono hidden whitespace-nowrap text-right md:block">
              {h}
            </span>
          ))}
        </div>

        <ol className="m-0 list-none p-0">
          {sorted.map((r, i) => {
            const open = openCode === r.code;
            const detailId = `${uid}-${r.code}`;
            const tl = tail(r);
            const v = val(r, active.key);
            return (
              <m.li
                key={r.code}
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
                  onClick={() => setOpenCode(open ? null : r.code)}
                  className={`${COLS} relative w-full px-3 py-2.5 text-left transition-colors hover:bg-hover ${open ? "bg-hover" : ""}`}
                >
                  <span className="font-mono text-[13px] text-faint tnum">{i + 1}</span>
                  <span className="flex min-w-0 items-center gap-2.5">
                    <span
                      aria-hidden
                      className="h-8 w-[3px] shrink-0 rounded-sm transition-[width] duration-150 group-hover:w-1.5"
                      style={{ background: teamColor(r.team) }}
                    />
                    <span className="min-w-0">
                      <span className="block truncate text-[15px] leading-5 text-fg">
                        <span className="font-mono text-[13px] font-semibold">{r.code}</span>{" "}
                        <span className="hidden sm:inline">{r.name}</span>
                      </span>
                      <span className="block truncate text-xs leading-4 text-muted-fg">{teamShort(r.team)}</span>
                    </span>
                  </span>
                  <span className="hidden pr-2 md:block">
                    <ProbBar value={v} domain={domain} revealed={inView} delay={Math.min(i, 11) * 0.025} />
                  </span>
                  {/* active metric value; on md+ it sits in its own column below */}
                  <span className="text-right font-mono text-[15px] font-semibold text-fg md:hidden">
                    {expMode ? (
                      <>
                        P{expOf(r).toFixed(1)}
                        {view === "race" && <DeltaVsGrid r={r} />}
                      </>
                    ) : (
                      <PctTicker value={v} duration={0.6} />
                    )}
                  </span>
                  {metrics.map((mt) => (
                    <span
                      key={mt.key}
                      className={`hidden text-right font-mono text-[13px] tnum md:block ${
                        !expMode && active.key === mt.key ? "font-semibold text-fg" : "text-muted-fg"
                      }`}
                    >
                      {!expMode && active.key === mt.key ? <PctTicker value={val(r, mt.key)} duration={0.6} /> : fmtPct(val(r, mt.key))}
                    </span>
                  ))}
                  {tl.map((c, ci) => (
                    <span key={ci} className="hidden text-right font-mono text-[13px] text-muted-fg tnum md:block">
                      {c}
                    </span>
                  ))}
                  {/* bar under the name on small screens */}
                  <span className="col-span-3 mt-2 block pl-[calc(1.75rem+0.75rem+13px)] md:hidden">
                    <ProbBar value={v} domain={domain} revealed={inView} delay={Math.min(i, 11) * 0.025} />
                  </span>
                </button>
                <Detail r={r} session={view} open={open} id={detailId} />
              </m.li>
            );
          })}
        </ol>
      </div>
      <p className="mt-3 text-xs leading-4 text-muted-fg">
        {view === "race"
          ? "Exp finish is the average finishing position over all simulated races (lower is better); the small +/- beside it is the places gained or lost versus the starting grid. DNF is the chance of not finishing."
          : `Exp quali is the average qualifying position over all simulated sessions. ${
              hasActual
                ? "Actual is the starting grid in use (penalties applied only if an override file was set)."
                : "Q1 out is the chance of being knocked out in Q1."
            } Predictions of the qualifying classification, before penalties.`}
        {view === "quali" && qualifying?.evidence && (
          <>
            {" "}
            Over {qualifying.evidence.races} races of 2026, the qualifying model scored a pole log-loss of{" "}
            {qualifying.evidence.pole_logloss.toFixed(3)} against {qualifying.evidence.baseline_pole_logloss.toFixed(3)} for{" "}
            {qualifying.evidence.baseline_name} (lower is better), and a rank correlation of{" "}
            {qualifying.evidence.spearman.toFixed(2)} vs {qualifying.evidence.baseline_spearman.toFixed(2)}.
          </>
        )}
      </p>
    </section>
  );
}
