"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SimulateResponse, StrategyResult } from "@/lib/api";
import { NOISE_FLOOR_S } from "@/lib/strategy-form";
import { fmtPct, fmtRaceTime, fmtSigned, fmtTime } from "@/lib/format";
import { Badge } from "@/components/ui/badge";
import { RangePlot } from "@/components/viz/range-plot";

const DELTA_HELP =
  "Paired gap vs the reference (lowest-mean) strategy: per simulation, this strategy's total minus the reference's total in the same simulation. Reported as the mean and the p10 to p90 range.";
const WIN_PROB_HELP =
  "Secondary. Share of simulations in which this strategy beat the other strategies compared, under the current model (one car in isolation, no traffic). Not a race-win chance.";

// line colours that read on both themes; fixed hexes because SVG strokes cannot use oklch channel tokens
const LINE_COLORS = ["#ff6a3d", "#a78bfa", "#4ade80", "#38bdf8", "#f9a8d4", "#facc15"];

function formatDeltaRange(p10: number, p90: number): string {
  // same stop count -> shared draws -> a single value, not a range
  if (Math.abs(p90 - p10) < 0.005) return `constant ${fmtSigned(p10)}`;
  return `${fmtSigned(p10)} to ${fmtSigned(p90)}`;
}

function buildLapChartData(results: StrategyResult[]) {
  const maxLaps = Math.max(...results.map((r) => r.lap_times.length));
  return Array.from({ length: maxLaps }, (_, i) => {
    const row: Record<string, number> = { lap: i + 1 };
    for (const r of results) {
      if (i < r.lap_times.length) row[r.name] = r.lap_times[i];
    }
    return row;
  });
}

interface DotProps {
  cx?: number;
  cy?: number;
  index?: number;
  key?: React.Key | null;
}

/** Invisible dot on every lap, except a highlighted marker on pit-stop laps. */
function makePitStopDot(pitLaps: number[], color: string) {
  const pitSet = new Set(pitLaps);
  const PitStopDot = (props: DotProps) => {
    const lap = (props.index ?? 0) + 1;
    if (pitSet.has(lap)) {
      return <circle key={props.key} cx={props.cx} cy={props.cy} r={5} fill={color} stroke="#14161a" strokeWidth={1.5} />;
    }
    return <circle key={props.key} cx={props.cx} cy={props.cy} r={0} />;
  };
  return PitStopDot;
}

export function CompareResults({ results, context }: { results: SimulateResponse; context?: string | null }) {
  const sc = results.safety_cars === true;
  const scCopy = sc ? "random safety cars included, no traffic yet" : "no safety cars or traffic yet";
  const sorted = [...results.results].sort((a, b) => a.mean - b.mean);
  const referenceName = results.reference_strategy;
  const lo = Math.min(...sorted.map((r) => r.p10));
  const hi = Math.max(...sorted.map((r) => r.p90));
  const pad = Math.max((hi - lo) * 0.1, 1);
  const lapData = buildLapChartData(sorted);

  return (
    <section aria-labelledby="res-h" className="space-y-6">
      <h2 id="res-h" className="display text-3xl text-fg">
        Results
      </h2>
      {context && <p className="-mt-3 font-mono text-[12px] text-muted-fg">{context}</p>}
      {results.assumptions && results.assumptions.length > 0 && (
        <ul className="max-w-3xl list-disc space-y-1 pl-5 text-xs leading-5 text-muted-fg">
          {results.assumptions.map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      )}

      <div className="panel overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-[13px]">
          <thead>
            <tr className="border-b border-line bg-sunken">
              <th className="label-mono px-3 py-2 font-normal">Strategy</th>
              <th className="label-mono px-3 py-2 font-normal" title={DELTA_HELP}>
                Time vs reference
              </th>
              <th className="label-mono px-3 py-2 font-normal" title={DELTA_HELP}>
                Gap p10 to p90
              </th>
              <th className="label-mono px-3 py-2 text-right font-normal">Mean</th>
              <th className="label-mono px-3 py-2 text-right font-normal">p10</th>
              <th className="label-mono px-3 py-2 text-right font-normal">p50</th>
              <th className="label-mono px-3 py-2 text-right font-normal">p90</th>
              <th className="label-mono px-3 py-2 text-right font-normal" title={WIN_PROB_HELP}>
                Beat others*
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => {
              const isRef = r.name === referenceName;
              return (
                <tr key={r.name} className="border-b border-line last:border-b-0">
                  <td className="px-3 py-2.5 font-medium text-fg">
                    {r.name}
                    {isRef && (
                      <Badge variant="best" className="ml-2">
                        reference
                      </Badge>
                    )}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-[15px] font-semibold text-fg tnum">
                    {isRef ? "0.00s" : `${fmtSigned(r.delta_mean)}`}
                  </td>
                  <td className="px-3 py-2.5 font-mono text-muted-fg tnum">
                    {isRef ? "--" : formatDeltaRange(r.delta_p10, r.delta_p90)}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono text-muted-fg tnum">{fmtRaceTime(r.mean)}</td>
                  <td className="px-3 py-2.5 text-right font-mono text-muted-fg tnum">{fmtRaceTime(r.p10)}</td>
                  <td className="px-3 py-2.5 text-right font-mono text-muted-fg tnum">{fmtRaceTime(r.p50)}</td>
                  <td className="px-3 py-2.5 text-right font-mono text-muted-fg tnum">{fmtRaceTime(r.p90)}</td>
                  <td className="px-3 py-2.5 text-right font-mono text-muted-fg tnum">{fmtPct(r.win_probability)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="max-w-3xl space-y-2 text-xs leading-5 text-muted-fg">
        <p>
          Time vs reference is the paired gap: for each simulation, this strategy&apos;s total minus the reference
          strategy&apos;s total in the same simulation (the reference has the lowest mean time). Strategies with the same
          stop count share their random draws, so their gap is a single number rather than a range. A negative p10 means
          the strategy beat the reference in at least 10% of simulations. Gaps of about {NOISE_FLOOR_S} seconds or less
          are inside the model&apos;s error (replays of real races are off by tens of seconds), so treat them as no
          meaningful difference.
        </p>
        <p>
          *Beat others is the share of simulations in which the strategy beat the other strategies compared, under the
          current model (one car in isolation; {scCopy}). It is not a race-win chance. With shared
          random draws it can read exactly 0% or 100% for a fraction-of-a-second gap, so read the time gap first.
        </p>
      </div>

      <div className="panel p-4 sm:p-5">
        <h3 className="text-base font-semibold text-fg">Race time range</h3>
        <p className="mt-1 mb-4 text-[13px] text-muted-fg">Line is p10 to p90 of total race time, dot is the mean.</p>
        <RangePlot
          rows={sorted.map((r) => ({
            id: r.name,
            label: r.name,
            sub: isRefLabel(r, referenceName),
            p10: r.p10,
            mean: r.mean,
            p90: r.p90,
            best: r.name === referenceName,
            right: fmtRaceTime(r.mean),
          }))}
          domain={[lo - pad, hi + pad]}
          tickFormat={fmtRaceTime}
          axisLabel="total race time"
        />
      </div>

      <div className="panel p-4 sm:p-5">
        <h3 className="text-base font-semibold text-fg">Lap time by lap</h3>
        <div className="mt-3 text-muted-fg">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lapData} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="currentColor" strokeOpacity={0.15} vertical={false} />
              <XAxis
                dataKey="lap"
                tick={{ fill: "currentColor", fontSize: 11 }}
                stroke="currentColor"
                strokeOpacity={0.3}
                label={{ value: "Lap", position: "insideBottom", offset: -4, fill: "currentColor", fontSize: 11 }}
              />
              <YAxis
                tickFormatter={(v) => fmtTime(v)}
                domain={["dataMin - 1", "dataMax + 1"]}
                tick={{ fill: "currentColor", fontSize: 11 }}
                stroke="currentColor"
                strokeOpacity={0.3}
                width={52}
              />
              <Tooltip
                formatter={(v) => fmtTime(Number(v))}
                labelFormatter={(l) => `Lap ${l}`}
                contentStyle={{
                  background: "oklch(var(--raised))",
                  border: "1px solid oklch(var(--line-strong))",
                  borderRadius: 6,
                  color: "oklch(var(--fg))",
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {sorted.map((r, i) => {
                const color = LINE_COLORS[i % LINE_COLORS.length];
                return (
                  <Line
                    key={r.name}
                    type="monotone"
                    dataKey={r.name}
                    stroke={color}
                    strokeWidth={2}
                    isAnimationActive={false}
                    dot={makePitStopDot(r.pit_laps, color)}
                    connectNulls={false}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-xs leading-5 text-muted-fg">
          Dots mark pit stops. One noise-free run per strategy, for shape only: use the table above for the
          statistically meaningful comparison.
        </p>
      </div>
    </section>
  );
}

function isRefLabel(r: StrategyResult, ref: string): string {
  return r.name === ref ? "reference" : `${r.pit_laps.length}-stop`;
}
