"use client";

import { AlertTriangle } from "lucide-react";
import { useMemo } from "react";
import type { OptimizeResponse, StrategyOut } from "@/lib/api";
import { fmtRaceTime, fmtSigned } from "@/lib/format";
import { NOISE_FLOOR_S } from "@/lib/strategy-form";
import { Badge } from "@/components/ui/badge";
import { RangePlot, type RangeRow } from "@/components/viz/range-plot";
import { StintStrip } from "@/components/viz/stint-strip";

/** Plans identical up to stint order are the same plan for display purposes: key = sorted (compound, laps) multiset. */
export function dedupeStrategies(strategies: StrategyOut[]): StrategyOut[] {
  const seen = new Set<string>();
  const out: StrategyOut[] = [];
  for (const s of strategies) {
    const key = s.stints
      .map((st) => `${st.compound}:${st.laps}`)
      .sort()
      .join("|");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
  }
  return out;
}

const planKey = (s: StrategyOut) => s.stints.map((x) => `${x.compound[0]}${x.laps}`).join("-");
const stopsLabel = (n: number) => (n === 0 ? "No stop" : `${n}-stop`);
const pitText = (s: StrategyOut) => (s.pit_laps.length ? `pit on lap ${s.pit_laps.join(", ")}` : "no pit stop");

function Caps({ result }: { result: OptimizeResponse }) {
  const { params_used: pu, track } = result;
  const eff = pu.effective_max_stint_laps;
  const comps = ["SOFT", "MEDIUM", "HARD"].filter((c) => eff?.[c] !== undefined || track.max_stint_laps[c] !== undefined);
  if (comps.length === 0 && pu.min_stint_laps == null) return null;
  return (
    <div>
      <h4 className="label-mono">Stint limits used</h4>
      <table className="mt-2 w-full max-w-sm text-[13px]">
        <thead>
          <tr className="text-left text-muted-fg">
            <th className="py-1 pr-3 font-normal">Tyre</th>
            <th className="py-1 pr-3 text-right font-normal">Longest seen here</th>
            <th className="py-1 text-right font-normal">Cap in search</th>
          </tr>
        </thead>
        <tbody>
          {comps.map((c) => {
            const obs = track.max_stint_laps[c];
            const cap = eff?.[c];
            return (
              <tr key={c} className="border-t border-line">
                <td className="py-1.5 pr-3">
                  <span aria-hidden className="mr-2 inline-block size-2 rounded-full" style={{ background: `var(--tyre-${c.toLowerCase()})` }} />
                  {c.charAt(0) + c.slice(1).toLowerCase()}
                </td>
                <td className="py-1.5 pr-3 text-right font-mono tnum">{obs !== undefined ? `${obs} laps` : "n/a"}</td>
                <td className="py-1.5 text-right font-mono tnum">
                  {cap !== undefined ? `${cap} laps` : "n/a"}
                  {cap !== undefined && obs !== undefined && cap !== obs ? " *" : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-xs leading-4 text-muted-fg">
        {eff && comps.some((c) => eff[c] !== track.max_stint_laps[c]) &&
          "* Raised to the 15-lap floor or clipped to the race distance. "}
        {pu.min_stint_laps != null && `Every stint is at least ${pu.min_stint_laps} laps.`}
      </p>
    </div>
  );
}

export function OptimizerResults({ result, topShown = 5 }: { result: OptimizeResponse; topShown?: number }) {
  const plans = useMemo(() => dedupeStrategies(result.strategies).slice(0, topShown), [result.strategies, topShown]);
  const best = plans[0];
  const compounds = result.track.compounds ?? {};
  const placeholders = result.assumptions.filter((a) => /placeholder/i.test(a));
  const near = best ? plans.slice(1).filter((p) => p.delta_to_best_mean_s < NOISE_FLOOR_S) : [];

  const rows: RangeRow[] = plans.map((p, i) => ({
    id: planKey(p),
    label: `#${i + 1} ${stopsLabel(p.n_stops)}`,
    sub: p.stints.map((s) => `${s.compound[0]}${s.laps}`).join(" "),
    p10: p.delta_to_best_p10_s,
    mean: p.delta_to_best_mean_s,
    p90: p.delta_to_best_p90_s,
    best: i === 0,
    right: i === 0 ? "best" : fmtSigned(p.delta_to_best_mean_s, 1),
  }));
  const lo = Math.min(0, ...rows.map((r) => r.p10));
  const hi = Math.max(NOISE_FLOOR_S, ...rows.map((r) => r.p90));
  const pad = (hi - lo) * 0.06;

  if (!best) return null;

  return (
    <section aria-labelledby="opt-h" className="space-y-6">
      {placeholders.length > 0 && (
        <div role="note" className="flex gap-3 rounded-panel border border-warn/50 bg-warn/10 p-4 text-sm">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden />
          <div className="min-w-0">
            <p className="font-medium text-fg">Some tyre inputs are still rough estimates, so treat compound trade-offs as indicative</p>
            {placeholders.map((a, i) => (
              <p key={i} className="mt-1 text-muted-fg [overflow-wrap:anywhere]">
                {a}
              </p>
            ))}
          </div>
        </div>
      )}

      <div className="panel p-4 sm:p-5">
        <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-1">
          <div>
            <h2 id="opt-h" className="label-mono">
              Optimal plan · {result.track.name}
            </h2>
            <p className="display mt-1 text-4xl text-fg sm:text-5xl">
              {stopsLabel(best.n_stops)}
              <span className="ml-3 font-sans text-base font-normal tracking-normal text-muted-fg" style={{ fontStretch: "100%" }}>
                {pitText(best)}
              </span>
            </p>
          </div>
          <div className="sm:text-right">
            <p className="font-mono text-2xl text-fg tnum">{fmtRaceTime(best.mean_s)}</p>
            <p className="font-mono text-xs text-muted-fg tnum">
              p10 to p90 {fmtRaceTime(best.p10_s)} to {fmtRaceTime(best.p90_s)}
            </p>
          </div>
        </div>
        <div className="mt-4">
          <StintStrip stints={best.stints} size="lg" playKey={planKey(best)} labels={compounds} />
        </div>
        <p className="mt-3 text-[13px] leading-5 text-muted-fg">
          {near.length > 0
            ? `${near.length} other plan${near.length === 1 ? " is" : "s are"} within ${NOISE_FLOOR_S} s of this one, which is inside the model's noise: treat ${near.length === 1 ? "it" : "them"} as equivalent, not worse.`
            : `The next plan is ${fmtSigned(plans[1]?.delta_to_best_mean_s ?? 0, 1)} behind in the model, which is outside the ${NOISE_FLOOR_S} s noise band.`}
        </p>
      </div>

      <div className="panel p-4 sm:p-5">
        <h3 className="text-base font-semibold text-fg">Top plans</h3>
        <p className="mt-1 text-[13px] text-muted-fg">
          {result.params_used.total_laps} laps, {result.params_used.pit_loss_s.toFixed(1)} s pit loss,{" "}
          {result.params_used.base_laptime_s.toFixed(2)} s base lap,{" "}
          {result.params_used.n_sims.toLocaleString("en-US")} runs per plan. {result.n_candidates_screened.toLocaleString("en-US")}{" "}
          plans screened, {result.n_candidates_simulated} simulated.
        </p>
        <ul className="mt-4 divide-y divide-line">
          {plans.map((p, i) => (
            <li key={planKey(p)} className="grid gap-2 py-3 first:pt-0 sm:grid-cols-[11rem_1fr] sm:items-center sm:gap-4">
              <div className="min-w-0 text-[13px] leading-5">
                <p className="font-medium text-fg">
                  <span className="font-mono text-muted-fg">#{i + 1}</span> {stopsLabel(p.n_stops)}
                </p>
                <p className="text-muted-fg">{pitText(p)}</p>
                <p className="mt-0.5 flex flex-wrap items-center gap-1.5 font-mono text-xs text-fg tnum">
                  {fmtRaceTime(p.mean_s)}
                  {i === 0 ? (
                    <Badge variant="best">best</Badge>
                  ) : p.delta_to_best_mean_s < NOISE_FLOOR_S ? (
                    <Badge variant="outline">within noise</Badge>
                  ) : null}
                </p>
              </div>
              <StintStrip stints={p.stints} size="md" playKey={planKey(p)} labels={compounds} />
            </li>
          ))}
        </ul>
      </div>

      <div className="panel p-4 sm:p-5">
        <h3 className="text-base font-semibold text-fg">Gap to the best plan</h3>
        <p className="mt-1 mb-4 max-w-2xl text-[13px] leading-5 text-muted-fg">
          Line is the p10 to p90 range of the paired gap, dot is the mean. Plans with the same stop count share their
          random draws, so their range can collapse to a point. The purple rule is the best plan; gaps under{" "}
          {NOISE_FLOOR_S} s are inside the model&apos;s error.
        </p>
        <RangePlot
          rows={rows}
          domain={[lo - pad, hi + pad]}
          rule={0}
          ruleLabel="Best plan"
          tickFormat={(v) => `${v > 0 ? "+" : ""}${v}s`}
          axisLabel="seconds vs best"
        />
      </div>

      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="panel p-4 sm:p-5">
          <Caps result={result} />
        </div>
        <details open className="panel group min-w-0 p-4 sm:p-5">
          <summary className="cursor-pointer list-none text-base font-semibold text-fg [&::-webkit-details-marker]:hidden">
            What this model ignores
            <span className="ml-2 font-mono text-xs font-normal text-faint group-open:hidden">(show)</span>
          </summary>
          <ul className="mt-3 list-disc space-y-1.5 pl-5 text-[13px] leading-5 text-muted-fg">
            {result.assumptions.map((a, i) => (
              <li key={i} className="[overflow-wrap:anywhere]">{a}</li>
            ))}
          </ul>
        </details>
      </div>
    </section>
  );
}
