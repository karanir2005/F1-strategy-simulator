"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { OptimizeResponse, StrategyOut } from "@/lib/api";

const BAR_COLOR: Record<string, string> = {
  SOFT: "bg-red-500",
  MEDIUM: "bg-yellow-500",
  HARD: "bg-neutral-400",
};

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

function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

function signed(seconds: number): string {
  return `${seconds >= 0 ? "+" : "-"}${Math.abs(seconds).toFixed(2)}s`;
}

export function OptimizerResults({
  result,
  topShown = 5,
}: {
  result: OptimizeResponse;
  topShown?: number;
}) {
  const plans = dedupeStrategies(result.strategies).slice(0, topShown);
  const compounds = result.track.compounds ?? {};
  const label = (c: string) => (compounds[c] ? `${c[0]} ${compounds[c]}` : c[0]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Best plans for {result.track.name}</CardTitle>
        <p className="text-xs text-muted-foreground">
          {result.params_used.total_laps} laps, {result.params_used.pit_loss_s.toFixed(1)}s pit loss,{" "}
          {result.params_used.base_laptime_s.toFixed(2)}s base lap,{" "}
          {result.params_used.n_sims.toLocaleString("en-US")} sims per plan.{" "}
          {result.n_candidates_screened.toLocaleString("en-US")} plans screened, {result.n_candidates_simulated}{" "}
          simulated.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {plans.map((p, i) => (
          <div key={p.rank} className="space-y-1">
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
              <span className="font-semibold">#{i + 1}</span>
              <span>
                {p.n_stops}-stop, pit on lap {p.pit_laps.join(", ")}
              </span>
              <span className="font-medium">{formatTime(p.mean_s)}</span>
              {i === 0 ? (
                <Badge variant="outline">best</Badge>
              ) : (
                <span className="text-muted-foreground">
                  {signed(p.delta_to_best_mean_s)} vs best (p10 to p90 {signed(p.delta_to_best_p10_s)} to{" "}
                  {signed(p.delta_to_best_p90_s)})
                </span>
              )}
            </div>
            <div className="flex h-7 w-full overflow-hidden rounded-md text-xs text-white">
              {p.stints.map((st, j) => (
                <div
                  key={j}
                  style={{ flexGrow: st.laps, flexBasis: 0 }}
                  className={`${BAR_COLOR[st.compound] ?? "bg-neutral-500"} flex min-w-0 items-center justify-center border-r border-white/60 last:border-r-0`}
                  title={`${st.compound} ${st.laps} laps`}
                >
                  <span className="truncate px-1">
                    {label(st.compound)} {st.laps}
                  </span>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              Race time p10 to p90: {formatTime(p.p10_s)} to {formatTime(p.p90_s)}
            </p>
          </div>
        ))}
        <p className="text-xs text-muted-foreground">
          Gaps under about 5 s are inside the model&apos;s noise: treat those plans as equivalent. Compound offsets and
          cliff penalties are still placeholders, so the ranking may lean toward softer tyres than a real race would.
        </p>
        <div className="rounded-md border bg-muted/40 px-4 py-3 text-sm">
          <p className="font-medium">What this model ignores</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 text-xs text-muted-foreground">
            {result.assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
