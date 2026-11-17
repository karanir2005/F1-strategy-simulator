"use client";

import { Loader2 } from "lucide-react";
import { useState } from "react";
import { runPitPolicy, type PolicyGain, type PolicyResponse, type StintOut } from "@/lib/api";
import { fmtPct } from "@/lib/format";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ErrorBlock } from "@/components/viz/states";

const PLURAL: Record<string, string> = { SOFT: "Softs", MEDIUM: "Mediums", HARD: "Hards" };
const signed = (v: number, d = 1) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;

function ruleSentence(r: PolicyResponse): string | null {
  const p = r.policy;
  if (!p) return null;
  const event = p.trigger === "sc" ? "a safety car comes out" : "a safety car or virtual safety car comes out";
  return `If ${event} between laps ${p.window_lo}–${p.window_hi} and your tyres are ≥ ${p.min_age} laps old, box for ${PLURAL[p.compound]}.`;
}

const planText = (plan: StintOut[]) => plan.map((s) => `${s.compound[0]}${s.laps}`).join("-");

function GainLine({ gain, label }: { gain: PolicyGain; label: string }) {
  return (
    <p className="text-[13px] leading-5 text-muted-fg">
      {label}{" "}
      <span className="font-mono text-fg tnum">
        {signed(gain.points)} pts (±{gain.points_ci95.toFixed(1)})
      </span>
      , {signed(gain.positions, 2)} places (±{gain.positions_ci95.toFixed(2)}).
    </p>
  );
}

/**
 * T-055: a safety-car-reactive rule for one car, found by the full-grid simulation. Runs on demand (it takes
 * 10 to 20 seconds). `plan` is the optimizer's best plan so the rule is added to what is shown above it.
 */
export function PitPolicyCard({
  trackId,
  plan,
  totalLaps,
}: {
  trackId: string;
  plan: { compound: "SOFT" | "MEDIUM" | "HARD"; laps: number }[];
  totalLaps: number;
}) {
  const [res, setRes] = useState<PolicyResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setRes(await runPitPolicy({ track_id: trackId, base_plan: plan, total_laps: totalLaps }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Safety-car rule search failed");
    } finally {
      setBusy(false);
    }
  }

  const sentence = res ? ruleSentence(res) : null;
  return (
    <div className="panel p-4 sm:p-5" aria-live="polite">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-base font-semibold text-fg">If a safety car comes out</h3>
          <p className="mt-1 max-w-2xl text-[13px] leading-5 text-muted-fg">
            The plan above is fixed. This searches for a rule that reacts to a safety car or virtual safety car (a stop costs
            far less under one), scored over the whole grid against keeping the plan.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={run} disabled={busy}>
          {busy ? (
            <>
              <Loader2 className="mr-1.5 size-3.5 animate-spin" aria-hidden /> Searching
            </>
          ) : res ? (
            "Search again"
          ) : (
            "Find a safety-car rule"
          )}
        </Button>
      </div>

      {error && (
        <div className="mt-3">
          <ErrorBlock title="Search failed" message={error} onRetry={run} />
        </div>
      )}

      {res && (
        <div className="mt-4 space-y-2">
          <p className="text-[15px] leading-6 text-fg">
            {sentence ?? res.rule}{" "}
            {res.significant && res.gain ? (
              <span className="font-medium">
                Expected gain {signed(res.gain.points)} pts (±{res.gain.points_ci95.toFixed(1)}).
              </span>
            ) : (
              <span className="text-muted-fg">
                {res.policy ? "The gain is inside the simulation noise, so there is no reliable rule here." : ""}
              </span>
            )}
          </p>
          {res.gain && res.policy && (
            <>
              <GainLine gain={res.gain} label={`Over all runs, vs keeping ${planText(res.base_plan)} for ${res.driver}:`} />
              <p className="text-[13px] leading-5 text-muted-fg">
                The rule fires in {fmtPct(res.gain.fire_rate)} of runs
                {res.points_gain_when_fired != null && (
                  <>
                    ; when it does, {signed(res.points_gain_when_fired)} pts (±{(res.points_gain_when_fired_ci95 ?? 0).toFixed(1)})
                  </>
                )}
                .
              </p>
              {res.if_field_follows && (
                <GainLine gain={res.if_field_follows} label="If every other car follows the same rule:" />
              )}
            </>
          )}
          <p className="flex flex-wrap items-center gap-2 pt-1 text-xs text-faint">
            <Badge variant="outline">
              {res.driver} from P{res.grid}
            </Badge>
            <span>
              {res.n_sims.toLocaleString("en-US")} runs, interval is 95%, rule chosen on a different random draw. The tyre
              cliff values are still rough, so a late fresh set may be over-valued.
            </span>
          </p>
        </div>
      )}
    </div>
  );
}
