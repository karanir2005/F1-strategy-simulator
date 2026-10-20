"use client";

import { Loader2, Plus, X } from "lucide-react";
import { MAX_RACE_LAPS, MAX_START_AGE, MAX_STRATEGIES, type Compound } from "@/lib/api";
import { COMPOUNDS } from "@/lib/format";
import type { StrategyRow } from "@/lib/strategy-form";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StintStrip } from "@/components/viz/stint-strip";

export interface ManualCompareProps {
  strategies: StrategyRow[];
  onAddStrategy: () => void;
  onRemoveStrategy: (id: string) => void;
  onRename: (id: string, name: string) => void;
  onAddStint: (strategyId: string) => void;
  onRemoveStint: (strategyId: string, stintId: string) => void;
  onUpdateStint: (strategyId: string, stintId: string, field: "compound" | "laps" | "startAge", value: string) => void;
  onUsedSet: (strategyId: string, stintId: string, usedSet: boolean) => void;
  onRun: () => void;
  loading: boolean;
  validationError: string | null;
}

const COMPOUND_NAME: Record<Compound, string> = { SOFT: "Soft", MEDIUM: "Medium", HARD: "Hard" };

export function ManualCompare(p: ManualCompareProps) {
  return (
    <section aria-labelledby="manual-h" className="space-y-4">
      <div>
        <h2 id="manual-h" className="display text-3xl text-fg">
          Compare your own strategies
        </h2>
        <p className="mt-1 max-w-2xl text-[13px] leading-5 text-muted-fg">
          Build up to {MAX_STRATEGIES} plans, each with the same total laps. The result is the paired gap between them
          over thousands of noisy runs.
        </p>
      </div>

      <div className="space-y-3">
        {p.strategies.map((strategy) => {
          const total = strategy.stints.reduce((s, st) => s + (typeof st.laps === "number" ? st.laps : 0), 0);
          const preview = strategy.stints
            .filter((st) => typeof st.laps === "number" && st.laps > 0)
            .map((st) => ({ compound: st.compound, laps: st.laps as number }));
          return (
            <div key={strategy.id} className="panel p-4">
              <div className="flex items-center justify-between gap-3">
                <Input
                  value={strategy.name}
                  onChange={(e) => p.onRename(strategy.id, e.target.value)}
                  aria-label="Strategy name"
                  className="max-w-[16rem] font-sans font-medium"
                />
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs text-muted-fg tnum">{total} laps</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => p.onRemoveStrategy(strategy.id)}
                    disabled={p.strategies.length <= 1}
                    aria-label={`Remove strategy ${strategy.name}`}
                  >
                    Remove
                  </Button>
                </div>
              </div>

              {preview.length > 0 && (
                <div className="mt-3">
                  <StintStrip stints={preview} size="sm" />
                </div>
              )}

              <ul className="mt-3 space-y-2">
                {strategy.stints.map((stint, idx) => (
                  <li key={stint.id} className="flex flex-wrap items-center gap-x-3 gap-y-2">
                    <span className="w-5 font-mono text-xs text-faint tnum">{idx + 1}</span>
                    <div role="radiogroup" aria-label={`Stint ${idx + 1} tyre`} className="inline-flex rounded-lg border border-line-strong p-0.5">
                      {COMPOUNDS.map((c) => (
                        <button
                          key={c}
                          type="button"
                          role="radio"
                          aria-checked={stint.compound === c}
                          onClick={() => p.onUpdateStint(strategy.id, stint.id, "compound", c)}
                          className={`flex h-8 items-center gap-1.5 rounded-md px-2.5 text-[13px] font-medium transition-colors ${
                            stint.compound === c ? "bg-hover text-fg" : "text-muted-fg hover:text-fg"
                          }`}
                        >
                          <span
                            aria-hidden
                            className="size-2.5 rounded-full ring-1 ring-line-strong"
                            style={{ background: `var(--tyre-${c.toLowerCase()})` }}
                          />
                          {COMPOUND_NAME[c]}
                        </button>
                      ))}
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Input
                        type="number"
                        min={1}
                        max={MAX_RACE_LAPS}
                        aria-label={`Stint ${idx + 1} laps`}
                        value={stint.laps}
                        onChange={(e) => p.onUpdateStint(strategy.id, stint.id, "laps", e.target.value)}
                        className="w-20"
                      />
                      <span className="text-[13px] text-muted-fg">laps</span>
                    </div>
                    <label className="flex cursor-pointer items-center gap-1.5 text-[13px] text-muted-fg">
                      <input
                        type="checkbox"
                        className="size-4 accent-signal"
                        checked={stint.usedSet}
                        onChange={(e) => p.onUsedSet(strategy.id, stint.id, e.target.checked)}
                      />
                      used set
                    </label>
                    {stint.usedSet && (
                      <div className="flex items-center gap-1.5">
                        <Input
                          type="number"
                          min={1}
                          max={MAX_START_AGE}
                          step={1}
                          aria-label={`Stint ${idx + 1} starting tyre age in laps`}
                          title="Tyre age on the stint's first lap. A set already run N laps = N + 1."
                          value={stint.startAge}
                          onChange={(e) => p.onUpdateStint(strategy.id, stint.id, "startAge", e.target.value)}
                          className="w-20"
                        />
                        <span className="text-[13px] text-muted-fg">start age</span>
                      </div>
                    )}
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => p.onRemoveStint(strategy.id, stint.id)}
                      disabled={strategy.stints.length <= 1}
                      aria-label={`Remove stint ${idx + 1}`}
                      className="ml-auto"
                    >
                      <X aria-hidden />
                    </Button>
                  </li>
                ))}
              </ul>
              <Button variant="outline" size="sm" className="mt-3" onClick={() => p.onAddStint(strategy.id)}>
                <Plus aria-hidden /> Add stint
              </Button>
            </div>
          );
        })}
        <Button variant="outline" onClick={p.onAddStrategy} disabled={p.strategies.length >= MAX_STRATEGIES}>
          <Plus aria-hidden /> Add strategy
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button size="lg" onClick={p.onRun} disabled={p.loading || !!p.validationError}>
          {p.loading && <Loader2 className="size-4 animate-spin motion-reduce:animate-none" aria-hidden />}
          {p.loading ? "Running simulation..." : "Run simulation"}
        </Button>
        {p.validationError && !p.loading && <p className="text-[13px] text-muted-fg">{p.validationError}</p>}
      </div>
    </section>
  );
}
