"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { OptimizerResults } from "@/components/optimizer-results";
import {
  MAX_RACE_LAPS,
  MAX_SIMULATIONS,
  MAX_START_AGE,
  MAX_STRATEGIES,
  fetchTracks,
  runOptimize,
  runSimulation,
  type Compound,
  type Strategy,
  type StrategyResult,
  type SimulateResponse,
  type TrackOut,
  type OptimizeResponse,
} from "@/lib/api";

let nextId = 0;
const newId = () => `${Date.now()}-${nextId++}`;

// monotonic, never reused even after strategies are removed -- naming
// off prev.length + 1 can collide (add a 3rd, remove the 1st, add
// again -> "strategy-3" twice), which silently drops a strategy from
// results since the backend keys them by name
let nextStrategyNumber = 3;

/** A numeric text field: "" while the user has cleared it (validated before run). */
type NumField = number | "";

interface StintRow {
  id: string;
  compound: Compound;
  laps: NumField;
  /** Used-set flag, kept separate from startAge so typing "15" can't collapse the input. */
  usedSet: boolean;
  /** Tyre age on the stint's first lap; only sent when usedSet. 1 = new set. */
  startAge: NumField;
}

interface StrategyRow {
  id: string;
  name: string;
  stints: StintRow[];
}

const initialStrategies: StrategyRow[] = [
  {
    id: newId(),
    name: "1-stop",
    stints: [
      { id: newId(), compound: "MEDIUM", laps: 25, usedSet: false, startAge: 2 },
      { id: newId(), compound: "HARD", laps: 32, usedSet: false, startAge: 2 },
    ],
  },
  {
    id: newId(),
    name: "2-stop",
    stints: [
      { id: newId(), compound: "SOFT", laps: 15, usedSet: false, startAge: 2 },
      { id: newId(), compound: "MEDIUM", laps: 20, usedSet: false, startAge: 2 },
      { id: newId(), compound: "HARD", laps: 22, usedSet: false, startAge: 2 },
    ],
  },
];

function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

interface DotProps {
  cx?: number;
  cy?: number;
  index?: number;
  key?: React.Key | null;
}

/** Renders an invisible dot on every lap, except a highlighted marker on pit-stop laps. */
function makePitStopDot(pitLaps: number[], color: string) {
  const pitSet = new Set(pitLaps);
  const PitStopDot = (props: DotProps) => {
    const lap = (props.index ?? 0) + 1;
    if (pitSet.has(lap)) {
      return (
        <circle
          key={props.key}
          cx={props.cx}
          cy={props.cy}
          r={5}
          fill={color}
          stroke="white"
          strokeWidth={1.5}
        />
      );
    }
    return <circle key={props.key} cx={props.cx} cy={props.cy} r={0} />;
  };
  return PitStopDot;
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

function parseNum(value: string): NumField {
  if (value === "") return "";
  const n = Number(value);
  return Number.isNaN(n) ? "" : n;
}

/** Checks for the editable race numbers (shown when overriding a track, or when no track is available). */
function getNumbersError(
  baseLaptime: NumField,
  totalLaps: NumField,
  pitLossSeconds: NumField,
): string | null {
  if (baseLaptime === "" || baseLaptime <= 0 || baseLaptime > 300) {
    return "Base lap time must be greater than 0 and at most 300 seconds.";
  }
  if (pitLossSeconds === "" || pitLossSeconds < 0 || pitLossSeconds > 120) {
    return "Pit stop loss must be between 0 and 120 seconds.";
  }
  if (totalLaps === "" || !Number.isInteger(totalLaps) || totalLaps < 1 || totalLaps > MAX_RACE_LAPS) {
    return `Total laps must be a whole number from 1 to ${MAX_RACE_LAPS}.`;
  }
  return null;
}

function getValidationError(
  numbersEditable: boolean,
  baseLaptime: NumField,
  totalLaps: NumField,
  nSimulations: NumField,
  pitLossSeconds: NumField,
  strategies: StrategyRow[],
): string | null {
  if (numbersEditable) {
    const err = getNumbersError(baseLaptime, totalLaps, pitLossSeconds);
    if (err) return err;
  }
  if (nSimulations === "" || !Number.isInteger(nSimulations) || nSimulations <= 0) {
    return "Simulations per strategy must be a positive whole number.";
  }
  if (nSimulations > MAX_SIMULATIONS) {
    return `Simulations per strategy must be at most ${MAX_SIMULATIONS.toLocaleString("en-US")}.`;
  }
  if (strategies.length === 0) return "Add at least one strategy.";
  if (strategies.length > MAX_STRATEGIES) {
    return `At most ${MAX_STRATEGIES} strategies can be compared at once (you have ${strategies.length}).`;
  }
  const names = strategies.map((s) => s.name.trim());
  const seen = new Set<string>();
  for (const name of names) {
    if (seen.has(name)) return `Two strategies are both named "${name}" -- names must be unique.`;
    seen.add(name);
  }
  for (const strategy of strategies) {
    if (!strategy.name.trim()) return "Every strategy needs a name.";
    if (strategy.stints.length === 0) {
      return `"${strategy.name}" needs at least one stint.`;
    }
    for (const stint of strategy.stints) {
      if (stint.laps === "" || !Number.isInteger(stint.laps) || stint.laps <= 0) {
        return `"${strategy.name}" has a stint with a missing or invalid lap count -- must be a positive whole number.`;
      }
      if (stint.laps > MAX_RACE_LAPS) {
        return `"${strategy.name}" has a stint of ${stint.laps} laps -- the limit is ${MAX_RACE_LAPS} per stint.`;
      }
      if (stint.usedSet) {
        if (
          stint.startAge === "" ||
          !Number.isInteger(stint.startAge) ||
          stint.startAge < 1 ||
          stint.startAge > MAX_START_AGE
        ) {
          return `"${strategy.name}" has a used set with a missing or invalid starting tyre age -- must be a whole number from 1 to ${MAX_START_AGE} (1 = new set).`;
        }
      }
    }
  }
  const totals = strategies.map((s) => ({
    name: s.name.trim(),
    total: s.stints.reduce((sum, st) => sum + (st.laps as number), 0),
  }));
  const tooLong = totals.find((t) => t.total > MAX_RACE_LAPS);
  if (tooLong) {
    return `"${tooLong.name}" totals ${tooLong.total} laps -- the limit is ${MAX_RACE_LAPS} laps per strategy.`;
  }
  if (totals.some((t) => t.total !== totals[0].total)) {
    return `All strategies must cover the same total race laps to be compared (${totals
      .map((t) => `"${t.name}": ${t.total}`)
      .join(", ")}).`;
  }
  return null;
}

const COMPOUND_COLOR: Record<Compound, string> = {
  SOFT: "bg-red-500",
  MEDIUM: "bg-yellow-500",
  HARD: "bg-neutral-400",
};

const DELTA_HELP =
  "Paired gap vs the reference (lowest-mean) strategy: per simulation, this strategy's total minus the reference's total in the same simulation. Reported as the mean and the p10 to p90 range.";

function formatDelta(seconds: number): string {
  return `${seconds >= 0 ? "+" : "-"}${Math.abs(seconds).toFixed(2)}s`;
}

function formatDeltaRange(p10: number, p90: number): string {
  // same stop count -> shared draws -> a single value, not a range
  if (Math.abs(p90 - p10) < 0.005) return `constant ${formatDelta(p10)}`;
  return `${formatDelta(p10)} to ${formatDelta(p90)}`;
}

const WIN_PROB_HELP =
  "Secondary. Share of simulations in which this strategy beat the other strategies compared, under the current model (one car in isolation; no safety cars or traffic yet). Not a race-win chance. Often near 0% / 100% even for tiny gaps -- use the time-vs-reference column.";

const CHART_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"];

// mirrors race_simulator.py's DEFAULT_PIT_LOSS_SECONDS -- kept editable
// here since it's genuinely track-dependent (e.g. Monaco's pit lane costs
// noticeably more than a short one like Qatar's)
const DEFAULT_PIT_LOSS_SECONDS = 23.5;

export default function Home() {
  const [baseLaptime, setBaseLaptime] = useState<NumField>(90);
  const [nSimulations, setNSimulations] = useState<NumField>(2000);
  const [pitLossSeconds, setPitLossSeconds] = useState<NumField>(DEFAULT_PIT_LOSS_SECONDS);
  const [totalLaps, setTotalLaps] = useState<NumField>(57);
  const [tracks, setTracks] = useState<TrackOut[] | null>(null);
  const [tracksError, setTracksError] = useState<string | null>(null);
  const [trackId, setTrackId] = useState<string>("");
  const [override, setOverride] = useState(false);
  const [maxStops, setMaxStops] = useState(3);
  const [optResult, setOptResult] = useState<OptimizeResponse | null>(null);
  const [optLoading, setOptLoading] = useState(false);
  const [optError, setOptError] = useState<string | null>(null);
  const [strategies, setStrategies] = useState<StrategyRow[]>(initialStrategies);
  const [results, setResults] = useState<SimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const track = tracks?.find((t) => t.track_id === trackId) ?? null;
  // numbers are typed by hand when overriding a track, or when no track is available at all
  const numbersEditable = override || !track;
  const validationError = getValidationError(
    numbersEditable,
    baseLaptime,
    totalLaps,
    nSimulations,
    pitLossSeconds,
    strategies,
  );
  const optValidationError = !track
    ? "Pick a track to find an optimal strategy."
    : override
      ? getNumbersError(baseLaptime, totalLaps, pitLossSeconds)
      : null;

  useEffect(() => {
    let cancelled = false;
    fetchTracks()
      .then((list) => {
        if (cancelled) return;
        setTracks(list);
        const first = list.find((t) => t.track_id === "austin") ?? list[0];
        if (first) setTrackId(first.track_id);
      })
      .catch((err) => {
        if (!cancelled) setTracksError(err instanceof Error ? err.message : "Could not load tracks");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function prefillFrom(t: TrackOut) {
    setBaseLaptime(t.base_laptime_s);
    setPitLossSeconds(t.pit_loss_s);
    setTotalLaps(t.total_laps);
  }

  function selectTrack(id: string) {
    setTrackId(id);
    setOptResult(null);
    setOptError(null);
    const t = tracks?.find((x) => x.track_id === id);
    if (t && override) prefillFrom(t);
  }

  function toggleOverride(on: boolean) {
    setOverride(on);
    if (on && track) prefillFrom(track);
  }

  async function handleOptimize() {
    if (!track || optValidationError) return;
    setOptLoading(true);
    setOptError(null);
    setOptResult(null);
    try {
      setOptResult(
        await runOptimize({
          track_id: track.track_id,
          max_stops: maxStops,
          top_n: 10, // ask for extra so client-side dedupe still leaves 5 distinct plans
          ...(override
            ? {
                total_laps: totalLaps as number,
                pit_loss_s: pitLossSeconds as number,
                base_laptime_s: baseLaptime as number,
              }
            : {}),
        }),
      );
    } catch (err) {
      setOptError(err instanceof Error ? err.message : "Optimizer failed");
    } finally {
      setOptLoading(false);
    }
  }

  function addStrategy() {
    setStrategies((prev) => [
      ...prev,
      {
        id: newId(),
        name: `strategy-${nextStrategyNumber++}`,
        stints: [{ id: newId(), compound: "MEDIUM", laps: 20, usedSet: false, startAge: 2 }],
      },
    ]);
  }

  function removeStrategy(id: string) {
    setStrategies((prev) => prev.filter((s) => s.id !== id));
  }

  function updateStrategyName(id: string, name: string) {
    setStrategies((prev) =>
      prev.map((s) => (s.id === id ? { ...s, name } : s)),
    );
  }

  function addStint(strategyId: string) {
    setStrategies((prev) =>
      prev.map((s) =>
        s.id === strategyId
          ? { ...s, stints: [...s.stints, { id: newId(), compound: "MEDIUM", laps: 10, usedSet: false, startAge: 2 }] }
          : s,
      ),
    );
  }

  function removeStint(strategyId: string, stintId: string) {
    setStrategies((prev) =>
      prev.map((s) =>
        s.id === strategyId
          ? { ...s, stints: s.stints.filter((st) => st.id !== stintId) }
          : s,
      ),
    );
  }

  function updateStint(
    strategyId: string,
    stintId: string,
    field: "compound" | "laps" | "startAge",
    value: string,
  ) {
    setStrategies((prev) =>
      prev.map((s) =>
        s.id === strategyId
          ? {
              ...s,
              stints: s.stints.map((st) =>
                st.id === stintId
                  ? {
                      ...st,
                      [field]: field === "compound" ? value : parseNum(value),
                    }
                  : st,
              ),
            }
          : s,
      ),
    );
  }

  function setUsedSet(strategyId: string, stintId: string, usedSet: boolean) {
    setStrategies((prev) =>
      prev.map((s) =>
        s.id === strategyId
          ? {
              ...s,
              stints: s.stints.map((st) =>
                st.id === stintId
                  ? { ...st, usedSet, startAge: usedSet && st.startAge === "" ? 2 : st.startAge }
                  : st,
              ),
            }
          : s,
      ),
    );
  }

  async function handleRun() {
    if (validationError) return;

    setLoading(true);
    setError(null);
    setResults(null);

    const payload: Strategy[] = strategies.map((s) => ({
      name: s.name,
      stints: s.stints.map((st) => ({
        compound: st.compound,
        laps: st.laps as number,
        // only sent for used sets; omitted = new set (backend default 1)
        ...(st.usedSet && st.startAge !== 1 ? { start_age: st.startAge as number } : {}),
      })),
    }));

    try {
      const response = await runSimulation({
        ...(track ? { track_id: track.track_id } : {}),
        // track numbers come from the backend unless overriding (or there is no track)
        ...(numbersEditable
          ? { base_laptime: baseLaptime as number, pit_loss_seconds: pitLossSeconds as number }
          : {}),
        strategies: payload,
        n_simulations: nSimulations as number,
      });
      setResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation failed");
    } finally {
      setLoading(false);
    }
  }

  const sortedResults = results
    ? [...results.results].sort((a, b) => a.mean - b.mean)
    : null;
  const referenceName = results?.reference_strategy ?? null;

  return (
    <div className="mx-auto max-w-4xl p-8 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">F1 Strategy Simulator</h1>
        <p className="text-sm text-muted-foreground">
          Compare pit strategies via Monte Carlo simulation.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Race Settings</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="space-y-1">
            <Label>Track</Label>
            {tracksError ? (
              <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                {tracksError}
                <p className="mt-1 text-xs">
                  Track selection and the optimizer are unavailable; enter the race numbers by hand below.
                </p>
              </div>
            ) : !tracks ? (
              <p className="text-sm text-muted-foreground">Loading tracks...</p>
            ) : (
              <Select
                value={trackId}
                onValueChange={(v) => {
                  if (v) selectTrack(v);
                }}
              >
                <SelectTrigger className="w-full max-w-sm" aria-label="Track">
                  <SelectValue>
                    {(v: string) => tracks.find((t) => t.track_id === v)?.name ?? "Select a track"}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {tracks.map((t) => (
                    <SelectItem key={t.track_id} value={t.track_id}>
                      {t.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
          {track && (
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">{track.total_laps} laps</Badge>
              <Badge variant="outline">pit loss {track.pit_loss_s.toFixed(1)}s</Badge>
              <Badge variant="outline">base lap {track.base_laptime_s.toFixed(2)}s</Badge>
              <Badge variant="outline">
                {["SOFT", "MEDIUM", "HARD"]
                  .map((c) => `${c[0]}=${track.compounds?.[c] ?? "?"}`)
                  .join(" ")}
              </Badge>
              <Badge
                variant="outline"
                className={track.source_season === 2026 ? "" : "border-amber-500 text-amber-700 dark:text-amber-400"}
                title={track.notes ?? undefined}
              >
                {track.source_season ?? "?"} data
                {track.source_season !== 2026 ? " (not 2026)" : ""}
              </Badge>
            </div>
          )}
          {track && (
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={override} onChange={(e) => toggleOverride(e.target.checked)} />
              Override numbers
            </label>
          )}
          <div className="flex flex-wrap gap-4">
            {numbersEditable && (
              <>
                <div className="space-y-1">
                  <Label htmlFor="base-laptime">Base lap time (s)</Label>
                  <Input
                    id="base-laptime"
                    type="number"
                    step="0.1"
                    value={baseLaptime}
                    onChange={(e) => setBaseLaptime(parseNum(e.target.value))}
                    className="w-32"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="pit-loss">Pit stop loss (s)</Label>
                  <Input
                    id="pit-loss"
                    type="number"
                    step="0.5"
                    value={pitLossSeconds}
                    onChange={(e) => setPitLossSeconds(parseNum(e.target.value))}
                    className="w-32"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="total-laps">Total laps (optimizer)</Label>
                  <Input
                    id="total-laps"
                    type="number"
                    step="1"
                    value={totalLaps}
                    onChange={(e) => setTotalLaps(parseNum(e.target.value))}
                    className="w-32"
                  />
                </div>
              </>
            )}
            <div className="space-y-1">
              <Label htmlFor="n-sims">Simulations per strategy</Label>
              <Input
                id="n-sims"
                type="number"
                step="500"
                value={nSimulations}
                onChange={(e) => setNSimulations(parseNum(e.target.value))}
                className="w-32"
              />
            </div>
          </div>
          {track && !override && (
            <p className="text-xs text-muted-foreground">
              The track&apos;s base lap time and pit loss are used for the comparison below. Your strategies define
              their own lap counts; they should add up to {track.total_laps} to match this race.
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            Base lap time shifts every lap of every strategy by the same amount, so it
            won&apos;t change which strategy wins if they run the same total lap count
            (it will if their lap counts differ). The charts below auto-scale their
            axes, so a base lap time change can look invisible there -- check the
            Mean/p10/p50/p90 columns in the results table for the actual numbers.
          </p>
          <div className="flex flex-wrap items-center gap-3 border-t pt-3">
            <div className="flex items-center gap-2">
              <Label>Max stops</Label>
              <Select value={String(maxStops)} onValueChange={(v) => v && setMaxStops(Number(v))}>
                <SelectTrigger className="w-20" aria-label="Max stops">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1</SelectItem>
                  <SelectItem value="2">2</SelectItem>
                  <SelectItem value="3">3</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button onClick={handleOptimize} disabled={optLoading || !!optValidationError}>
              {optLoading && (
                <span
                  className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"
                  aria-hidden="true"
                />
              )}
              {optLoading ? "Searching..." : "Find optimal strategy"}
            </Button>
            {optValidationError && !optLoading && (
              <span className="text-xs text-muted-foreground">{optValidationError}</span>
            )}
          </div>
        </CardContent>
      </Card>

      {optError && (
        <div className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          <p className="font-medium">Optimizer failed</p>
          <p className="mt-1">{optError}</p>
        </div>
      )}
      {optResult && <OptimizerResults result={optResult} />}

      <div className="space-y-4">
        {strategies.map((strategy) => (
          <Card key={strategy.id}>
            <CardHeader className="flex flex-row items-center justify-between gap-2">
              <Input
                value={strategy.name}
                onChange={(e) => updateStrategyName(strategy.id, e.target.value)}
                className="max-w-xs font-medium"
              />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => removeStrategy(strategy.id)}
                disabled={strategies.length <= 1}
              >
                Remove strategy
              </Button>
            </CardHeader>
            <CardContent className="space-y-2">
              {strategy.stints.map((stint) => (
                <div key={stint.id} className="flex flex-wrap items-center gap-2">
                  <Badge className={`${COMPOUND_COLOR[stint.compound]} text-white w-16 justify-center`}>
                    {stint.compound[0]}
                  </Badge>
                  <Select
                    value={stint.compound}
                    onValueChange={(value) => {
                      if (value) updateStint(strategy.id, stint.id, "compound", value);
                    }}
                  >
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="SOFT">SOFT</SelectItem>
                      <SelectItem value="MEDIUM">MEDIUM</SelectItem>
                      <SelectItem value="HARD">HARD</SelectItem>
                    </SelectContent>
                  </Select>
                  <Input
                    type="number"
                    min={1}
                    max={MAX_RACE_LAPS}
                    aria-label="Stint laps"
                    value={stint.laps}
                    onChange={(e) =>
                      updateStint(strategy.id, stint.id, "laps", e.target.value)
                    }
                    className="w-24"
                  />
                  <span className="text-sm text-muted-foreground">laps</span>
                  <label className="flex items-center gap-1 text-sm text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={stint.usedSet}
                      onChange={(e) => setUsedSet(strategy.id, stint.id, e.target.checked)}
                    />
                    used set
                  </label>
                  {stint.usedSet && (
                    <>
                      <Input
                        type="number"
                        min={1}
                        max={MAX_START_AGE}
                        step={1}
                        aria-label="Starting tyre age (laps)"
                        title="Tyre age on the stint's first lap. A set already run N laps = N + 1."
                        value={stint.startAge}
                        onChange={(e) =>
                          updateStint(strategy.id, stint.id, "startAge", e.target.value)
                        }
                        className="w-20"
                      />
                      <span className="text-sm text-muted-foreground">start age</span>
                    </>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeStint(strategy.id, stint.id)}
                    disabled={strategy.stints.length <= 1}
                  >
                    ×
                  </Button>
                </div>
              ))}
              <Button variant="outline" size="sm" onClick={() => addStint(strategy.id)}>
                + Add stint
              </Button>
            </CardContent>
          </Card>
        ))}
        <Button
          variant="outline"
          onClick={addStrategy}
          disabled={strategies.length >= MAX_STRATEGIES}
        >
          + Add strategy
        </Button>
      </div>

      <div className="space-y-2">
        <Button onClick={handleRun} disabled={loading || !!validationError} size="lg">
          {loading && (
            <span
              className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent"
              aria-hidden="true"
            />
          )}
          {loading ? "Running simulation..." : "Run simulation"}
        </Button>
        {validationError && !loading && (
          <p className="text-sm text-muted-foreground">{validationError}</p>
        )}
      </div>

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          <p className="font-medium">Simulation failed</p>
          <p className="mt-1">{error}</p>
          <p className="mt-1 text-red-700 dark:text-red-400">
            Is the backend running? (uvicorn app.main:app --reload from backend/)
          </p>
        </div>
      )}

      {sortedResults && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Mean race time by strategy</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={sortedResults.map((r) => ({
                    name: r.name,
                    mean: r.mean,
                    // Recharts ErrorBar wants [distance below, distance above]
                    p10p90: [r.mean - r.p10, r.p90 - r.mean],
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis
                    domain={["dataMin - 20", "dataMax + 20"]}
                    tickFormatter={(v) => formatTime(v)}
                  />
                  <Tooltip formatter={(v) => formatTime(Number(v))} />
                  <Bar dataKey="mean">
                    {sortedResults.map((_, i) => (
                      <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                    ))}
                    <ErrorBar dataKey="p10p90" width={6} strokeWidth={2} stroke="#00000099" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="mt-2 text-xs text-muted-foreground">
                Error bars show the p10-p90 range from the Monte Carlo runs.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Lap time by lap</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={buildLapChartData(sortedResults)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="lap" label={{ value: "Lap", position: "insideBottom", offset: -5 }} />
                  <YAxis tickFormatter={(v) => formatTime(v)} domain={["dataMin - 1", "dataMax + 1"]} />
                  <Tooltip formatter={(v) => formatTime(Number(v))} labelFormatter={(l) => `Lap ${l}`} />
                  <Legend />
                  {sortedResults.map((r, i) => {
                    const color = CHART_COLORS[i % CHART_COLORS.length];
                    return (
                      <Line
                        key={r.name}
                        type="monotone"
                        dataKey={r.name}
                        stroke={color}
                        strokeWidth={2}
                        dot={makePitStopDot(r.pit_laps, color)}
                        connectNulls={false}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
              <p className="mt-2 text-xs text-muted-foreground">
                Dots mark pit stops. One noise-free run per strategy, for shape only --
                use the table below for the statistically meaningful comparison.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Results</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Strategy</TableHead>
                    <TableHead title={DELTA_HELP}>Time vs reference</TableHead>
                    <TableHead title={DELTA_HELP}>Gap p10 to p90</TableHead>
                    <TableHead>Mean</TableHead>
                    <TableHead>p10</TableHead>
                    <TableHead>p50</TableHead>
                    <TableHead>p90</TableHead>
                    <TableHead title={WIN_PROB_HELP}>Win probability*</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedResults.map((r, i) => (
                    <TableRow key={r.name}>
                      <TableCell className="font-medium">
                        {i === 0 && "🏆 "}
                        {r.name}
                        {r.name === referenceName && (
                          <Badge variant="outline" className="ml-2">
                            reference
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="font-semibold">
                        {r.name === referenceName
                          ? "reference"
                          : `${formatDelta(r.delta_mean)} vs ${referenceName}`}
                      </TableCell>
                      <TableCell>
                        {r.name === referenceName ? "--" : formatDeltaRange(r.delta_p10, r.delta_p90)}
                      </TableCell>
                      <TableCell>{formatTime(r.mean)}</TableCell>
                      <TableCell>{formatTime(r.p10)}</TableCell>
                      <TableCell>{formatTime(r.p50)}</TableCell>
                      <TableCell>{formatTime(r.p90)}</TableCell>
                      <TableCell>{(r.win_probability * 100).toFixed(1)}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <p className="mt-2 text-xs text-muted-foreground">
                Time vs reference is the paired gap: for each simulation, this
                strategy&apos;s total minus the reference strategy&apos;s total in the
                same simulation (the reference is the strategy with the lowest mean
                time). Strategies with the same stop count share their random draws,
                so their gap is a single number rather than a range. A negative p10
                means the strategy beat the reference in at least 10% of simulations.
                Gaps of about a second or less are well inside the model&apos;s error
                (replays of real races are off by tens of seconds), so treat them as
                no meaningful difference.
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                *Win probability is the share of simulations in which this strategy
                beat the other strategies compared, under the current model (one car
                in isolation; no safety cars or traffic yet). It is not a race-win
                chance. All strategies share the same random draws, so strategies with
                the same stop count can show exactly 0% / 100% even when the time gap
                is a fraction of a second -- read the time-vs-reference gap first.
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
