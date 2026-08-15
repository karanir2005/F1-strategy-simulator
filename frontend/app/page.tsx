"use client";

import { useState } from "react";
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
import {
  runSimulation,
  type Compound,
  type Strategy,
  type StrategyResult,
} from "@/lib/api";

let nextId = 0;
const newId = () => `${Date.now()}-${nextId++}`;

// monotonic, never reused even after strategies are removed -- naming
// off prev.length + 1 can collide (add a 3rd, remove the 1st, add
// again -> "strategy-3" twice), which silently drops a strategy from
// results since the backend keys them by name
let nextStrategyNumber = 3;

interface StintRow {
  id: string;
  compound: Compound;
  laps: number;
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
      { id: newId(), compound: "MEDIUM", laps: 25 },
      { id: newId(), compound: "HARD", laps: 32 },
    ],
  },
  {
    id: newId(),
    name: "2-stop",
    stints: [
      { id: newId(), compound: "SOFT", laps: 15 },
      { id: newId(), compound: "MEDIUM", laps: 20 },
      { id: newId(), compound: "HARD", laps: 22 },
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
  return (props: DotProps) => {
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

function getValidationError(
  baseLaptime: number,
  nSimulations: number,
  strategies: StrategyRow[],
): string | null {
  if (baseLaptime <= 0) return "Base lap time must be greater than 0.";
  if (nSimulations <= 0) return "Simulations per strategy must be greater than 0.";
  if (strategies.length === 0) return "Add at least one strategy.";
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
      if (!Number.isInteger(stint.laps) || stint.laps <= 0) {
        return `"${strategy.name}" has a stint with an invalid lap count -- must be a positive whole number.`;
      }
    }
  }
  return null;
}

const COMPOUND_COLOR: Record<Compound, string> = {
  SOFT: "bg-red-500",
  MEDIUM: "bg-yellow-500",
  HARD: "bg-neutral-400",
};

const CHART_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"];

// mirrors race_simulator.py's DEFAULT_PIT_LOSS_SECONDS -- kept editable
// here since it's genuinely track-dependent (e.g. Monaco's pit lane costs
// noticeably more than a short one like Qatar's)
const DEFAULT_PIT_LOSS_SECONDS = 23.5;

export default function Home() {
  const [baseLaptime, setBaseLaptime] = useState(90);
  const [nSimulations, setNSimulations] = useState(2000);
  const [pitLossSeconds, setPitLossSeconds] = useState(DEFAULT_PIT_LOSS_SECONDS);
  const [strategies, setStrategies] = useState<StrategyRow[]>(initialStrategies);
  const [results, setResults] = useState<StrategyResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validationError = getValidationError(baseLaptime, nSimulations, strategies);

  function addStrategy() {
    setStrategies((prev) => [
      ...prev,
      {
        id: newId(),
        name: `strategy-${nextStrategyNumber++}`,
        stints: [{ id: newId(), compound: "MEDIUM", laps: 20 }],
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
          ? { ...s, stints: [...s.stints, { id: newId(), compound: "MEDIUM", laps: 10 }] }
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
    field: "compound" | "laps",
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
                      [field]: field === "laps" ? Number(value) || 0 : value,
                    }
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
      stints: s.stints.map((st) => ({ compound: st.compound, laps: st.laps })),
    }));

    try {
      const response = await runSimulation({
        base_laptime: baseLaptime,
        strategies: payload,
        n_simulations: nSimulations,
        pit_loss_seconds: pitLossSeconds,
      });
      setResults(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation failed");
    } finally {
      setLoading(false);
    }
  }

  const sortedResults = results
    ? [...results].sort((a, b) => a.mean - b.mean)
    : null;

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
        <CardContent className="flex flex-col gap-2">
        <div className="flex gap-4">
          <div className="space-y-1">
            <Label htmlFor="base-laptime">Base lap time (s)</Label>
            <Input
              id="base-laptime"
              type="number"
              step="0.1"
              value={baseLaptime}
              onChange={(e) => setBaseLaptime(Number(e.target.value) || 0)}
              className="w-32"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="n-sims">Simulations per strategy</Label>
            <Input
              id="n-sims"
              type="number"
              step="500"
              value={nSimulations}
              onChange={(e) => setNSimulations(Number(e.target.value) || 0)}
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
              onChange={(e) => setPitLossSeconds(Number(e.target.value) || 0)}
              className="w-32"
            />
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Base lap time shifts every lap of every strategy by the same amount, so it
          won&apos;t change which strategy wins if they run the same total lap count
          (it will if their lap counts differ). The charts below auto-scale their
          axes, so a base lap time change can look invisible there -- check the
          Mean/p10/p50/p90 columns in the results table for the actual numbers.
        </p>
        </CardContent>
      </Card>

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
                <div key={stint.id} className="flex items-center gap-2">
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
                    value={stint.laps}
                    onChange={(e) =>
                      updateStint(strategy.id, stint.id, "laps", e.target.value)
                    }
                    className="w-24"
                  />
                  <span className="text-sm text-muted-foreground">laps</span>
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
        <Button variant="outline" onClick={addStrategy}>
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
                    <TableHead>Mean</TableHead>
                    <TableHead>p10</TableHead>
                    <TableHead>p50</TableHead>
                    <TableHead>p90</TableHead>
                    <TableHead>Win probability</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedResults.map((r, i) => (
                    <TableRow key={r.name}>
                      <TableCell className="font-medium">
                        {i === 0 && "🏆 "}
                        {r.name}
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
                Win probability is relative to the strategies compared here, not a
                probability of winning the actual race against a full grid of
                opponents.
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
