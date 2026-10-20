import { MAX_RACE_LAPS, MAX_SIMULATIONS, MAX_START_AGE, MAX_STRATEGIES, type Compound } from "@/lib/api";

/** A numeric text field: "" while the user has cleared it (validated before run). */
export type NumField = number | "";

export interface StintRow {
  id: string;
  compound: Compound;
  laps: NumField;
  /** Used-set flag, kept separate from startAge so typing "15" can't collapse the input. */
  usedSet: boolean;
  /** Tyre age on the stint's first lap; only sent when usedSet. 1 = new set. */
  startAge: NumField;
}

export interface StrategyRow {
  id: string;
  name: string;
  stints: StintRow[];
}

let nextId = 0;
export const newId = () => `${Date.now()}-${nextId++}`;

// monotonic, never reused even after strategies are removed -- naming
// off prev.length + 1 can collide (add a 3rd, remove the 1st, add
// again -> "strategy-3" twice), which silently drops a strategy from
// results since the backend keys them by name
let nextStrategyNumber = 3;
export const nextStrategyName = () => `strategy-${nextStrategyNumber++}`;

export function initialStrategies(): StrategyRow[] {
  return [
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
}

export function parseNum(value: string): NumField {
  if (value === "") return "";
  const n = Number(value);
  return Number.isNaN(n) ? "" : n;
}

/** Checks for the editable race numbers (shown when overriding a track, or when no track is available). */
export function getNumbersError(
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

export function getValidationError(
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

// mirrors race_simulator.py's DEFAULT_PIT_LOSS_SECONDS -- kept editable
// since it's genuinely track-dependent
export const DEFAULT_PIT_LOSS_SECONDS = 23.5;

/** Seconds below which a gap between plans is smaller than the model can resolve (T-017 note). */
export const NOISE_FLOOR_S = 5;
