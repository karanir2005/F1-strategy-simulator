export type Compound = "SOFT" | "MEDIUM" | "HARD";

export interface Stint {
  compound: Compound;
  /** Integer, 1..MAX_RACE_LAPS (backend `StintIn.laps`). */
  laps: number;
  /**
   * Tyre age (laps) on the stint's first lap; integer 1..MAX_START_AGE, backend default 1 (new set).
   * A set already run N laps -> N + 1. Omit (don't send null) for a new set.
   */
  start_age?: number;
}

// Request bounds mirrored from backend/app/api/routes/simulation.py (T-007).
export const MAX_RACE_LAPS = 100; // per stint and per strategy total
export const MAX_START_AGE = 25;
export const MAX_STRATEGIES = 20;
export const MAX_SIMULATIONS = 50000;

export interface Strategy {
  name: string;
  stints: Stint[];
}

export interface SimulateRequest {
  /** Required by the backend only when `track_id` is not sent. */
  base_laptime?: number;
  /** Fills base_laptime / pit_loss_seconds / pit-loss noise from the track when not given explicitly. */
  track_id?: string;
  strategies: Strategy[];
  n_simulations?: number;
  lap_time_noise_std?: number;
  pit_loss_noise_std?: number;
  pit_loss_seconds?: number;
  seed?: number;
  /** T-030. Shared SC/VSC timeline per Monte Carlo run; default false. */
  safety_cars?: boolean;
}

export interface StrategyResult {
  name: string;
  mean: number;
  std: number;
  p10: number;
  p50: number;
  p90: number;
  win_probability: number;
  /** Seconds. Mean over sims of (this total - reference total, same sim). >= 0; exactly 0 for the reference. */
  delta_mean: number;
  /** Seconds. 10th percentile of the paired gap; can be negative for a non-reference strategy. 0 for the reference. */
  delta_p10: number;
  /** Seconds. 90th percentile of the paired gap; 0 for the reference. */
  delta_p90: number;
  /** Noise-free lap-by-lap trace (one deterministic run), for charting -- not a Monte Carlo statistic. */
  lap_times: number[];
  pit_laps: number[];
}

export interface SimulateResponse {
  /** Strategy all delta_* are measured against: lowest mean total; ties -> first listed. */
  reference_strategy: string;
  results: StrategyResult[];
  /** T-030. Echo of the request flag. */
  safety_cars?: boolean | null;
  assumptions?: string[] | null;
}

const FETCH_TIMEOUT_MS = 60_000;
const API_URL =process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** FastAPI 422 `detail` is a list of {loc, msg}; render as readable text, else fall back to JSON. */
function formatDetail(detail: unknown): string | null {
  if (!detail) return null;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((d) => {
      if (d && typeof d === "object" && "msg" in d) {
        const loc = Array.isArray((d as { loc?: unknown }).loc)
          ? ((d as { loc: unknown[] }).loc.filter((p) => p !== "body").join(" > "))
          : "";
        const msg = String((d as { msg: unknown }).msg).replace(/^Value error, /, "");
        return loc ? `${loc}: ${msg}` : msg;
      }
      return JSON.stringify(d);
    });
    return parts.join("; ");
  }
  return JSON.stringify(detail);
}

/** Mirrors backend `TrackOut` (extra fields allowed). */
export interface TrackOut {
  track_id: string;
  name: string;
  circuit: string | null;
  total_laps: number;
  pit_loss_s: number;
  pit_loss_noise_std_s: number | null;
  base_laptime_s: number;
  compounds: Record<string, string | null> | null;
  max_stint_laps: Record<string, number>;
  source_season: number | null;
  source_races: string[];
  on_2026_calendar: boolean | null;
  already_raced_2026: boolean | null;
  notes: string | null;
  /** T-026. p90 of observed stint lengths per compound; null when too few stints. */
  typical_stint_laps?: Record<string, number | null> | null;
  /** T-028. Per-track wear fit per compound; absent/null when the track has no fit. */
  tire_wear?: Record<
    string,
    { age_coef: number; global_age_coef: number; n_stints: number; pooled: boolean }
  > | null;
}

/** Mirrors backend `OptimizeRequest`. Null/omitted overrides = use the track's value. */
export interface OptimizeRequest {
  track_id: string;
  /** 1..3 */
  max_stops?: number;
  /** 1..20000 */
  n_sims?: number;
  /** 1..10 */
  top_n?: number;
  /** 1..100 */
  total_laps?: number | null;
  /** 0..120 */
  pit_loss_s?: number | null;
  /** >0, <=300 */
  base_laptime_s?: number | null;
  seed?: number | null;
  /** T-030. Default false. */
  safety_cars?: boolean;
  /** T-030. Use the per-track tyre-wear fit when the track has one; default false. */
  use_track_wear?: boolean;
}

export interface StintOut {
  compound: string;
  laps: number;
  /** Always 1 (new tyres). */
  start_age: number;
}

export interface StrategyOut {
  rank: number;
  n_stops: number;
  stints: StintOut[];
  /** Last lap of each stint except the final one (pit at the end of that lap). */
  pit_laps: number[];
  mean_s: number;
  p10_s: number;
  p90_s: number;
  delta_to_best_mean_s: number;
  delta_to_best_p10_s: number;
  delta_to_best_p90_s: number;
}

export interface ParamsUsed {
  total_laps: number;
  pit_loss_s: number;
  base_laptime_s: number;
  n_sims: number;
  /** T-021. Stint cap per compound actually used: observed max floored to 15 and clipped to race laps. */
  effective_max_stint_laps?: Record<string, number> | null;
  /** T-021. Minimum laps for every stint in the plan search. */
  min_stint_laps?: number | null;
  /** T-030. What was actually applied (use_track_wear is false when the track has no fit). */
  safety_cars?: boolean | null;
  use_track_wear?: boolean | null;
}

export interface OptimizeResponse {
  track: TrackOut;
  params_used: ParamsUsed;
  strategies: StrategyOut[];
  n_candidates_screened: number;
  n_candidates_simulated: number;
  assumptions: string[];
}

async function failure(response: Response, what: string): Promise<Error> {
  const body = await response.json().catch(() => null);
  const detail = formatDetail(body?.detail) ?? `HTTP ${response.status}`;
  return new Error(`${what} failed (${response.status}): ${detail}`);
}

/** fetch() rejects with a bare TypeError when the server is unreachable; give that a useful message. */
async function call(path: string, what: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  const ctl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctl.abort();
  }, FETCH_TIMEOUT_MS);
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, signal: ctl.signal });
  } catch {
    if (timedOut) throw new Error(`${what} failed: the backend did not respond within ${FETCH_TIMEOUT_MS / 1000} s.`);
    throw new Error(`${what} failed: cannot reach the backend at ${API_URL}.`);
  } finally {
    clearTimeout(timer);
  }
  if (!response.ok) throw await failure(response, what);
  return response;
}

export async function runSimulation(
  request: SimulateRequest,
): Promise<SimulateResponse> {
  const response = await call("/simulate", "Simulation request", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return response.json();
}

export async function fetchTracks(): Promise<TrackOut[]> {
  const response = await call("/tracks", "Loading tracks");
  return response.json();
}

export async function runOptimize(request: OptimizeRequest): Promise<OptimizeResponse> {
  const response = await call("/optimize", "Optimizer request", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return response.json();
}

// ---------------------------------------------------------------------------
// Race prediction (backend/app/api/routes/predict.py, T-023). Probabilities are fractions 0-1.
// ---------------------------------------------------------------------------

export interface RaceInfo {
  round: number;
  name: string;
  circuit: string;
  /** ISO date, YYYY-MM-DD */
  date: string;
  quali_date: string | null;
  /** ISO datetime (UTC) of the session start; optional, absent until the backend provides it. */
  race_start_utc?: string | null;
  quali_start_utc?: string | null;
}

export interface DriverPrediction {
  code: string;
  name: string;
  team: string;
  /** null in pre_quali mode; 0 = pit-lane start. */
  grid: number | null;
  p_win: number;
  p_podium: number;
  p_top10: number;
  exp_pos: number;
  p_dnf: number;
  /** T-029. Grid mode only: the saved pre-quali numbers for the same round; null if no saved file. */
  p_win_prequali?: number | null;
  p_podium_prequali?: number | null;
}

export interface Headline {
  winner: string;
  /** Top 3 by p_podium (not an ordered finish). */
  podium: string[];
}

export interface WalkForward {
  races: number;
  win_logloss: number;
  baseline_win_logloss: number;
  /** "grid" | "prequali" */
  mode: string;
}

export interface Evidence {
  walk_forward: WalkForward | null;
  notes: string[];
}

export interface PredictionResponse {
  race: RaceInfo;
  /** "pre_quali" | "grid" */
  mode: string;
  /** ISO 8601 UTC; for a stale response it is the saved CSV's mtime. */
  generated_at: string;
  stale: boolean;
  /** T-027. null in pre_quali mode, or for an old saved CSV that did not record it. */
  grid_source?: "qualifying_classification" | "override_file" | null;
  /** Sorted by p_win descending. */
  drivers: DriverPrediction[];
  headline: Headline;
  evidence: Evidence;
  disclaimer: string;
  /** T-035. Predicted qualifying order; null when unavailable (e.g. grid mode without a saved pre-quali snapshot). */
  qualifying?: QualifyingBlock | null;
}

export interface QualifyingDriver {
  code: string;
  p_pole: number;
  /** grid <= 2 */
  p_front_row: number;
  /** grid <= 10 */
  p_q3: number;
  /** knocked out in Q1 (slot >= 17 of 22) */
  p_q1_exit: number;
  /** mean simulated qualifying rank, 1 = pole */
  exp_grid: number;
  /** grid mode only: the grid in use (0 = pit lane), null before qualifying */
  actual_grid: number | null;
}

export interface QualifyingEvidence {
  races: number;
  pole_logloss: number;
  baseline_pole_logloss: number;
  baseline_name: string;
  spearman: number;
  baseline_spearman: number;
  q3_logloss?: number;
  baseline_q3_logloss?: number;
}

export interface QualifyingBlock {
  available: boolean;
  /** Sorted by p_pole descending. */
  drivers: QualifyingDriver[];
  evidence: QualifyingEvidence | null;
}

export interface HistoryRow {
  round: number;
  race: string;
  /** Code of the driver who started 1st. */
  pole_sitter: string | null;
  predicted_winner: string;
  predicted_winner_p: number;
  actual_winner: string;
  actual_winner_grid: number | null;
  p_actual_winner: number;
  winner_correct: boolean;
  predicted_podium: string[];
  actual_podium: string[];
  /** 0-3 */
  podium_hits: number;
}

export interface HistoryResponse {
  season: number;
  /** Always "grid": predictions made with the actual qualifying grid. */
  mode: string;
  races: HistoryRow[];
}

/** The first uncached /predict/next call runs the full prediction (several seconds). */
export async function fetchPredictionNext(): Promise<PredictionResponse> {
  const response = await call("/predict/next", "Loading the prediction");
  return response.json();
}

export async function fetchPredictionHistory(): Promise<HistoryResponse> {
  const response = await call("/predict/history", "Loading prediction history");
  return response.json();
}
