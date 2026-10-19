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
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new Error(`${what} failed: cannot reach the backend at ${API_URL}.`);
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
