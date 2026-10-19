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
  base_laptime: number;
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

export async function runSimulation(
  request: SimulateRequest,
): Promise<SimulateResponse> {
  const response = await fetch(`${API_URL}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = formatDetail(body?.detail) ?? `HTTP ${response.status}`;
    throw new Error(`Simulation request failed: ${detail}`);
  }

  return response.json();
}
