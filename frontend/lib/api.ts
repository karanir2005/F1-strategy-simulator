export type Compound = "SOFT" | "MEDIUM" | "HARD";

export interface Stint {
  compound: Compound;
  laps: number;
}

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
  /** Noise-free lap-by-lap trace (one deterministic run), for charting -- not a Monte Carlo statistic. */
  lap_times: number[];
  pit_laps: number[];
}

export interface SimulateResponse {
  results: StrategyResult[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
    const detail = body?.detail
      ? JSON.stringify(body.detail)
      : `HTTP ${response.status}`;
    throw new Error(`Simulation request failed: ${detail}`);
  }

  return response.json();
}
