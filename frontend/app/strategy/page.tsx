"use client";

import { useEffect, useRef, useState } from "react";
import {
  fetchTracks,
  runOptimize,
  runSimulation,
  type OptimizeResponse,
  type SimulateResponse,
  type Strategy,
  type TrackOut,
} from "@/lib/api";
import {
  DEFAULT_PIT_LOSS_SECONDS,
  getNumbersError,
  getValidationError,
  initialStrategies,
  newId,
  nextStrategyName,
  parseNum,
  type NumField,
  type StrategyRow,
} from "@/lib/strategy-form";
import { CompareResults } from "@/components/strategy/compare-results";
import { ManualCompare } from "@/components/strategy/manual-compare";
import { OptimizerResults } from "@/components/strategy/optimizer-results";
import { RaceSettings } from "@/components/strategy/race-settings";
import { ErrorBlock, Sk } from "@/components/viz/states";

function ResultsSkeleton({ label }: { label: string }) {
  return (
    <div aria-busy="true" className="space-y-4">
      <p className="sr-only">{label}</p>
      <Sk className="h-[170px] w-full rounded-panel" />
      <Sk className="h-[280px] w-full rounded-panel" />
    </div>
  );
}

export default function StrategyPage() {
  const [baseLaptime, setBaseLaptime] = useState<NumField>(90);
  const [nSimulations, setNSimulations] = useState<NumField>(2000);
  const [pitLossSeconds, setPitLossSeconds] = useState<NumField>(DEFAULT_PIT_LOSS_SECONDS);
  const [totalLaps, setTotalLaps] = useState<NumField>(57);
  const [tracks, setTracks] = useState<TrackOut[] | null>(null);
  const [tracksError, setTracksError] = useState<string | null>(null);
  const [trackId, setTrackId] = useState<string>("");
  const [override, setOverride] = useState(false);
  const [maxStops, setMaxStops] = useState(3);
  const [safetyCars, setSafetyCars] = useState(false);
  const [trackWear, setTrackWear] = useState(false);
  const [optResult, setOptResult] = useState<OptimizeResponse | null>(null);
  const [optLoading, setOptLoading] = useState(false);
  const [optError, setOptError] = useState<string | null>(null);
  const [strategies, setStrategies] = useState<StrategyRow[]>(initialStrategies);
  const [results, setResults] = useState<SimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const epoch = useRef(0);
  const [resultsCtx, setResultsCtx] = useState<string | null>(null);

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
    numbersEditable ? null : track?.total_laps,
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
        if (first) {
          setTrackId(first.track_id);
          setStrategies((rows) => fitToRaceLaps(rows, first.total_laps));
        }
      })
      .catch((err) => {
        if (!cancelled) setTracksError(err instanceof Error ? err.message : "Could not load tracks");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // keep the manual plans at the selected track's race distance: absorb any difference in the last stint
  function fitToRaceLaps(rows: StrategyRow[], laps: number): StrategyRow[] {
    return rows.map((s) => {
      const sum = s.stints.reduce((a, st) => a + (typeof st.laps === "number" ? st.laps : 0), 0);
      const last = s.stints[s.stints.length - 1];
      if (!last || sum === laps || typeof last.laps !== "number") return s;
      const adjusted = last.laps + (laps - sum);
      if (adjusted < 1) return s;
      return { ...s, stints: [...s.stints.slice(0, -1), { ...last, laps: adjusted }] };
    });
  }

  function prefillFrom(t: TrackOut) {
    setBaseLaptime(t.base_laptime_s);
    setPitLossSeconds(t.pit_loss_s);
    setTotalLaps(t.total_laps);
  }

  function selectTrack(id: string) {
    setTrackId(id);
    epoch.current += 1; // any in-flight request now belongs to a previous track
    setOptResult(null);
    setOptError(null);
    setOptLoading(false);
    setResults(null);
    setResultsCtx(null);
    setError(null);
    setLoading(false);
    const t = tracks?.find((x) => x.track_id === id);
    if (t && override) prefillFrom(t);
    if (t) setStrategies((rows) => fitToRaceLaps(rows, t.total_laps));
  }

  function toggleOverride(on: boolean) {
    setOverride(on);
    if (on && track) prefillFrom(track);
  }

  async function handleOptimize() {
    if (!track || optValidationError) return;
    const myEpoch = epoch.current;
    setOptLoading(true);
    setOptError(null);
    setOptResult(null);
    try {
      const res = await runOptimize({
          track_id: track.track_id,
          max_stops: maxStops,
          safety_cars: safetyCars,
          use_track_wear: trackWear,
          top_n: 10, // ask for extra so client-side dedupe still leaves 5 distinct plans
          ...(override
            ? {
                total_laps: totalLaps as number,
                pit_loss_s: pitLossSeconds as number,
                base_laptime_s: baseLaptime as number,
              }
            : {}),
      });
      if (myEpoch === epoch.current) setOptResult(res);
    } catch (err) {
      if (myEpoch === epoch.current) setOptError(err instanceof Error ? err.message : "Optimizer failed");
    } finally {
      if (myEpoch === epoch.current) setOptLoading(false);
    }
  }

  const mapStrategy = (id: string, f: (s: StrategyRow) => StrategyRow) =>
    setStrategies((prev) => prev.map((s) => (s.id === id ? f(s) : s)));

  function addStrategy() {
    setStrategies((prev) => [
      ...prev,
      {
        id: newId(),
        name: nextStrategyName(),
        stints: [{ id: newId(), compound: "MEDIUM", laps: 20, usedSet: false, startAge: 2 }],
      },
    ]);
  }

  function addStint(strategyId: string) {
    mapStrategy(strategyId, (s) => ({
      ...s,
      stints: [...s.stints, { id: newId(), compound: "MEDIUM", laps: 10, usedSet: false, startAge: 2 }],
    }));
  }

  function updateStint(strategyId: string, stintId: string, field: "compound" | "laps" | "startAge", value: string) {
    mapStrategy(strategyId, (s) => ({
      ...s,
      stints: s.stints.map((st) =>
        st.id === stintId ? { ...st, [field]: field === "compound" ? value : parseNum(value) } : st,
      ),
    }));
  }

  function setUsedSet(strategyId: string, stintId: string, usedSet: boolean) {
    mapStrategy(strategyId, (s) => ({
      ...s,
      stints: s.stints.map((st) =>
        st.id === stintId ? { ...st, usedSet, startAge: usedSet && st.startAge === "" ? 2 : st.startAge } : st,
      ),
    }));
  }

  async function handleRun() {
    if (validationError) return;

    const myEpoch = epoch.current;
    setLoading(true);
    setError(null);
    setResults(null);
    setResultsCtx(null);

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
      const res = await runSimulation({
          ...(track ? { track_id: track.track_id } : {}),
          // track numbers come from the backend unless overriding (or there is no track)
          ...(numbersEditable
            ? { base_laptime: baseLaptime as number, pit_loss_seconds: pitLossSeconds as number }
            : {}),
          strategies: payload,
          n_simulations: nSimulations as number,
          safety_cars: safetyCars,
      });
      if (myEpoch === epoch.current) {
        setResults(res);
        setResultsCtx(
          `${track ? track.name : "Custom numbers"} · ${strategies.reduce((m, s) => Math.max(m, s.stints.reduce((a, st) => a + (typeof st.laps === "number" ? st.laps : 0), 0)), 0)} laps · safety cars ${safetyCars ? "on" : "off"}`,
        );
      }
    } catch (err) {
      if (myEpoch === epoch.current) setError(err instanceof Error ? err.message : "Simulation failed");
    } finally {
      if (myEpoch === epoch.current) setLoading(false);
    }
  }

  return (
    <div className="space-y-12">
      <header>
        <p className="label-mono">Monte Carlo</p>
        <h1 className="display mt-1 text-[34px] leading-[1.02] text-fg sm:text-5xl">Pit strategy</h1>
        <p className="mt-2 max-w-2xl text-[15px] text-muted-fg">
          Search for the fastest stint plan at a circuit, then test your own plans against it. One car, clean air:
          the gaps between plans are useful, the absolute times are not a forecast.
        </p>
      </header>

      <RaceSettings
        tracks={tracks}
        tracksError={tracksError}
        trackId={trackId}
        track={track}
        onSelectTrack={selectTrack}
        override={override}
        onOverride={toggleOverride}
        numbersEditable={numbersEditable}
        baseLaptime={baseLaptime}
        onBaseLaptime={setBaseLaptime}
        pitLossSeconds={pitLossSeconds}
        onPitLoss={setPitLossSeconds}
        totalLaps={totalLaps}
        onTotalLaps={setTotalLaps}
        nSimulations={nSimulations}
        onNSimulations={setNSimulations}
        maxStops={maxStops}
        onMaxStops={setMaxStops}
        safetyCars={safetyCars}
        onSafetyCars={setSafetyCars}
        trackWear={trackWear}
        onTrackWear={setTrackWear}
        onOptimize={handleOptimize}
        optLoading={optLoading}
        optValidationError={optValidationError}
      />

      {optLoading && <ResultsSkeleton label="Searching for the best plan" />}
      {optError && <ErrorBlock title="Optimizer failed" message={optError} onRetry={handleOptimize} />}
      {optResult && <OptimizerResults result={optResult} />}

      <ManualCompare
        strategies={strategies}
        onAddStrategy={addStrategy}
        onRemoveStrategy={(id) => setStrategies((prev) => prev.filter((s) => s.id !== id))}
        onRename={(id, name) => mapStrategy(id, (s) => ({ ...s, name }))}
        onAddStint={addStint}
        onRemoveStint={(sid, stintId) =>
          mapStrategy(sid, (s) => ({ ...s, stints: s.stints.filter((st) => st.id !== stintId) }))
        }
        onUpdateStint={updateStint}
        onUsedSet={setUsedSet}
        onRun={handleRun}
        loading={loading}
        validationError={validationError}
      />

      {loading && <ResultsSkeleton label="Running the simulation" />}
      {error && (
        <ErrorBlock
          title="Simulation failed"
          message={error}
          hint={
            error.includes("cannot reach") || error.includes("did not respond")
              ? "Is the backend running? Start it with uvicorn app.main:app --reload from backend/."
              : undefined
          }
        />
      )}
      {results && <CompareResults results={results} context={resultsCtx} />}
    </div>
  );
}
