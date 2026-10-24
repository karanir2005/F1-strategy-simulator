"use client";

import { AlertTriangle } from "lucide-react";
import {
  fetchPredictionHistory,
  fetchPredictionNext,
  type HistoryResponse,
  type PredictionResponse,
} from "@/lib/api";
import { fmtOneIn, fmtPct, fmtUpdated } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { EvidencePanel, HistoryStates } from "@/components/predict/history";
import { Hero } from "@/components/predict/hero";
import { ProbabilityTable } from "@/components/predict/probability-table";
import { RaceHeader } from "@/components/predict/race-header";
import { EmptyBlock, ErrorBlock, Sk } from "@/components/viz/states";

function PredictSkeleton() {
  return (
    <div aria-busy="true" aria-live="polite" className="space-y-10">
      <p className="sr-only">Loading the prediction</p>
      <div className="space-y-3">
        <Sk className="h-3 w-32" />
        <Sk className="h-12 w-72 max-w-full" />
        <Sk className="h-4 w-48" />
        <Sk className="h-4 w-80 max-w-full" />
      </div>
      <div className="grid gap-10 border-t border-line pt-8 lg:grid-cols-12 lg:gap-8">
        <div className="space-y-4 lg:col-span-7">
          <Sk className="h-3 w-36" />
          <Sk className="h-6 w-64 max-w-full" />
          <Sk className="h-[100px] w-48 sm:h-[126px]" />
          <Sk className="h-1.5 w-full max-w-md" />
          <Sk className="h-[180px] w-full max-w-[360px]" />
        </div>
        <div className="lg:col-span-5">
          <Sk className="h-3 w-48" />
          <Sk className="mt-3 h-[272px] w-full" />
        </div>
      </div>
      <p className="text-sm text-muted-fg">
        Simulating the race. The first load of the day can take several seconds, later loads are instant.
      </p>
      <div className="space-y-px">
        {Array.from({ length: 8 }).map((_, i) => (
          <Sk key={i} className="h-[52px] w-full" />
        ))}
      </div>
    </div>
  );
}

export default function PredictPage() {
  const next = useApi<PredictionResponse>("predict-next", fetchPredictionNext);
  const hist = useApi<HistoryResponse>("predict-history", fetchPredictionHistory);
  const data = next.data;

  if (!data) {
    if (next.error)
      return (
        <ErrorBlock
          title="Could not load the prediction"
          message={next.error}
          hint="Is the backend running? Start it with uvicorn app.main:app --reload from backend/."
          onRetry={next.reload}
        />
      );
    return <PredictSkeleton />;
  }

  return (
    <div className="space-y-14">
      {data.stale && (
        <div role="status" className="flex gap-3 rounded-panel border border-warn/50 bg-warn/10 p-4 text-sm">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden />
          <p className="text-fg">
            <span className="font-medium">Showing a saved prediction, not a live one.</span>{" "}
            <span className="text-muted-fg">
              Live data was unavailable, so this is the newest saved result from {fmtUpdated(data.generated_at)}. It may
              not reflect the latest qualifying or results.
            </span>
          </p>
        </div>
      )}

      <RaceHeader data={data} />

      {data.drivers.length === 0 ? (
        <EmptyBlock title="The prediction has no drivers">
          The backend returned an empty field. Try again in a moment.
        </EmptyBlock>
      ) : (
        <>
          <Hero data={data} />
          <ProbabilityTable drivers={data.drivers} qualifying={data.qualifying} />
        </>
      )}

      <section aria-labelledby="trust-h" className="grid gap-8 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <h2 id="trust-h" className="sr-only">
            Track record and caveats
          </h2>
          <EvidencePanel evidence={data.evidence} />
        </div>
        <div className="space-y-3 lg:col-span-7">
          <h3 className="text-base font-semibold text-fg">Probabilities, not picks</h3>
          <p className="max-w-prose text-[15px] leading-6 text-fg">
            {data.drivers[0]
              ? `A ${fmtPct(data.drivers[0].p_win)} favourite still loses ${fmtOneIn(1 - data.drivers[0].p_win)} races. `
              : ""}
            Use these numbers to compare drivers, not to expect a single outcome.
          </p>
          <p className="max-w-prose text-[13px] leading-5 text-muted-fg">{data.disclaimer}</p>
        </div>
      </section>

      <section aria-labelledby="hist-h">
        <h2 id="hist-h" className="display text-3xl text-fg">
          How it did in 2026
        </h2>
        <p className="mt-1 mb-5 text-[13px] text-muted-fg">Predicted versus actual winner, race by race.</p>
        <HistoryStates
          history={hist.data}
          error={hist.error}
          loading={hist.loading}
          onRetry={hist.reload}
        />
      </section>
    </div>
  );
}
