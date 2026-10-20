"use client";

import * as m from "motion/react-m";
import { useMemo, useState } from "react";
import type { Evidence, HistoryResponse, HistoryRow } from "@/lib/api";
import { fmtPct, shortRace } from "@/lib/format";
import { Button } from "@/components/ui/button";
import { EmptyBlock, ErrorBlock, Sk } from "@/components/viz/states";

/** Log-loss dumbbell: model vs baseline on a 0-based axis so the size of the gap is not exaggerated. */
export function EvidencePanel({ evidence }: { evidence: Evidence }) {
  const wf = evidence.walk_forward;
  if (!wf) {
    return (
      <div className="panel p-5">
        <h3 className="text-base font-semibold text-fg">How much to trust this</h3>
        <p className="mt-2 text-sm text-muted-fg">
          No walk-forward evaluation is available for this prediction, so there is no measured track record to show.
        </p>
      </div>
    );
  }
  const hi = Math.ceil(Math.max(wf.win_logloss, wf.baseline_win_logloss) * 1.3 * 100) / 100;
  const x = (v: number) => `${(v / hi) * 100}%`;
  const rel = ((wf.baseline_win_logloss - wf.win_logloss) / wf.baseline_win_logloss) * 100;
  const better = rel > 0;
  const modeLabel = wf.mode === "grid" ? "with the qualifying grid" : "before qualifying";

  return (
    <div className="panel p-5">
      <h3 className="text-base font-semibold text-fg">How much to trust this</h3>
      <p className="mt-1 text-[13px] leading-5 text-muted-fg">
        Walk-forward test over {wf.races} 2026 races, predicting each one {modeLabel} using only earlier races. Win
        log-loss: lower is better.
      </p>

      <div className="mt-5">
        <div className="relative h-10">
          <span aria-hidden className="absolute top-1/2 right-0 left-0 h-px bg-line-strong" />
          <span
            aria-hidden
            className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-fg/50"
            style={{
              left: x(Math.min(wf.win_logloss, wf.baseline_win_logloss)),
              width: `${(Math.abs(wf.win_logloss - wf.baseline_win_logloss) / hi) * 100}%`,
            }}
          />
          <span
            aria-hidden
            className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-muted-fg bg-raised"
            style={{ left: x(wf.baseline_win_logloss) }}
          />
          <span
            aria-hidden
            className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-signal ring-2 ring-raised"
            style={{ left: x(wf.win_logloss) }}
          />
        </div>
        <div className="flex justify-between font-mono text-[11px] text-faint tnum">
          <span>0</span>
          <span>{hi.toFixed(2)}</span>
        </div>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-4 text-sm">
        <div>
          <dt className="flex items-center gap-2 text-muted-fg">
            <span aria-hidden className="size-2.5 rounded-full bg-signal" />
            Model
          </dt>
          <dd className="font-mono text-xl text-fg tnum">{wf.win_logloss.toFixed(4)}</dd>
        </div>
        <div>
          <dt className="flex items-center gap-2 text-muted-fg">
            <span aria-hidden className="size-2.5 rounded-full border-2 border-muted-fg" />
            Baseline
          </dt>
          <dd className="font-mono text-xl text-fg tnum">{wf.baseline_win_logloss.toFixed(4)}</dd>
        </div>
      </dl>

      <p className="mt-3 text-sm leading-6 text-fg">
        {better
          ? `The model's log-loss is ${Math.abs(rel).toFixed(1)}% lower than the baseline. `
          : `The model's log-loss is ${Math.abs(rel).toFixed(1)}% higher than the baseline, so it is not beating it. `}
        With only {wf.races} races, treat this as a modest signal, not proof of skill.
      </p>
      {evidence.notes.length > 0 && (
        <ul className="mt-2 list-disc space-y-1 pl-5 text-[13px] leading-5 text-muted-fg">
          {evidence.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Pips({ hits }: { hits: number }) {
  return (
    <span className="inline-flex items-center gap-1" role="img" aria-label={`${hits} of 3 podium places right`}>
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          aria-hidden
          className={`size-2.5 rounded-sm border ${i < hits ? "border-good bg-good" : "border-line-strong"}`}
        />
      ))}
    </span>
  );
}

const ROW_COLS = "md:grid-cols-[2.5rem_minmax(0,1.5fr)_3.5rem_minmax(0,1.3fr)_minmax(0,1.6fr)_4.5rem]";

function Row({ r, i }: { r: HistoryRow; i: number }) {
  return (
    <m.li
      className={`border-b border-line px-3 py-3 last:border-b-0 md:grid md:items-center md:gap-x-3 ${ROW_COLS}`}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: Math.min(i, 8) * 0.02 }}
    >
      <div className="flex items-center gap-3 md:contents">
        <span className="w-8 font-mono text-[13px] text-faint tnum md:w-auto">R{r.round}</span>
        <span className="min-w-0 flex-1 truncate text-[15px] text-fg md:flex-none">{shortRace(r.race)}</span>
        <span className="md:hidden">
          <Pips hits={r.podium_hits} />
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 pl-11 text-[13px] md:mt-0 md:contents md:pl-0">
        <span className="text-muted-fg md:font-mono">
          <span className="label-mono mr-1.5 md:hidden">Pole</span>
          {r.pole_sitter ?? "n/a"}
        </span>
        <span className="text-muted-fg">
          <span className="label-mono mr-1.5 md:hidden">Called</span>
          <span className="font-mono text-fg">{r.predicted_winner}</span> <span className="tnum">{fmtPct(r.predicted_winner_p)}</span>
        </span>
        <span className="flex items-center gap-2">
          <span
            className={`inline-flex h-5 items-center rounded-sm px-1.5 font-mono text-[11px] uppercase ${
              r.winner_correct ? "bg-good/15 text-good" : "border border-line-strong text-muted-fg"
            }`}
          >
            {r.winner_correct ? "Hit" : "Miss"}
          </span>
          <span>
            <span className="font-mono text-fg">{r.actual_winner}</span>{" "}
            <span className="text-muted-fg tnum" title="Probability the model gave the actual winner">
              had {fmtPct(r.p_actual_winner)}
            </span>
          </span>
        </span>
      </div>
      <span className="hidden justify-self-end md:block">
        <Pips hits={r.podium_hits} />
      </span>
    </m.li>
  );
}

export function HistoryTable({ history }: { history: HistoryResponse }) {
  const [all, setAll] = useState(false);
  const rows = useMemo(() => [...history.races].sort((a, b) => b.round - a.round), [history.races]);
  const stats = useMemo(() => {
    const n = history.races.length;
    return {
      n,
      correct: history.races.filter((r) => r.winner_correct).length,
      poleWon: history.races.filter((r) => r.pole_sitter && r.pole_sitter === r.actual_winner).length,
      pickedPole: history.races.filter((r) => r.pole_sitter && r.pole_sitter === r.predicted_winner).length,
      hits: history.races.reduce((s, r) => s + r.podium_hits, 0),
    };
  }, [history.races]);
  const shown = all ? rows : rows.slice(0, 6);

  return (
    <div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
        {[
          { k: "Winner called", v: `${stats.correct}/${stats.n}` },
          { k: "Pole sitter won", v: `${stats.poleWon}/${stats.n}` },
          { k: "Pick was the pole sitter", v: `${stats.pickedPole}/${stats.n}` },
          { k: "Podium places right", v: `${stats.hits}/${stats.n * 3}` },
        ].map((s) => (
          <div key={s.k}>
            <dt className="label-mono">{s.k}</dt>
            <dd className="display text-3xl text-fg">{s.v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 max-w-2xl text-[13px] leading-5 text-muted-fg">
        Each race is predicted with its actual qualifying grid. Compare the winner hit rate with how often the pole
        sitter simply won: a model that mostly follows the grid will look good in a season where pole converts often.
        {stats.n > 0 && stats.pickedPole === stats.n &&
          " Here the model's pick was the pole sitter in every race, so its winner record is the pole sitter's record."}
      </p>

      <div className="mt-4 overflow-hidden rounded-panel border border-line">
        <div className={`hidden gap-x-3 border-b border-line bg-sunken px-3 py-2 md:grid ${ROW_COLS}`}>
          <span className="label-mono">Rd</span>
          <span className="label-mono">Race</span>
          <span className="label-mono">Pole</span>
          <span className="label-mono">Model called</span>
          <span className="label-mono">Actual winner</span>
          <span className="label-mono text-right">Podium</span>
        </div>
        <ol className="m-0 list-none p-0">
          {shown.map((r, i) => (
            <Row key={r.round} r={r} i={i} />
          ))}
        </ol>
      </div>
      {rows.length > 6 && (
        <Button variant="outline" size="sm" className="mt-3" onClick={() => setAll((v) => !v)} aria-expanded={all}>
          {all ? "Show the latest 6" : `Show all ${rows.length} races`}
        </Button>
      )}
    </div>
  );
}

export function HistoryStates({
  history,
  error,
  loading,
  onRetry,
}: {
  history: HistoryResponse | null;
  error: string | null;
  loading: boolean;
  onRetry: () => void;
}) {
  if (history && history.races.length > 0) return <HistoryTable history={history} />;
  if (error) return <ErrorBlock title="Could not load the season replay" message={error} onRetry={onRetry} />;
  if (loading)
    return (
      <div className="space-y-3" aria-busy="true">
        <Sk className="h-16 w-full max-w-xl" />
        <Sk className="h-60 w-full" />
      </div>
    );
  return <EmptyBlock title="No season replay yet">No completed 2026 races have a saved prediction to compare.</EmptyBlock>;
}
