import { StintStrip, type StripStint } from "@/components/viz/stint-strip";

/** Stint bar coloured by compound (S/M/H) with the pit laps marked underneath at their position in the race. */
export function PlanTimeline({
  stints,
  pitLaps,
  size = "md",
  playKey,
  labels,
}: {
  stints: StripStint[];
  /** last lap of each stint except the final one */
  pitLaps: number[];
  size?: "sm" | "md" | "lg";
  playKey?: string;
  labels?: Record<string, string | null | undefined>;
}) {
  const total = stints.reduce((s, x) => s + x.laps, 0);
  return (
    <div className="w-full">
      <StintStrip stints={stints} size={size} playKey={playKey} labels={labels} />
      {pitLaps.length > 0 && total > 0 && (
        <div className="relative mt-1 h-4" aria-hidden>
          {pitLaps.map((lap) => (
            <span
              key={lap}
              className="absolute top-0 flex -translate-x-1/2 flex-col items-center font-mono text-[10px] leading-none text-muted-fg tnum"
              style={{ left: `${(lap / total) * 100}%` }}
            >
              <span className="h-1.5 w-px bg-line-strong" />
              <span className="mt-0.5">L{lap}</span>
            </span>
          ))}
        </div>
      )}
      {pitLaps.length > 0 && (
        <p className="sr-only">Pit stops at the end of lap {pitLaps.join(", ")}.</p>
      )}
    </div>
  );
}
