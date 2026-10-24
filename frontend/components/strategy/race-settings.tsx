"use client";

import { Loader2 } from "lucide-react";
import type { TrackOut } from "@/lib/api";
import { parseNum, type NumField } from "@/lib/strategy-form";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ErrorBlock, Sk } from "@/components/viz/states";

export interface RaceSettingsProps {
  tracks: TrackOut[] | null;
  tracksError: string | null;
  trackId: string;
  track: TrackOut | null;
  onSelectTrack: (id: string) => void;
  override: boolean;
  onOverride: (on: boolean) => void;
  numbersEditable: boolean;
  baseLaptime: NumField;
  onBaseLaptime: (v: NumField) => void;
  pitLossSeconds: NumField;
  onPitLoss: (v: NumField) => void;
  totalLaps: NumField;
  onTotalLaps: (v: NumField) => void;
  nSimulations: NumField;
  onNSimulations: (v: NumField) => void;
  maxStops: number;
  onMaxStops: (n: number) => void;
  safetyCars: boolean;
  onSafetyCars: (on: boolean) => void;
  trackWear: boolean;
  onTrackWear: (on: boolean) => void;
  onOptimize: () => void;
  optLoading: boolean;
  optValidationError: string | null;
}

function Meta({ k, v }: { k: string; v: string }) {
  return (
    <div className="min-w-0">
      <dt className="label-mono">{k}</dt>
      <dd className="font-mono text-[15px] text-fg tnum">{v}</dd>
    </div>
  );
}

export function RaceSettings(p: RaceSettingsProps) {
  const t = p.track;
  return (
    <section aria-labelledby="settings-h" className="panel p-4 sm:p-5">
      <h2 id="settings-h" className="sr-only">
        Race settings
      </h2>

      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div className="min-w-0 space-y-1.5">
          <Label htmlFor="track-select">Track</Label>
          {p.tracksError ? (
            <ErrorBlock
              title="Could not load the track list"
              message={p.tracksError}
              hint="Track selection and the optimizer are unavailable. Enter the race numbers by hand below to use the manual comparison."
            />
          ) : !p.tracks ? (
            <Sk className="h-9 w-full sm:w-80" />
          ) : (
            <Select
              value={p.trackId}
              onValueChange={(v) => {
                if (v) p.onSelectTrack(v);
              }}
            >
              <SelectTrigger id="track-select" className="w-full sm:w-80" aria-label="Track">
                <SelectValue>
                  {(v: string) => p.tracks?.find((x) => x.track_id === v)?.name ?? "Select a track"}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {p.tracks.map((x) => (
                  <SelectItem key={x.track_id} value={x.track_id}>
                    {x.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        {t && (
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4 lg:flex lg:gap-x-8">
            <Meta k="Laps" v={String(t.total_laps)} />
            <Meta k="Pit loss" v={`${t.pit_loss_s.toFixed(1)}s`} />
            <Meta k="Base lap" v={`${t.base_laptime_s.toFixed(2)}s`} />
            <Meta
              k="Tyres"
              v={["SOFT", "MEDIUM", "HARD"].map((c) => `${c[0]}${t.compounds?.[c] ?? "?"}`).join(" ")}
            />
          </dl>
        )}
      </div>

      {t && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Badge variant={t.source_season === 2026 ? "outline" : "warn"} title={t.notes ?? undefined}>
            {t.source_season ?? "?"} data{t.source_season !== 2026 ? ", not 2026" : ""}
          </Badge>
          {t.already_raced_2026 === false && <Badge variant="outline">Not raced yet in 2026</Badge>}
          {t.notes && t.source_season !== 2026 && (
            <p className="basis-full text-[13px] leading-5 text-muted-fg">{t.notes}</p>
          )}
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-end gap-x-5 gap-y-3 border-t border-line pt-4">
        {t && (
          <label className="flex min-h-9 cursor-pointer items-center gap-2 text-sm text-fg">
            <input
              type="checkbox"
              className="size-4 accent-signal"
              checked={p.override}
              onChange={(e) => p.onOverride(e.target.checked)}
            />
            Override numbers
          </label>
        )}
        {p.numbersEditable && (
          <>
            <div className="space-y-1.5">
              <Label htmlFor="base-laptime">Base lap time (s)</Label>
              <Input
                id="base-laptime"
                type="number"
                step="0.1"
                value={p.baseLaptime}
                onChange={(e) => p.onBaseLaptime(parseNum(e.target.value))}
                className="w-28"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="pit-loss">Pit stop loss (s)</Label>
              <Input
                id="pit-loss"
                type="number"
                step="0.5"
                value={p.pitLossSeconds}
                onChange={(e) => p.onPitLoss(parseNum(e.target.value))}
                className="w-28"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="total-laps">Total laps (optimizer)</Label>
              <Input
                id="total-laps"
                type="number"
                step="1"
                value={p.totalLaps}
                onChange={(e) => p.onTotalLaps(parseNum(e.target.value))}
                className="w-28"
              />
            </div>
          </>
        )}
        <div className="space-y-1.5">
          <Label htmlFor="n-sims">Simulations per strategy</Label>
          <Input
            id="n-sims"
            type="number"
            step="500"
            value={p.nSimulations}
            onChange={(e) => p.onNSimulations(parseNum(e.target.value))}
            className="w-32"
          />
        </div>
      </div>

      {t && !p.override && (
        <p className="mt-3 text-[13px] leading-5 text-muted-fg">
          The track&apos;s base lap time and pit loss are used for the comparison below. Your strategies define their own
          lap counts; they should add up to {t.total_laps} to match this race.
        </p>
      )}
      <p className="mt-2 text-[13px] leading-5 text-muted-fg">
        Base lap time shifts every lap of every strategy by the same amount, so it won&apos;t change which strategy wins
        when they run the same total lap count.
      </p>

      <fieldset className="mt-4 border-t border-line pt-4">
        <legend className="sr-only">Model options</legend>
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          <label className="flex min-h-9 cursor-pointer items-center gap-2 text-sm text-fg">
            <input
              type="checkbox"
              className="size-4 accent-signal"
              checked={p.safetyCars}
              onChange={(e) => p.onSafetyCars(e.target.checked)}
            />
            Safety cars
          </label>
          {t && (
            <label
              className={`flex min-h-9 items-center gap-2 text-sm ${t.tire_wear ? "cursor-pointer text-fg" : "text-muted-fg"}`}
              title={t.tire_wear ? undefined : "No 2026 race data at this circuit yet, so the global wear fit is used"}
            >
              <input
                type="checkbox"
                className="size-4 accent-signal"
                disabled={!t.tire_wear}
                checked={p.trackWear && !!t.tire_wear}
                onChange={(e) => p.onTrackWear(e.target.checked)}
              />
              Track-specific tyre wear{!t.tire_wear && <span className="text-[12px]">(not available here)</span>}
            </label>
          )}
        </div>
        <p className="mt-1 text-[13px] leading-5 text-muted-fg">
          Safety cars add random SC/VSC periods, the same for every plan in a run, so gaps stay comparable. Pit plans are
          fixed, so they can&apos;t react by pitting under a safety car yet. Track wear uses this circuit&apos;s own
          tyre fit where there is enough data (optimizer only); it hasn&apos;t been shown to beat the global fit.
        </p>
      </fieldset>

      <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-line pt-4">
        <div className="flex items-center gap-2">
          <Label htmlFor="max-stops">Max stops</Label>
          <Select value={String(p.maxStops)} onValueChange={(v) => v && p.onMaxStops(Number(v))}>
            <SelectTrigger id="max-stops" className="w-20" aria-label="Max stops">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1">1</SelectItem>
              <SelectItem value="2">2</SelectItem>
              <SelectItem value="3">3</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Button onClick={p.onOptimize} disabled={p.optLoading || !!p.optValidationError}>
          {p.optLoading && <Loader2 className="size-4 animate-spin motion-reduce:animate-none" aria-hidden />}
          {p.optLoading ? "Searching..." : "Find optimal strategy"}
        </Button>
        {p.optValidationError && !p.optLoading && (
          <span className="text-[13px] text-muted-fg">{p.optValidationError}</span>
        )}
      </div>
    </section>
  );
}
