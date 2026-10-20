import type { Compound } from "@/lib/api";

/** Whole % from 10% up, one decimal below, "<1%" instead of a misleading 0%. */
export function fmtPct(p: number): string {
  if (p <= 0) return "0%";
  if (p < 0.01) return "<1%";
  const v = p * 100;
  return v >= 10 ? `${Math.round(v)}%` : `${v.toFixed(1)}%`;
}

/** Plain frequency phrasing for a probability: "about 1 in 4", "about 7 in 10". */
export function fmtOneIn(p: number): string {
  if (p <= 0) return "almost never";
  if (p >= 0.5) return `about ${Math.round(p * 10)} in 10`;
  return `about 1 in ${Math.max(2, Math.round(1 / p))}`;
}

export function fmtTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

export function fmtSigned(seconds: number, digits = 2): string {
  return `${seconds >= 0 ? "+" : "-"}${Math.abs(seconds).toFixed(digits)}s`;
}

const dateFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });

/** "2026-10-25" -> "25 Oct" (UTC, so the viewer's timezone never shifts the day). */
export function fmtDay(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  return Number.isNaN(d.getTime()) ? iso : dateFmt.format(d);
}

/** Whole calendar days from today (local calendar date) to an ISO date; negative if past. */
export function daysUntil(iso: string, now: Date = new Date()): number {
  const target = Date.parse(`${iso}T00:00:00Z`);
  const today = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  return Math.round((target - today) / 86_400_000);
}

export function fmtUpdated(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

/** "Australian Grand Prix" -> "Australian GP". */
export function shortRace(name: string): string {
  return name.replace(/ Grand Prix/g, " GP");
}

export const COMPOUND_VAR: Record<string, string> = {
  SOFT: "var(--tyre-soft)",
  MEDIUM: "var(--tyre-medium)",
  HARD: "var(--tyre-hard)",
};

export const COMPOUNDS: Compound[] = ["SOFT", "MEDIUM", "HARD"];

/** Total race seconds as h:mm:ss.s (e.g. 5597.2 -> "1:33:17.2"). */
export function fmtRaceTime(totalSeconds: number): string {
  const h = Math.floor(totalSeconds / 3600);
  const mm = Math.floor((totalSeconds - h * 3600) / 60);
  const s = totalSeconds - h * 3600 - mm * 60;
  const ss = s.toFixed(1).padStart(4, "0");
  return h > 0 ? `${h}:${String(mm).padStart(2, "0")}:${ss}` : `${mm}:${ss}`;
}
