"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutGroup } from "motion/react";
import * as m from "motion/react-m";
import { fetchPredictionNext, type PredictionResponse } from "@/lib/api";
import { fmtDay } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { ThemeToggle } from "./theme-toggle";

const TABS = [
  { href: "/predict", label: "Predict" },
  { href: "/strategy", label: "Strategy" },
];

/** Race chip reads the shared prediction cache only; it never triggers the (slow) prediction request itself. */
function RaceChip() {
  const { data } = useApi<PredictionResponse>("predict-next", fetchPredictionNext, true);
  if (!data) return null;
  const { race, stale } = data;
  return (
    <div
      className="hidden items-center gap-2 rounded-sm border border-line px-2 py-1 font-mono text-[11px] tracking-wide text-muted-fg uppercase lg:flex"
      title={stale ? "Showing a saved prediction; live data was unavailable" : "Prediction computed live"}
    >
      <span
        aria-hidden
        className={`live-dot size-1.5 rounded-full ${stale ? "bg-warn" : "bg-good"}`}
        style={{ animation: stale ? undefined : "pulse-dot 2.4s ease-in-out infinite" }}
      />
      <span>
        {race.name} · R{race.round} · {fmtDay(race.date)}
      </span>
      <span className="sr-only">{stale ? "(saved prediction)" : "(live prediction)"}</span>
    </div>
  );
}

export function SiteNav() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg">
      <div className="mx-auto flex h-12 max-w-[1200px] items-center gap-4 px-4 sm:px-6">
        <Link href="/predict" className="display text-[17px] leading-none text-fg" aria-label="Apex, home">
          APEX<span className="text-signal">/</span>
        </Link>
        <nav aria-label="Primary" className="flex h-full items-stretch">
          <LayoutGroup id="nav">
            {TABS.map((t) => {
              const active = pathname === t.href;
              return (
                <Link
                  key={t.href}
                  href={t.href}
                  aria-current={active ? "page" : undefined}
                  className={`relative flex items-center px-3 text-sm font-medium transition-colors ${
                    active ? "text-fg" : "text-muted-fg hover:text-fg"
                  }`}
                >
                  {t.label}
                  {active && (
                    <m.span
                      layoutId="nav-underline"
                      className="absolute inset-x-2 -bottom-px h-0.5 bg-signal"
                      transition={{ type: "spring", stiffness: 500, damping: 38 }}
                    />
                  )}
                </Link>
              );
            })}
          </LayoutGroup>
        </nav>
        <div className="ml-auto flex items-center gap-3">
          <RaceChip />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
