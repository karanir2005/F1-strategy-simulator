"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * Tiny session cache so Predict -> Strategy -> Predict does not refetch (the first
 * /predict/next call runs a full simulation) and the nav chip can read the same data.
 * Entries older than MAX_AGE_MS refetch in the background on mount and when the tab
 * regains focus, so a tab left open across a race weekend moves on to the next race.
 */
const MAX_AGE_MS = 5 * 60 * 1000;

interface Entry<T> {
  data: T | null;
  error: string | null;
  promise: Promise<void> | null;
  fetchedAt: number;
  listeners: Set<() => void>;
}

const cache = new Map<string, Entry<unknown>>();

function entry<T>(key: string): Entry<T> {
  let e = cache.get(key) as Entry<T> | undefined;
  if (!e) {
    e = { data: null, error: null, promise: null, fetchedAt: 0, listeners: new Set() };
    cache.set(key, e as Entry<unknown>);
  }
  return e;
}

function load<T>(key: string, fetcher: () => Promise<T>, force = false): void {
  const e = entry<T>(key);
  if (e.promise || (!force && e.data && Date.now() - e.fetchedAt < MAX_AGE_MS)) return;
  e.error = null;
  e.promise = fetcher()
    .then((d) => {
      e.data = d;
      e.fetchedAt = Date.now();
    })
    .catch((err) => {
      // a failed background refresh keeps the data on screen
      if (!e.data) e.error = err instanceof Error ? err.message : "Request failed";
    })
    .finally(() => {
      e.promise = null;
      e.listeners.forEach((l) => l());
    });
  e.listeners.forEach((l) => l());
}

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** passive = read the cache only, never trigger a request (used by the nav chip). */
export function useApi<T>(key: string, fetcher: () => Promise<T>, passive = false): ApiState<T> {
  const [, bump] = useState(0);

  useEffect(() => {
    const e = entry<T>(key);
    const l = () => bump((n) => n + 1);
    e.listeners.add(l);
    if (!passive) load(key, fetcher);
    l();
    const onFocus = () => {
      if (!passive && document.visibilityState === "visible") load(key, fetcher);
    };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onFocus);
    return () => {
      e.listeners.delete(l);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onFocus);
    };
    // fetcher is a stable module-level function at every call site
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, passive]);

  const reload = useCallback(() => load(key, fetcher, true), [key, fetcher]);
  const e = entry<T>(key);
  return { data: e.data, error: e.error, loading: !e.data && (!!e.promise || (!e.error && !passive)), reload };
}
