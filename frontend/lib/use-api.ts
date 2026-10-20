"use client";

import { useCallback, useEffect, useState } from "react";

/**
 * Tiny session cache so Predict -> Strategy -> Predict does not refetch (the first
 * /predict/next call runs a full simulation) and the nav chip can read the same data.
 */
interface Entry<T> {
  data: T | null;
  error: string | null;
  promise: Promise<void> | null;
  listeners: Set<() => void>;
}

const cache = new Map<string, Entry<unknown>>();

function entry<T>(key: string): Entry<T> {
  let e = cache.get(key) as Entry<T> | undefined;
  if (!e) {
    e = { data: null, error: null, promise: null, listeners: new Set() };
    cache.set(key, e as Entry<unknown>);
  }
  return e;
}

function load<T>(key: string, fetcher: () => Promise<T>, force = false): void {
  const e = entry<T>(key);
  if (e.promise || (!force && e.data)) return;
  e.error = null;
  e.promise = fetcher()
    .then((d) => {
      e.data = d;
    })
    .catch((err) => {
      e.error = err instanceof Error ? err.message : "Request failed";
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
    return () => {
      e.listeners.delete(l);
    };
    // fetcher is a stable module-level function at every call site
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, passive]);

  const reload = useCallback(() => load(key, fetcher, true), [key, fetcher]);
  const e = entry<T>(key);
  return { data: e.data, error: e.error, loading: !!e.promise || (!e.data && !e.error && !passive), reload };
}
