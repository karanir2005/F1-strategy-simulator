import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js";

/**
 * The dev server builds into its own folder so `npm run build` (run by agents and
 * lint/build checks) can't overwrite the chunks of a running `next dev`, which shows
 * up in the browser as "ChunkLoadError: Loading chunk app/layout failed".
 * @type {(phase: string) => import('next').NextConfig}
 */
const nextConfig = (phase) => ({
  distDir: phase === PHASE_DEVELOPMENT_SERVER ? ".next-dev" : ".next",
});

export default nextConfig;
