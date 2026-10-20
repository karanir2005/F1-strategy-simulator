# T-022 — researcher — UI inspiration and design spec

**Status:** done
**Next owner:** frontend-engineer — implement T-024 from `team/research/T-022-ui-inspiration.md`.

## What I did
- Fetched motion.dev docs (bundle, AnimatePresence, accessibility), npm registry (motion 14.4.0, React 18 ok), Linear and Vercel Geist colour write-ups, forecast-uncertainty research, prediction-market UI critiques, Pirelli 2026 colours.
- Wrote design spec: direction "Pit wall instrument", dark+light tokens, fonts (Archivo, Geist, JetBrains Mono), scale, wireframes for /predict and /strategy, motion table, motion code sketches, sources, don't list.

## Files changed
- team/research/T-022-ui-inspiration.md — new

## Verification
- Motion size/peer/API facts from motion.dev and npm registry fetches. Not verified: Geist availability in next/font/google on Next 14.2, 2026 team hex values, formula1.com typeface, View Transitions support in Next 14.

## Interface / contract changes
- none

## Requests for other roles
- frontend-engineer: use enter-only `template.tsx` (the common AnimatePresence-in-template exit pattern does not work); `LazyMotion domMax strict` + `motion/react-m`; set Recharts `isAnimationActive={false}` or draw bars/range plots with plain SVG/divs; check Geist font import at install; verify team hexes for the 2026 grid.
- backend-engineer: none required; `/predict/next` fields in the contract cover the wireframe (need `p_dnf`, `exp_pos`, `mode`, `stale`). A per-driver team colour map lives in the frontend.

## Open questions / risks
- Accent colour (orange-flare) is a proposal; orchestrator/user may prefer another.
- Quantile dotplot on /predict needs only p_win (100 dots sampled by largest remainder), no new API field.
