# T-031 — frontend-engineer — race-week /predict

**Status:** done, ready for QA. **Next owner:** qa-engineer.

## What changed
- `lib/api.ts`: `PredictionResponse.grid_source?: "qualifying_classification" | "override_file" | null` (backend: `str | None = None`).
- `components/predict/race-header.tsx`: note under the mode badge from `grid_source` ("Grid adjusted for penalties" / "Grid is the qualifying classification; penalties not applied"; nothing for null/pre-quali; grid mode with null source gets a neutral line). The old hard-coded "penalties are not included" sentence is gone because it would be wrong for `override_file`.
- New `components/predict/countdown.tsx`: live countdown to qualifying day (pre-quali, quali_date in future) else race day; hidden after that. The API has dates only, so the target is 00:00 UTC of that day and the label says so. Renders nothing until mounted (no hydration mismatch); reduced motion drops the seconds and ticks once a minute; screen reader gets a static sentence.
- Hero: "Grid set" (accent outline) vs "Pre-qualifying" (muted) chip next to "Most likely winner".
- Driver detail: added a "Not finishing" bar next to the Win/Podium/Top 10 bars; expected finish and start position stay as text.
- NOT built: the "what changed since pre-quali" strip. `/predict/next` returns only the current mode's numbers, with no pre-quali win probabilities. Needs a backend field (e.g. `drivers[].p_win_prequali`) or a saved pre-quali CSV exposed; I did not derive or fake it.

## Verification
- `npm run lint` clean, `npm run build` passes (`/predict` 149 kB).
- Headless Edge over DevTools against the live backend (8001) and the Next server already on 3000. Grid mode and both `grid_source` values were checked with a mocked response (live API is currently `pre_quali`, `grid_source: null`). Seen: 1280 dark (pre-quali), 1280 light (grid, override_file, expanded row), 400 dark /predict (grid, classification note, expanded row) and /strategy (before and after Find optimal). No horizontal overflow (scrollWidth = viewport) and no console errors or warnings in any run; no overlapping rows seen.
- Hero dot plot: second driver of each team is a hollow ring and legend swatches match, in dark and light.
- Not verified: reduced-motion behaviour of the countdown (code path only), Safari, countdown across a DST/timezone boundary (it uses UTC epoch maths).

## Housekeeping
- I started no servers. The uvicorn (8001) and `next dev` (3000, pid 9172) were already running and are still running. `npm run build` overwrote `.next`, so restart the 3000 dev server if it serves errors.
- No commits; nothing touched in backend/ or data/.
