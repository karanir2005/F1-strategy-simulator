# T-018 — frontend-engineer — Track dropdown + "Find optimal strategy"

**Status:** done
**Next owner:** qa-engineer — verify the UI flow in a browser (not exercised visually by me); then orchestrator.

## What I did
- Race Settings: track dropdown fed by `GET /tracks` (defaults to Austin, else first). Read-only chips: laps, pit loss, base lap, compounds (`S=C4 M=C3 H=C1`, `?` if unknown), source season (amber "(not 2026)" flag for 2025 data; hover shows the track `notes`).
- "Override numbers" checkbox: reveals editable base lap / pit loss / total laps, pre-filled from the track (re-prefilled when the track changes while on). `/simulate` and `/optimize` always send `track_id`; numeric overrides are sent only when the toggle is on (or when no track is available, e.g. `/tracks` failed: the old manual fields are then shown and `/simulate` sends `base_laptime` + `pit_loss_seconds` as before).
- "Find optimal strategy" + max stops select (1-3, default 3). Requests `top_n: 10`, dedupes client-side by sorted (compound, laps) multiset, shows top 5: stint bar (existing compound colours, C-number in label), pit laps, mean time, delta to best with p10-p90, race-time p10-p90, params used, screened/simulated counts, a note that gaps under ~5 s are noise and compound offsets/cliffs are placeholders, and a visible "What this model ignores" box listing `assumptions` verbatim.
- States: tracks loading / failed (red box, manual fallback), optimizer button spinner, optimizer error box (shows backend detail + status: 404, 422, 503 text come through; unreachable backend gives "cannot reach the backend at <url>"). Changing track clears the old optimizer result.
- Existing manual comparison, used-set / start-age inputs and paired-delta table untouched. Small hint when a track is selected: strategies should add up to the track's lap count.

## Files changed
- frontend/lib/api.ts — `TrackOut`, `OptimizeRequest/Response`, `StintOut`, `StrategyOut`, `ParamsUsed` (mirror pydantic), `fetchTracks`, `runOptimize`; `SimulateRequest.base_laptime` now optional + `track_id`; shared error handling (`call`) with unreachable-backend message.
- frontend/components/optimizer-results.tsx (new) — results card, `dedupeStrategies`.
- frontend/app/page.tsx — Race Settings rework, track/override/optimizer state, validation split (`getNumbersError`; numbers validated only when editable).

## Verification
- `npm run lint`: no warnings/errors. `npm run build`: success (types check).
- Backend on port 8001 (stopped afterwards), real `data/tracks/track_params.json` (23 tracks): `GET /tracks` ok; `POST /optimize` austin max_stops=3 top_n=10 returned 10 plans, params 56 laps / 24.03 s / 98.11 s; `POST /simulate` with `track_id` and no base_laptime ok; unknown track 404. These were curl calls with the same payload shapes the UI sends, NOT the browser UI.
- NOT verified: actual rendering/interaction (dropdown label via base-ui `SelectValue` render function, narrow width, dark mode), 503 path in the UI.

## Interface / contract changes
- none to the backend. Frontend `SimulateRequest.base_laptime` is optional (matches backend).

## Requests for other roles
- backend-engineer: none. Observed: backend already dedupes order-permutations (per its assumptions list), so client dedupe is a no-op safety net.
- data-scientist: optimizer top plan at Austin is M28/S28 with SOFT at its 28-lap cap; ranking is driven by placeholder compound offsets (already stated in the UI note). Plans #2/#3 (M5/S25/S26 vs H5/S25/S26) are near-duplicates in spirit.

## Open questions / risks
- The "Total laps" field only affects `/optimize` (manual strategies define their own laps); labelled "(optimizer)".
- Chip compound order is hard-coded S/M/H; extra compounds in `compounds` would not show.
