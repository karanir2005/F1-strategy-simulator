# T-006 — frontend-engineer — Start-age input + win-probability copy

**Status:** done
**Next owner:** qa-engineer — review the UI behavior listed under "not verified" if a browser check is wanted; otherwise orchestrator can mark done.

## What I did
- Per-stint "used set" checkbox. Default (unchecked) = new set, UI looks as before. Checking it reveals a "start age" integer input (starts at 2, min 1). Semantics per T-004: set already run N laps -> start age N+1.
- Payload sends `start_age` only when it is not 1; omitted otherwise (backend default 1).
- Client validation: start age must be an integer >= 1 (disables Run, shows message).
- Win probability: column header is now "Win probability*" with a hover title, and the footnote now says it is the share of simulations in which the strategy beat the other strategies compared, under the current model (one car in isolation; no safety cars or traffic yet), not a race-win chance. It also notes that shared random draws can give near 0%/100% for same-length/same-stop strategies, and points to mean and p10-p90 first. Table column order unchanged (mean/p10/p50/p90 already precede win probability; the bar chart with p10-p90 error bars remains the primary comparison).
- Fixed a pre-existing lint error (`react/display-name` on the `makePitStopDot` returned component) that was failing `npm run build`; named the inner component.

## Files changed
- frontend/lib/api.ts — `Stint.start_age?: number` (matches `StintIn.start_age`, optional, >= 1)
- frontend/app/page.tsx — startAge state/UI/validation/payload, win-probability copy, display-name lint fix

## Verification
- `npm run lint`: clean. `npm run build`: passes (before my display-name fix, both failed on the pre-existing error).
- Started backend on :8000 and POSTed /simulate directly: start_age omitted/1 mean 5249.6; start_age=15 on the first stint mean 5260.7 (slower, as expected); start_age=0 returns 422. Backend stopped afterward.
- NOT verified: the UI in a browser (dev server not run; checkbox/input interaction, narrow-width wrapping and dark mode were not visually checked; stint row uses flex-wrap so should wrap). The request was exercised via curl, not through the frontend.

## Interface / contract changes
- `Stint` in frontend/lib/api.ts gains `start_age?: number`. No backend change.

## Requests for other roles
- none

## Open questions / risks
- Unchecking "used set" resets age to 1 (value lost); acceptable for now.
- Results table has no explicit "time delta vs best" column; mean/p10-p90 are already shown, so I did not add derived numbers.
