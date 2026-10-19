# T-008 — frontend-engineer — Paired delta in the UI + T-006 QA fixes

**Status:** done
**Next owner:** qa-engineer — verify against the T-007 contract; browser checks listed below are still open.

## What I did
- `api.ts` now matches T-007 exactly: `StrategyResult.delta_mean/delta_p10/delta_p90` (seconds), `SimulateResponse.reference_strategy`. Bounds exported as constants (`MAX_RACE_LAPS` 100, `MAX_START_AGE` 25, `MAX_STRATEGIES` 20, `MAX_SIMULATIONS` 50000). Backend 422 `detail` lists are rendered as readable text (`strategies: all strategies must cover the same total ...`) instead of raw JSON.
- Results table: new primary columns "Time vs reference" ("+1.73s vs 2-stop"; reference row shows a "reference" badge and the text "reference") and "Gap p10 to p90" ("-4.61s to +8.00s"; when p10 == p90 within 0.005s it shows "constant +0.31s", since same-stop-count strategies share draws). Mean/p10/p50/p90 kept after them; Win probability is last and labelled secondary in its tooltip. Footnotes: added a paired-gap explanation (including negative p10 meaning, and that gaps of about a second or less are inside model error, which is a frontend wording choice, no threshold from the API); win-probability footnote updated to say same-stop-count strategies can show exactly 0%/100% and to read the gap first.
- Client-side mirror of API bounds with inline messages (Run disabled, message under the button): laps per stint 1..100, per-strategy total <= 100, equal totals across strategies (message lists each total), start_age 1..25 (only for used sets), strategies <= 20 (Add strategy button also disabled at 20), simulations 1..50,000 integer, pit loss >= 0. Number inputs have min/max attributes.
- F-1: "used set" is its own `usedSet` boolean in stint state, separate from `startAge`; typing "15" keeps the input mounted. Payload sends `start_age` only if `usedSet && startAge !== 1`.
- F-4: numeric fields (base lap time, sims, pit loss, laps, start age) hold `number | ""`; clearing shows empty, validation blocks Run until filled.

## Files changed
- frontend/lib/api.ts — contract types, bound constants, 422 formatting
- frontend/app/page.tsx — delta columns, validation, usedSet state, empty-able numeric fields, footnotes

## Verification
- `npm run lint`: clean. `npm run build`: compiled, 5/5 static pages.
- Backend run on port 8010 (stopped afterwards, no listener left). POSTed a payload shaped exactly as the UI builds it (used set `start_age: 15` on stint 1 of "1-stop", other stints omit `start_age`): 200; `reference_strategy` "2-stop"; "1-stop" delta_mean +1.73, p10 -4.61, p90 +8.00; reference delta 0/0/0; win_probability 0.355/0.645. Unequal-total payload (40 vs 30 laps) returned the expected 422 with the "same total race laps" message, which `formatDetail` parses.
- NOT verified: anything in a browser. Dev server never started, so I did not see: the new table at narrow widths (table is now 8 columns; it relies on the existing Table wrapper scrolling horizontally, unchecked), dark mode of the reference badge, typing "15" into start age (fixed by code reading only), clearing fields, inline messages, Add strategy disabling at 20, or the rendered 422 message. `formatDetail` was not unit-tested against a live 422 through the UI path (only the response body shape was seen via curl). There are no frontend tests.

## Interface / contract changes
- none to the backend; frontend types now equal the T-007 response/request contract.

## Requests for other roles
- backend-engineer (optional): `win_probability` tie-splitting (already on backlog) would also help the UI; nothing blocking.

## Open questions / risks
- The "about a second or less is inside model error" wording is hardcoded copy, not data-driven.
- Unticking then re-ticking "used set" keeps the previously typed age (changed from T-006, which reset to 1).
- The error box still appends "Is the backend running?" even for 422s.
