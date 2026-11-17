# T-054 MCP eval: canned questions for Claude Desktop / Claude Code

Spot-check list for `backend/app/mcp_server.py`. Ask each question in a fresh chat with the `f1-strategy-simulator` server enabled. Pass = Claude calls the expected tool and its answer quotes the listed facts (numbers must match the tool JSON; the ones marked "stable" are historical and will not change, the others move when data is refreshed, so compare against the tool output instead). Also check that the answer mentions at least one caveat from the result's `caveats` list.

Facts below were taken from the tools on 2026-11-16.

| # | Question | Expected tool (args) | Facts the answer must quote |
|---|---|---|---|
| 1 | Which circuits can the strategy tools handle? | `list_tracks` | 23 tracks, each with total laps, pit loss and base lap time. Example: Australia 58 laps, pit loss 19.06 s, base lap 81.46 s (may be refit). |
| 2 | Who is predicted to win the next race and with what probability? | `predict_race` (round omitted) | Race name and round (Las Vegas GP, round 21 on 2026-11-22), `mode` (pre_quali until qualifying is out), top driver and `p_win` (about 0.29 for Antonelli on 2026-11-16, will move), the `stale` flag, and the caveat that it is a small Plackett-Luce model. |
| 3 | What did the model predict for Mexico (round 19) and what happened? | `predict_race` (round=19) | Walk-forward prediction, Hamilton on pole with `p_win` about 0.525 (stable); actual result VER won from grid 5, HAM 2nd, ANT 3rd. |
| 4 | Give me the Brazil GP result. | `get_race_results` (round=20) | Norris won from pole (grid 1), Verstappen 2nd after starting 21st, 25 and 18 points, 71 laps (stable). |
| 5 | Who finished on the podium in Mexico? | `get_race_results` (round=19) | VER, HAM, ANT in that order, with grid slots 5, 1, 4 (stable). |
| 6 | What is the best pit strategy for Austin? | `optimize_strategy` (track_id=austin) | 56 laps; best plan is a 1-stop MEDIUM 24 then HARD 32; runner-up SOFT 23 then HARD 33 about 1.5 s slower; best 2-stop about 14 s slower (n_sims=1000 seed 0; small drift with n_sims). Caveats: mostly 1-stop plans, single car, plans within ~1 s are equivalent. |
| 7 | Compare a 1-stop (MEDIUM 30, HARD 41) with a 2-stop (SOFT 20, MEDIUM 25, HARD 26) at Brazil. | `simulate_strategy` (track_id=brazil, two strategies, seed 0) | 1-stop is the reference; 2-stop delta_mean about +10 s (n_simulations=500: 10.2 s). Must NOT present win_probability as a race-win chance; must cite the single-car caveat. |
| 8 | What happens at Monaco if I force a safety car? (trap: not supported yet) | `simulate_strategy` or `optimize_strategy` with `safety_cars=true` (random SC/VSC, not forced at a lap) | Claude must say a forced SC lap is not available in this server yet and that safety_cars=true draws random SC/VSC periods with a fixed pit plan (no reactive stops). |
| 9 | Show Norris's first five laps in Austin. | `query_laps` (race=Austin, driver=NOR, lap_from=1, lap_to=5) | Five rows with lap time, compound, tyre age, position; `n_matched` 5, not truncated. |
| 10 | Why does the optimizer treat SOFT, MEDIUM and HARD as equal on pace? | `search_project_notes` (query about compound pace offset / SOFT ordering) | Cites file paths and headings (expect team/handoffs or team/research hits such as T-049 tyre wear audit or the ROADMAP Phase 3 section); says the 2026 free fit gives no pace advantage for softer tyres so the ordering is clipped and the compounds differ by wear and stint caps. |
| 11 | What is on the roadmap after the single-car simulator? | `search_project_notes` (query: multi-car grid simulation roadmap) | Cites `team/ROADMAP.md` (and README Limitations); full-grid multi-car sim is the next step. |
| 12 | Which MCP-related ideas did the AI/ML research note rank first? | `search_project_notes` (query: MCP server ranking) | `team/research/T-051-ai-ml-ideas.md`, ranking item 1: MCP server plus forced-SC parameter, effort S-M. |

Scoring: 10 or more of 12 pass is good. A miss on 8 is a trap failure only if Claude pretends a forced SC lap was simulated.
