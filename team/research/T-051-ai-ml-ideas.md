# T-051: AI/ML ideas for the F1 strategy simulator

Researched 2026-11-16. "V" = verified from a source I opened. "I" = my inference. Effort: S <1 day, M 2-4 days, L 1+ week.

## Code facts checked
- API today (`backend/app/api/routes/`): `GET /predict/next|history|races|race/{round}`, `GET /tracks`, `POST /optimize`, `POST /simulate`. Prediction responses already carry p_win/p_podium/exp_pos/p_dnf/exp_points per driver.
- `safety_cars.py`: SC/VSC timeline is drawn once per MC iteration and shared across strategies; plans are fixed, so there are no reactive pit calls (stated in the docstring).
- `optimizer.py`: screen all plans deterministically, hill-climb, then CRN Monte Carlo on finalists. It is not DP/GA.
- Nothing in the app produces natural language. Nothing exposes lap-level data (`query laps`) over the API; an MCP tool for that needs a new read-only endpoint or direct CSV access.

## 1. MCP server (tools for Claude Desktop / Claude Code)
**What:** a small Python process exposing `predict_race(round)`, `optimize_strategy(track, laps, ...)`, `simulate(strategy)`, `get_race_results(round)`, `list_tracks()`, `query_laps(...)` (read-only, parameterised, row-capped). Lets a user ask "what should Norris do if SC on lap 30?" in Claude and have it call our models.
- **SDK basics (V, https://github.com/modelcontextprotocol/python-sdk):** `pip install "mcp[cli]"`, Python 3.11+ (we're on 3.14). Tools are decorated functions; type hints become the input schema and the docstring the description. Transports: stdio, Streamable HTTP, SSE. `mcp dev server.py` opens the Inspector; `mcp run server.py --transport streamable-http` serves at `http://localhost:8000/mcp`. The README I fetched names the class `MCPServer` (`from mcp.server import MCPServer`); older releases used `FastMCP` from `mcp.server.fastmcp`. **Check which name the pinned version exports before coding** (unverified which release we'd get).
- **Transport choice (I):** stdio for Claude Desktop / Claude Code (client launches it as a subprocess, no ports/auth). Streamable HTTP only if hosted or used by the Messages API MCP connector (https://platform.claude.com/docs/en/agents-and-tools/mcp-connector, listed in the tool-use docs; remote servers only).
- **Wrapping (I):** import `app.services.prediction_service` and the optimizer/simulate functions directly (in-process, no HTTP hop). Alternative: call `localhost:8000` with httpx so MCP and the frontend share one code path; slower to build but zero logic duplication. Prefer in-process for the pure functions, with the Pydantic models reused as return types. Beware stdout: on stdio, nothing may `print()` to stdout (it corrupts the protocol); FastF1/pandas warnings must go to stderr.
- **Value:** conversational access with zero frontend work; strong demo. Does not improve model accuracy. A SC-on-lap-30 question can only be answered with the current engine as "simulate the plan with an SC forced at lap 30", which needs a `forced_sc_lap` parameter on simulate/optimize (small backend change, and the first honest use of the SC machinery).
- **Effort:** S-M (3-6 tools, ~150 lines, plus config snippet in README). **Cost:** none (no API key; Claude is the client). **Risks:** tool outputs too large (cap rows, summarise); optimizer latency (run time unknown to me, measure; return top-3 only); model's known weakness (always ~1 stop) will be repeated confidently by the LLM, so tool results should include caveats/"scope" strings. **Measure:** fixed list of ~10 questions; pass = correct tool called with correct args and the answer's numbers match the endpoint JSON (check by script, not by eye).

## 2. RAG
**Corpus that would actually help (I):**
- Our own `team/handoffs`, `team/research`, model docstrings, calibration JSONs: highest value, because they hold the "why" (e.g. why SOFT pace offset is forced, SC pit-loss ratio) that no public doc has. Small (tens of files).
- FIA 2026 Sporting/Technical Regulations PDFs (public on fia.com; I did not fetch them, so exact URLs unverified): regulation Q&A ("two-compound rule", SC procedures, overtake mode rules).
- Pirelli race previews / tyre allocation (compound choice, expected stops), stewards' decisions (penalties explain grid changes), race reports. Licence/scraping terms unchecked; keep local and private.
**Store:** corpus is small (<~5k chunks). `sqlite-vec` (V: pip package `sqlite-vec`, load via `sqlite_vec.load(db)` after enabling extension loading; 0.1.6/0.1.7 alpha seen on PyPI, https://github.com/vlasky/sqlite-vec) or Chroma are enough locally. pgvector only makes sense once Postgres is actually deployed; not now. Chunk by heading/regulation article (~300-500 tokens, keep article number as metadata so answers can cite "Article X"). Embeddings: needs an embedding model (Anthropic does not offer one; options are Voyage/OpenAI API or local sentence-transformers; choice is a decision for the user, I did not price them).
**Use cases:** (a) "why did the model predict X" - best served by tools + our notes, not generic RAG; (b) regulation Q&A - genuine fit; (c) explaining a strategy - fit if notes are retrieved alongside tool output.
**Honest limits (I):** RAG adds nothing to numeric accuracy. For model numbers (p_win, lap times) tool calls over live data beat retrieval; a vector index of CSV rows is the wrong tool. Regs PDFs are long, table-heavy and revised; stale-answer risk. With a corpus this small, simply putting the handoff notes into context (or a grep tool) may beat a vector store. **Effort:** M (ingestion + eval set). **Cost:** embedding API or local model download. **Measure:** 20 hand-written regulation questions with known article numbers; score retrieval hit@5 and answer correctness with citation.

## 3. LLM features via the Claude API
Docs checked (V, https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): client tools are defined with `name`/`description`/`input_schema`; Claude returns `stop_reason: "tool_use"`; your code runs it and returns a `tool_result`; there is a tool-runner helper in the SDK that automates the loop; `strict: true` on a tool guarantees inputs match the schema; model ids on that page include `claude-opus-5-5`, Sonnet 5.5 and Haiku 5.5 (I use `claude-sonnet-5-5` / `claude-haiku-5-5` per the task statement).
- **3a. Race preview / post-race explanation** (grounded text from `/predict/race/{round}` JSON, plus actual results post-race). Value: the missing natural-language layer on `/predict`. Effort S. Cost: a few cents per race with Sonnet/Haiku (token cost unmeasured; **unverified**, check the models page pricing). Risk: hallucinated numbers; mitigate by passing JSON and requiring every figure to come from it, then a regex check that every number in the text appears in the input. Measure: automated numeric-faithfulness check (100% target) plus a 1-5 human read of 5 races. Cache per (round, model version) so each race is generated once.
- **3b. Strategy chat assistant in `/strategy`** with the same tools as the MCP server, run via the Messages API. Effort M (backend route + streaming chat UI; frontend work). Cost: API key + per-use tokens, must be kept server-side. Risk: users treat output as validated; engine isn't (placeholders). Measure: same 10-question harness as MCP.
- **3c. Structured outputs** for parsing a free-text scenario ("SC on lap 30, Norris on mediums lap 18") into a `/simulate` request. Effort S, but only worth it inside 3b.
- Shared build: the MCP tools and 3b tools should be one function registry (I).

## 4. Classical ML / RL targeting weak spots
- **SC-reactive pit policy via DP over SC states (recommended modelling upgrade).** State = (lap, tyre age, compound, stops made, SC status), action = pit/stay/which compound; expectation over SC timeline we already model (`safety_cars.py`). Tractable because the single-car state space is small. Closest literature: Heine & Thraves 2023, stochastic DP delaying stops to benefit from future SC periods (known only via the RSRL paper's summary, original not opened; **unverified**). Directly attacks "44% of 2026 stops were under SC/VSC" (user-supplied figure). Effort L. Cost none. Risk: SC lap and pit-loss ratios come from only 26 SC and 30 VSC events (per docstring), so policy gains will be noisy; report CIs. Measure: on held-out real races, does the policy pit under real SCs in the laps real top-10 cars did? Compare policy value vs best fixed plan on simulated SC timelines (paired, CRN).
- **RL pit policies.** RSRL (Thomas et al., arXiv 2501.04068, SAC 2025): average finish P5.33 vs P5.63 best baseline on 2023 Bahrain, with explainability (feature importance, decision-tree surrogates); it was for a simulated race; I could not see how it handles SCs from the abstract (the news summary says it does; unverified). Fieni et al. (ETH) arXiv 2512.21570 and multi-agent arXiv 2602.23056: MINLP+RL with energy, tyres, pit timing. Heilmeier et al. 2020 "Virtual Strategy Engineer" (https://mdpi.com/2076-3417/10/21/7805, V): two neural networks decide pit/no-pit and compound, trained on 2014-2019 timing data; code at https://github.com/coderask/f1racesim (V exists). RL needs a multi-car simulator to mean much; our engine is single-car until T-034, so RL is premature (I). Effort L+.
- **XGBoost/LightGBM finishing-position model.** CLAUDE.md plans it. Current PL + DNF model is refit each race and walk-forward evaluated; with ~20 races x 22 drivers in 2026 (~400 rows), gradient boosting is unlikely to beat a well-regularised PL (I, small-sample argument; not tested). Could be a feature-stacking experiment with grid, pace, team form, track. Effort M. Measure: walk-forward log-loss on winner/podium vs current model; adopt only if it beats it (project rule).
- **Hierarchical/Bayesian tyre model** (partial pooling by track and compound, PyMC/NumPyro or simple shrinkage): addresses per-track params and inverted SOFT offset noted in ROADMAP Phase 3. Effort M-L. Measure: leave-one-race-out per-stint s/lap error (existing scorecard).
- **Calibration and conformal intervals.** Reliability diagrams + isotonic/Platt on p_win/p_podium from walk-forward predictions: S, cheap, no dependencies. Conformal prediction for exp_pos intervals or lap-time/race-time bands (tutorial: Angelopoulos and Bates, arXiv 2107.07511, cited from memory; not opened): M, but with ~20 races exchangeability is shaky (I). Measure: coverage of nominal 80% intervals on walk-forward folds; ECE.

## 5. Agentic live "race engineer"
- OpenF1 (V, https://openf1.org/): free tier is historical only, no auth, 3 req/s and 30 req/min; Sponsor tier EUR 9.90/month adds live REST/MQTT/WebSocket, 6 req/s and 60 req/min, up to 10 concurrent MQTT/WS connections; "live" means from 30 min before to 30 min after a session; ~3 s latency. Matches the BOARD incident (FastF1 live paywalled).
- Feasible technically (poll/stream lap, position, pit, race-control, weather; recommend call). Next race with a live window: Las Vegas 2026-11-21 (ROADMAP). But (I): the engine is single-car, has no reactive SC logic, and no gaps/traffic, so live "calls" would be unvalidated and cannot be measured. Multi-agent LLM orchestration adds latency and failure modes for little modelling gain. Do after DP-SC policy + T-034. Effort L. Cost: EUR 9.90/month. Measure (later): replay a past race through the same pipeline offline using historical OpenF1 and score calls against what top teams did.

## Ranking (value for this app / effort / risk)
1. MCP server (+ forced-SC parameter): S-M, no cost, uses what exists, demos the AI concept, zero accuracy risk.
2. Calibration check on predictions: S, improves honesty of existing outputs. (Not an AI "feature" but cheap and measurable.)
3. LLM race preview/post-race text (3a): S, visible in UI, needs API key.
4. SC-reactive DP policy: L, biggest real modelling gain, closes the stated weak spot.
5. Strategy chat (3b): M, builds on 1.
6. RAG over notes + regs: M, narrow use, moderate payoff.
7. Hierarchical tyre model: M-L, depends on Phase 3 work.
8. XGBoost: M, low expected gain on ~400 rows (I).
9. RL / live agentic engineer: L+, premature until multi-car sim.

## Recommended next step: MCP server (T-052 candidate), 5 steps
1. Add `mcp[cli]` to `requirements.txt` (pinned; check class name `MCPServer` vs `FastMCP` in the installed version).
2. Backend: add optional `forced_sc_lap` (+ `sc_length`) to the simulate/optimize core functions and `POST /simulate`/`/optimize`, defaulting to current behaviour; add a read-only `query_laps(race, driver, lap_from, lap_to)` helper over `data/processed/*.csv` with a row cap. Test that defaults reproduce current outputs.
3. New `backend/app/mcp_server.py` (stdio): tools `predict_race`, `get_race_results`, `list_tracks`, `optimize_strategy`, `simulate_strategy`, `query_laps`; return Pydantic-derived dicts plus a `caveats` field (engine placeholders, single-car). Log to stderr only.
4. README snippet for Claude Desktop (`mcpServers` command pointing at `.venv\Scripts\python.exe -m app.mcp_server`) and `claude mcp add` for Claude Code; test with `mcp dev`.
5. Eval script with ~10 canned questions; assert tool selection/args and that quoted numbers match the endpoint JSON. Then 3a (preview text) reuses the same tool functions.

Open: user decision on embedding provider (only needed for RAG); whether to budget an Anthropic API key for 3a/3b.

## Sources
- https://github.com/modelcontextprotocol/python-sdk
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- https://openf1.org/
- https://arxiv.org/abs/2501.04068 (RSRL); https://arxiv.org/html/2512.21570; https://arxiv.org/html/2602.23056
- https://mdpi.com/2076-3417/10/21/7805 ; https://github.com/coderask/f1racesim
- https://github.com/vlasky/sqlite-vec
