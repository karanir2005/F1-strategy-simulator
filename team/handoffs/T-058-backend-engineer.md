# T-058 — backend-engineer — in-app strategy chat (local Ollama)

**Status:** done
**Next owner:** orchestrator — commit; optionally QA/manual test once Ollama + `qwen2.5:7b` is installed (never run against a real model here).

## What I did
- `backend/app/services/chat_service.py` (new): Ollama tool-calling loop over `requests` (no `ollama` package). Tools are the MCP server's functions, imported lazily from `app.mcp_server.TOOLS` (no refactor, no duplication; MCP tests untouched and passing). Hand-trimmed Ollama-format `tools` schemas; model-invented arguments are dropped; tool exceptions become `{"error"}` results; tool results sent back to the model are capped at 6000 chars. Max 5 tool rounds, then one tool-less call to force a final answer. System prompt: answer only from tool results, quote caveats. Track context is appended to the system prompt.
- `backend/app/api/routes/chat.py` (new) and a 2-line registration in `backend/app/main.py` (re-read right before; kept the T-056 CORS change).
- Frontend: `frontend/components/strategy/strategy-chat.tsx` (collapsible panel, collapsed by default; status check on mount; setup note if unavailable; messages, "used tools: ..." line, caveats list), one line plus import in `frontend/app/strategy/page.tsx`, chat types and fetchers in `frontend/lib/api.ts`. `call()` got an optional `timeoutMs` param (default unchanged 60 s; chat uses 200 s).
- README section "In-app chat (local, optional)".

## Files changed
backend/app/services/chat_service.py, backend/app/api/routes/chat.py, backend/app/main.py, backend/tests/test_chat.py (8 tests), frontend/components/strategy/strategy-chat.tsx, frontend/app/strategy/page.tsx, frontend/lib/api.ts, README.md.

## Verification
- `pytest backend/tests/test_chat.py`: 8 passed (status down / model missing / pulled, mocked round trip incl. tools+context sent, unknown tool, 503 down, 503 model 404, round cap = 5 tool rounds + 1 forced tool-less call, validation).
- Full `pytest`: 390 passed (153 s).
- `npx tsc --noEmit` clean; `npm run lint` clean. No `npm run build`.
- Not verified: a real Ollama model, and the panel in a browser (Ollama not installed; status would show the setup note). Small-model tool-call quality is untested.

## Interface / contract changes
Env: `OLLAMA_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `qwen2.5:7b`).

`GET /chat/status` -> 200
```json
{"available": bool, "ollama_reachable": bool, "model": str, "model_pulled": bool, "url": str, "message": str}
```
`available` = reachable and model pulled; `message` is a human setup hint when not available. Ollama probe timeout 2 s.

`POST /chat`
```json
request:  {"messages": [{"role": "user"|"assistant", "content": str(<=4000)}] (1..40),
           "context": {"track_id": str|null, "race": str|null} | null}
response: {"reply": str,
           "tool_calls": [{"name": str, "args": object, "result_summary": str}],
           "caveats": [str]}      // deduplicated union of the tool results' caveats
```
Errors: 503 `{"detail": "<message>"}` when Ollama is unreachable, the model is not pulled (Ollama 404), or Ollama returns an error; 422 on a bad body. Each Ollama call has a 180 s timeout.

## Requests for other roles
- T-034 owner: when `simulate_grid` is added to `mcp_server.TOOLS`, it is not in `chat_service.TOOL_SCHEMAS` automatically. Add a schema entry there (unknown names are reported to the model as errors).

## Open questions / risks
- The route is synchronous and blocks a worker thread for the duration of inference (up to minutes). Fine for a single local user.
- `chat_service` reaches tools via `app.mcp_server`, which calls `logging.basicConfig(stderr, WARNING)` on import; harmless.
- Deployed free-tier backend has no Ollama: panel shows the setup note (status returns available=false).
