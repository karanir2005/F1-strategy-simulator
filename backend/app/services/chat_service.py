"""
In-app strategy chat (T-058): a tool-calling loop against a local Ollama model.

Why local Ollama: no API key and no cost. The tools are the same functions the MCP server
(T-054) exposes (imported from app.mcp_server, not copied), so the chat and Claude Desktop
answer from one code path and the same `caveats` lists.

Small local models are unreliable at complex JSON schemas, so the tool schemas below are
hand-trimmed to the common arguments (simulate_strategy's `strategies` is the hard one).
Unknown arguments the model invents are dropped before the call.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
MAX_TOOL_ROUNDS = 5
STATUS_TIMEOUT_S = 2.0
CHAT_TIMEOUT_S = 180.0  # CPU-only inference of a 7B model can be slow
MAX_TOOL_RESULT_CHARS = 6000  # keep the small model's context from filling with one tool result


class OllamaUnavailable(Exception):
    """Ollama is not running, or the model is not pulled."""


SYSTEM_PROMPT = (
    "You are an assistant inside an F1 race-strategy simulator (2026 season). Answer ONLY from the results of the "
    "tools; never use your own memory for lap times, pit plans, race results or probabilities. If a tool result has "
    "an 'error', say so or retry with corrected arguments. If no tool can answer, say that plainly. "
    "Each tool result has a 'caveats' list: quote or paraphrase the relevant ones in your answer, and do not "
    "overstate the numbers (the engine simulates one car in clean air; gaps under about 1 second are not meaningful). "
    "Be concise. Times are in seconds. Use search_project_notes for 'why' questions and cite the file path."
)

_S = {"type": "string"}
_I = {"type": "integer"}
_B = {"type": "boolean"}


def _tool(name: str, desc: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required or []}}}


TOOL_SCHEMAS = [
    _tool("list_tracks", "List circuits known to the strategy tools (track_id, name, total_laps, pit_loss_s).", {}),
    _tool("predict_race", "Win/podium probabilities for a race. Omit round for the next race.",
          {"round": _I, "top_n": _I}),
    _tool("get_race_results", "Official classification of a finished round.", {"round": _I, "season": _I}, ["round"]),
    _tool("optimize_strategy", "Best pit-stop plans for a track. track_id comes from list_tracks.",
          {"track_id": _S, "max_stops": _I, "n_sims": _I, "top_n": _I, "safety_cars": _B, "use_track_wear": _B},
          ["track_id"]),
    _tool("simulate_strategy",
          "Monte Carlo comparison of explicit strategies; every strategy must cover the same total laps.",
          {"strategies": {"type": "array", "description": "list of {name, stints:[{compound, laps}]}",
                          "items": {"type": "object"}},
           "track_id": _S, "n_simulations": _I, "safety_cars": _B}, ["strategies"]),
    _tool("simulate_grid",
          "Simulate the WHOLE grid together (traffic, overtakes, safety cars): per-driver win/podium/points. "
          "forced_sc_lap=N forces a safety car from lap N ('what if SC on lap 30'). strategies overrides "
          "{driver code: [{compound, laps}]}.",
          {"track_id": _S, "round": _I, "n_sims": _I, "forced_sc_lap": _I, "forced_sc_laps": _I,
           "safety_cars": _B, "strategies": {"type": "object", "description": "{CODE: [{compound, laps}]}"},
           "top_n": _I}, ["track_id"]),
    _tool("optimize_pit_policy",
          "Best safety-car-reactive pit rule for one driver at a track: 'if an SC/VSC starts between laps a-b and "
          "tyres are >= k laps old, box for compound C', with expected points gain vs the fixed plan and a 95% "
          "interval. Takes about 10 seconds. driver defaults to the pole sitter.",
          {"track_id": _S, "driver": _S, "base_plan": {"type": "array", "description": "[{compound, laps}] optional",
                                                       "items": {"type": "object"}},
           "n_sims": _I}, ["track_id"]),
    _tool("query_laps", "Filtered per-lap data for one 2026 race (name fragment, e.g. 'Austin'). Max 200 rows.",
          {"race": _S, "driver": _S, "lap_from": _I, "lap_to": _I, "compound": _S, "green_only": _B,
           "max_rows": _I}, ["race"]),
    _tool("search_project_notes", "Search the project's research notes and handoffs for why/how questions.",
          {"query": _S, "k": _I}, ["query"]),
]


def _tool_functions() -> dict[str, Any]:
    from app import mcp_server  # lazy: keeps route import cheap

    return {fn.__name__: fn for fn in mcp_server.TOOLS}


def _summarize(result: Any) -> str:
    """One-line description of a tool result for the UI."""
    if not isinstance(result, dict):
        return str(result)[:120]
    if "error" in result:
        return f"error: {str(result['error'])[:160]}"
    parts = []
    for k, v in result.items():
        if k == "caveats":
            continue
        parts.append(f"{k}[{len(v)}]" if isinstance(v, (list, dict)) else f"{k}={v}")
        if len(parts) >= 5:
            break
    return ", ".join(parts)[:200]


def run_tool(name: str, args: Any) -> dict:
    fns = _tool_functions()
    fn = fns.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'; available: {sorted(fns)}", "caveats": []}
    if isinstance(args, str):  # some models send arguments as a JSON string
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    args = args if isinstance(args, dict) else {}
    allowed = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    clean = {k: v for k, v in args.items() if k in allowed}
    try:
        return fn(**clean)
    except Exception as exc:  # a bad model-supplied argument must not 500 the chat
        return {"error": f"{type(exc).__name__}: {exc}", "caveats": []}


# ------------------------------------------------------------------ ollama


def status() -> dict:
    out = {"available": False, "ollama_reachable": False, "model": OLLAMA_MODEL, "model_pulled": False,
           "url": OLLAMA_URL, "message": ""}
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=STATUS_TIMEOUT_S)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError):
        out["message"] = f"Ollama is not reachable at {OLLAMA_URL}. Install it from ollama.com and start it."
        return out
    out["ollama_reachable"] = True
    want = OLLAMA_MODEL if ":" in OLLAMA_MODEL else OLLAMA_MODEL + ":latest"
    out["model_pulled"] = want in names
    out["available"] = out["model_pulled"]
    if not out["model_pulled"]:
        out["message"] = f"Ollama is running but model '{OLLAMA_MODEL}' is not pulled. Run: ollama pull {OLLAMA_MODEL}"
    return out


def _ollama_chat(messages: list[dict], tools: list[dict] | None) -> dict:
    body: dict = {"model": OLLAMA_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0.2}}
    if tools:
        body["tools"] = tools
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=CHAT_TIMEOUT_S)
    except requests.RequestException as exc:
        raise OllamaUnavailable(
            f"Cannot reach Ollama at {OLLAMA_URL} ({type(exc).__name__}). Install Ollama and run "
            f"`ollama pull {OLLAMA_MODEL}`.") from exc
    if r.status_code == 404:
        raise OllamaUnavailable(f"Model '{OLLAMA_MODEL}' is not pulled. Run: ollama pull {OLLAMA_MODEL}")
    if not r.ok:
        raise OllamaUnavailable(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
    return r.json().get("message") or {}


def chat(messages: list[dict], context: dict | None = None) -> dict:
    """Run the tool loop. At most MAX_TOOL_ROUNDS rounds of tool calls, then one tool-less call to force an answer."""
    system = SYSTEM_PROMPT
    ctx = {k: v for k, v in (context or {}).items() if v}
    if ctx:
        system += f" The user is currently looking at: {json.dumps(ctx)}. Use that track_id when they say 'this track'."
    convo = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in messages]
    calls: list[dict] = []
    caveats: list[str] = []

    for round_no in range(MAX_TOOL_ROUNDS + 1):
        capped = round_no == MAX_TOOL_ROUNDS
        msg = _ollama_chat(convo, None if capped else TOOL_SCHEMAS)
        tcs = msg.get("tool_calls") or []
        if not tcs or capped:
            reply = (msg.get("content") or "").strip()
            if not reply:
                reply = "I could not finish an answer within the tool-call limit; try a narrower question."
            return {"reply": reply, "tool_calls": calls, "caveats": caveats}
        convo.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
        for tc in tcs:
            fn = tc.get("function") or {}
            name, args = fn.get("name", ""), fn.get("arguments") or {}
            result = run_tool(name, args)
            for c in result.get("caveats", []) if isinstance(result, dict) else []:
                if c not in caveats:
                    caveats.append(c)
            calls.append({"name": name, "args": args if isinstance(args, dict) else {"raw": str(args)},
                          "result_summary": _summarize(result)})
            text = json.dumps(result, default=str)
            if len(text) > MAX_TOOL_RESULT_CHARS:
                text = text[:MAX_TOOL_RESULT_CHARS] + "...[truncated]"
            convo.append({"role": "tool", "tool_name": name, "content": text})
    raise AssertionError("unreachable")
