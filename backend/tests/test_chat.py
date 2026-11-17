"""T-058: in-app chat. Ollama is always mocked (it is not installed in CI or on the dev machine)."""

import pytest
import requests
from fastapi.testclient import TestClient

from app.main import app
from app.services import chat_service as cs

client = TestClient(app)


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.text = payload, status, str(payload)
        self.ok = status < 400

    def json(self):
        return self._p

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(str(self.status_code))


def _tool_msg(name, args):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}}


def _text_msg(text):
    return {"message": {"role": "assistant", "content": text}}


BODY = {"messages": [{"role": "user", "content": "What tracks?"}], "context": {"track_id": "monza"}}


def test_status_when_ollama_down(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(cs.requests, "get", boom)
    r = client.get("/chat/status")
    assert r.status_code == 200
    d = r.json()
    assert d["available"] is False and d["ollama_reachable"] is False
    assert "ollama pull" not in d["message"] and "Install" in d["message"]


def test_status_model_not_pulled_and_pulled(monkeypatch):
    monkeypatch.setattr(cs, "OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: FakeResp({"models": [{"name": "llama3:8b"}]}))
    d = client.get("/chat/status").json()
    assert d["ollama_reachable"] and not d["available"] and "ollama pull qwen2.5:7b" in d["message"]
    monkeypatch.setattr(cs.requests, "get", lambda *a, **k: FakeResp({"models": [{"name": "qwen2.5:7b"}]}))
    d = client.get("/chat/status").json()
    assert d["available"] and d["model_pulled"]


def test_tool_calling_round_trip(monkeypatch):
    sent = []
    replies = iter([_tool_msg("list_tracks", {"bogus": 1}), _text_msg("Monza is available.")])

    def fake_post(url, json=None, timeout=None):
        sent.append(json)
        return FakeResp(next(replies))

    monkeypatch.setattr(cs.requests, "post", fake_post)
    r = client.post("/chat", json=BODY)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["reply"] == "Monza is available."
    assert [c["name"] for c in d["tool_calls"]] == ["list_tracks"]
    assert d["tool_calls"][0]["result_summary"]
    assert d["caveats"]  # collected from the tool result
    # first request carries tools and the track context; second carries the tool result
    assert sent[0]["tools"] and any("monza" in m["content"] for m in sent[0]["messages"] if m["role"] == "system")
    assert sent[1]["messages"][-1]["role"] == "tool"


def test_simulate_grid_schema_and_call():
    schema = {t["function"]["name"]: t["function"]["parameters"] for t in cs.TOOL_SCHEMAS}
    props = schema["simulate_grid"]["properties"]
    assert "forced_sc_lap" in props and schema["simulate_grid"]["required"] == ["track_id"]
    r = cs.run_tool("simulate_grid", {"track_id": "austin", "n_sims": 30, "forced_sc_lap": 15, "top_n": 3, "bogus": 1})
    assert "error" not in r and len(r["drivers"]) == 3 and r["forced_sc_lap"] == 15


def test_unknown_tool_is_reported_not_crashed(monkeypatch):
    replies = iter([_tool_msg("nope", {}), _text_msg("ok")])
    monkeypatch.setattr(cs.requests, "post", lambda *a, **k: FakeResp(next(replies)))
    d = client.post("/chat", json=BODY).json()
    assert d["tool_calls"][0]["result_summary"].startswith("error")


def test_503_when_ollama_down(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(cs.requests, "post", boom)
    r = client.post("/chat", json=BODY)
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]


def test_503_when_model_missing(monkeypatch):
    monkeypatch.setattr(cs.requests, "post", lambda *a, **k: FakeResp({"error": "model not found"}, 404))
    r = client.post("/chat", json=BODY)
    assert r.status_code == 503 and "ollama pull" in r.json()["detail"]


def test_round_cap(monkeypatch):
    calls = {"n": 0, "tools_sent": []}

    def fake_post(url, json=None, timeout=None):
        calls["n"] += 1
        calls["tools_sent"].append("tools" in json)
        if "tools" in json:
            return FakeResp(_tool_msg("list_tracks", {}))  # never stops asking for tools
        return FakeResp(_text_msg("final answer"))

    monkeypatch.setattr(cs.requests, "post", fake_post)
    d = client.post("/chat", json=BODY).json()
    assert len(d["tool_calls"]) == cs.MAX_TOOL_ROUNDS
    assert calls["n"] == cs.MAX_TOOL_ROUNDS + 1
    assert calls["tools_sent"][-1] is False  # the last call is forced tool-less
    assert d["reply"] == "final answer"


def test_validation():
    assert client.post("/chat", json={"messages": []}).status_code == 422
    assert client.post("/chat", json={"messages": [{"role": "system", "content": "x"}]}).status_code == 422
