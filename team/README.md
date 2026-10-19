# Team protocol

Agents working on this repo, defined in `.claude/agents/`:

| Role | Agent | Owns |
|---|---|---|
| Orchestrator | main Claude session | `team/BOARD.md`, task routing, git commits, final call on scope |
| Researcher | `researcher` | `team/research/` (read-only on code) |
| Data scientist | `data-scientist` | `data_loader.py`, `tire_model.py` + other fitting modules, `pull_races.py`, `calibrate_*.py`, `validate_replay.py`, `data/` |
| Backend engineer | `backend-engineer` | `race_simulator.py`, `monte_carlo.py`, engine/optimizer modules, `api/`, `main.py`, `services/` |
| Frontend engineer | `frontend-engineer` | `frontend/` |
| QA engineer | `qa-engineer` | `backend/tests/` additions, QA reports. Never fixes product code |

## How work flows

1. The orchestrator writes a task in `BOARD.md` with an id (`T-###`), an owner, dependencies, and a definition of done.
2. The orchestrator launches the owner agent with the task id. Agents can't talk to each other directly. All communication goes through files in `team/`, and the orchestrator routes.
3. The owner works **only in their lane**. If a change is needed in someone else's files, they write it as a request in their handoff and don't make it themselves.
4. The owner writes a handoff (template below) to `team/handoffs/<task-id>-<role>.md`.
5. The orchestrator sends code changes to `qa-engineer` for verification. On `FAIL`, the task goes back to the owner with the QA report.
6. On `PASS`, the orchestrator updates `BOARD.md`, commits, and opens the follow-up tasks the handoff asked for.

Only the orchestrator edits `BOARD.md` and runs `git commit`. When two agents run in parallel and could touch overlapping files, the orchestrator gives each one an isolated worktree.

## Handoff template

```markdown
# <task-id> — <role> — <short title>

**Status:** done | partial | blocked
**Next owner:** <role> — <what they should do>

## What I did
- ...

## Files changed
- path — one-line reason

## Verification
- commands run + results (paste key numbers; before → after for metrics)
- what I did NOT verify

## Interface / contract changes
- exact new fields, signatures, units (or "none")

## Requests for other roles
- <role>: <concrete ask>

## Open questions / risks
- ...
```

QA reports use the same template, with `VERDICT: PASS | PASS WITH NOTES | FAIL` as the first line.
