# Instruction audit — 2026-08-13

Audit of every instruction source reachable from this repo and this session: CLAUDE.md files,
rules files, skills, hooks, and memory. Each source was tested against three questions:

1. Would the agent already do this without being told?
2. Is it correcting a weakness the agent no longer has?
3. Does it conflict with anything else the agent has been told?

A "no" on any question marks the instruction for removal.

---

## Headline

**There was almost nothing to audit.** This project has no CLAUDE.md at any scope, no rules
files, no user-authored skills, no user-authored hooks, and no memory entries.

The entire user-authored instruction surface was one file — `.github/copilot-instructions.md`
— and it contained **no behavioral rules at all**. It was a pasted transcript of a planning
conversation describing an architecture that does not exist in this repo.

| Metric | Before | After |
|---|---|---|
| User-authored instruction files | 1 | 1 (`KEEP.md`, staged) |
| Actual behavioral rules | 0 | 4 |
| Words of instruction | 1,203 | 58 |

The rule count going *up* is not a contradiction: the source file contained project decisions
buried in prose, not rules. The audit extracted four of them and discarded the surrounding
1,145 words.

---

## Inventory (13 sources)

| # | Source | Scope | Author | Verdict |
|---|--------|-------|--------|---------|
| 1 | `.github/copilot-instructions.md` (184 lines, 1,203 words) | Project | **User** | **REMOVED** |
| 2 | `~/.claude/launcher-settings.json` | Global | Harness | Out of scope |
| 3 | `~/.claude/session-start-git-identity.sh` | Global | Harness | Out of scope |
| 4 | `~/.claude/stop-hook-git-check.sh` | Global | Harness | Out of scope |
| 5 | `~/.claude/stop-hook-reply-gate.py` | Global | Harness | Inert here |
| 6 | `~/.claude/user-prompt-submit-reply-reminder.py` | Global | Harness | Inert here |
| 7 | `~/.claude/skills/session-start-hook/SKILL.md` | Global | Claude Code built-in | Out of scope |
| 8–13 | `~/.claude/skills/synced/{docx,pdf,pptx,xlsx,morning,skill-creator}` | Global | `source: anthropic` | Out of scope |

### Why sources 2–13 were not touched

They are not user-authored rules. Sources 2–6 are container infrastructure, regenerated inside
`~/.claude/` on every session — they configure git identity and commit signing, and gate the
stop event on uncommitted work. Deleting them would break the session and change nothing
durable, since the next container recreates them.

Sources 5 and 6 are **inert in this environment**: both are Slack-origin hooks that check for
env vars (`CCR_OPUS_REMINDER_HOOK_BODY`) that are unset here, so both exit immediately without
emitting anything.

Sources 7–13 are Anthropic-shipped skills. The `synced/manifest.json` marks every one as
`source: anthropic` or `anthropic-example` — none authored by the user. They also re-sync from
the account, so local deletion would not persist.

### Confirmed absent

Verified by filesystem sweep, not assumed:

- No `CLAUDE.md` anywhere — project, user, or enterprise scope
- No `CLAUDE.local.md`, `AGENTS.md`, `.cursorrules`, or `.mdc` rule files
- No project `.claude/` directory, no project `settings.json`
- No memory directory; `~/.claude.json` holds only OAuth identity, feature flags, and
  migration state — zero memory entries

---

## The removal: `.github/copilot-instructions.md`

Failed all three questions.

**1. Would the agent already do this without being told?** — Not applicable, and that is the
problem. The file issued no directives. It was context, not instruction: the user's original
prompt, the agent's clarifying questions, an 8-week phased plan, and a revised "fastest
practical start" section that contradicted the plan printed directly above it.

**2. Was it correcting a weakness the agent no longer has?** — No. It was never corrective.
Nothing in it addressed agent behavior at all.

**3. Did it conflict?** — Yes, and this was the real cost. It described a codebase that does
not exist:

| Described in the file | Actually in the repo |
|---|---|
| `backend/app/services/fastf1_service.py` | absent |
| `backend/app/core/tire_model.py`, `pit_strategy.py`, `safety_car.py`, `overtaking.py`, `weather_model.py`, `simulation_engine.py` | absent |
| `backend/app/api/routes/`, `backend/app/ml/` | absent |
| `frontend/` (Next.js, 8+ components) | absent |
| PostgreSQL, Redis, Docker Compose | absent |

The repo contains exactly two Python files: `backend/app/core/data_loader.py` and
`test_data_load.py`. Any agent reading the instruction file would assume roughly thirty modules
that were never written, and plan edits against paths that do not exist.

**Additional finding:** the file sat at the path GitHub Copilot auto-loads as repo-wide
instructions. It was therefore silently supplying a stale, self-contradicting 8-week plan to
any Copilot user in this repo *as if it were policy*. Claude never auto-loaded it — there is no
CLAUDE.md — so the exposure was Copilot-only.

The verbatim original is preserved at `removed/copilot-instructions.md` and remains in git
history.

---

## What survived, and what was cut

Four project decisions in the source were genuinely non-derivable from the code and are kept in
`KEEP.md`:

- **Jolpica-F1 over Ergast** — the highest-value line in the entire file. Ergast is deprecated;
  Jolpica is the maintained fork. An agent that did not know this would reach for Ergast by
  default and write against a dead API.
- **Simulation engine before ML** — a priority ordering, not inferable from code.
- **FastAPI + Next.js + FastF1 stack** — only the FastF1 half is visible in the code today.
- **XGBoost over neural nets** — a deliberate tradeoff for interpretability on sparse data.

Two candidates were **rejected on question 1** — the agent does these by default:

- *"Use time-series cross-validation to avoid future leakage."* Standard practice for any
  temporally ordered dataset. Telling the agent not to train on future races to predict past
  ones is telling it not to make an elementary error.
- *"Validate the simulation against historical races."* Validating a model against ground truth
  is not a rule; it is what modeling is.

Everything else — the week-by-week schedule, the ~30 speculative file paths, the duplicated and
contradictory milestone lists, the conversational scaffolding ("Excellent choices!", "Asked 4
questions") — was cut as narrative, not instruction.

---

## Note on the parameter estimates

The source file carried specific numbers: tire degradation rates per compound (SOFT
0.08–0.12 s/lap, MEDIUM 0.04–0.07, HARD 0.02–0.04), safety car probability ~1.5%/lap, VSC ~1%,
pit stop μ=22 s σ=1.5 s.

These were **not** kept as rules. They are unsourced estimates from a planning conversation, and
the project's whole premise is deriving such values from real FastF1 data rather than assuming
them. Hard-coding them into a rules file would freeze guesses into policy and pre-empt the
calibration work. They remain available in `removed/copilot-instructions.md` as a starting
sanity-check range if useful.
