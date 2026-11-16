# T-051 — researcher — AI/ML ideas

**Status:** done
**Next owner:** orchestrator — pick the next step; recommended is a new task for backend-engineer: MCP server (see note, 5-step plan).

## What I did
- Researched MCP, RAG, Claude API tool use, RL/classical ML for F1 strategy, OpenF1 live access. Ranked 9 ideas.

## Files changed
- team/research/T-051-ai-ml-ideas.md — full note with sources and the build plan

## Verification
- Opened: MCP python-sdk README, Claude tool-use docs, openf1.org, RSRL abstract. Others from search snippets only (marked unverified).
- Did NOT verify: FIA reg URLs, API pricing, Heine and Thraves paper, conformal tutorial, MCP class name in the version pip will install.

## Interface / contract changes
- none

## Requests for other roles
- backend-engineer (if approved): MCP server plus `forced_sc_lap` parameter; ask them to check `MCPServer` vs `FastMCP` naming.

## Open questions / risks
- Embedding provider and Anthropic API key budget (only for RAG / in-app LLM features).
- LLM layer will repeat engine weaknesses (single-car, ~1 stop); tool outputs should carry caveats.
