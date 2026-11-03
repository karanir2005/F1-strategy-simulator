# T-049 — researcher — tyre wear model audit

**Status:** done (read-only; nothing run, no shell)
**Next owner:** data-scientist — run the cheap diagnostics in step 1 of the note, then build a caps-free evaluation and a durability parameter per compound/C-number (after T-047).

## What I did
- Audited tire_model.py, optimizer.py, T-043 handoff, track_params, laps CSV header. Full ranked list in `team/research/T-049-tyre-wear-audit.md`.
- Key result: linear wear with equal slopes (~0.037 s/lap) means the optimizer cannot prefer 2 stops (needs ~0.09 s/lap, my arithmetic: 1-stop 28/28 wear 30.0 s vs 2-stop 20.4 s, pit loss 24.2 s) and splits symmetrically (H28/M28). Teams' M24/H32 implies M wear about 1.33x H under linear wear. Caps (team-derived) currently supply all structure.
- Ranked: (1) linear equal wear confirmed, (2) stint censoring likely, (3) compound inversion likely a race-phase/traffic confound, not proven, (4) C-number/track pooling confirmed structurally, (5) fuel/wear weak identification likely, (6) cliff placeholder non-binding confirmed, (7) omissions: track temp, warm-up, traffic.

## Files changed
- team/research/T-049-tyre-wear-audit.md — new
- team/handoffs/T-049-researcher.md — new

## Requests / open
- data-scientist: diagnostics listed in the note (not computed by me): early-stint-only slope vs full, fuel +/-0.01 sensitivity, current cliff_age/max_reliable_age, laps beyond typical stint length.
- Unverified: Heilmeier equations and parameter values, 2026 fuel s/kg and kg limit, any Pirelli 2026 deg numbers, which of the two temperature columns is track temp.
