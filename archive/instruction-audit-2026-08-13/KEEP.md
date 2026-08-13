# Project rules — kept set

Staged for review. Not active: nothing loads this file. To make it live, move it to
`CLAUDE.md` at the repo root.

---

- **Use the Jolpica-F1 API, not Ergast.** Ergast is deprecated; Jolpica is the maintained fork.
- **Build the simulation engine before the ML predictor.** Sim first, predictions second.
- **Stack is FastAPI (Python) + Next.js, with FastF1 as the session data source.**
- **Use XGBoost for the winner predictor, not a neural net** — interpretable, handles sparse
  historical data.

---

4 rules, 58 words of rule text. Distilled from a 1,203-word source — a 95% reduction.
See `README.md` for what was cut and why.
