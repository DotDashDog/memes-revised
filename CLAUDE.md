# CLAUDE.md

Claude Code auto-loads `CLAUDE.md`, **not** `AGENTS.md`. This file exists so that the
tool-neutral protocol in `AGENTS.md` actually reaches Claude Code. `AGENTS.md` remains the
single source of truth — it is imported below, not duplicated.

@AGENTS.md

---

## Claude Code specifics

Everything above is the binding protocol. The notes below are orientation for this tool
and for traps in the repo layout.

### Who grades what

`AGENTS.md` §"Verifying scientific claims" requires that the party who produced a result
never certifies it. In Claude Code that maps as:

- **Main session = orchestrator/grader.** It may render a PASS/CONFIRMED verdict into the
  project record only after inspecting the underlying artifact (the plot and raw numbers),
  not the subagent's summary — and with user concurrence on consequential claims.
- **Agent-tool subagents = executors.** They compute metrics and *propose* a verdict,
  marked UNCERTIFIED. They never write a verdict into `docs/`.
- **No subagent in play?** Then you are the proposer, and the **user** is the grader. Do
  not self-certify because you happen to be the only agent in the session.

### Repo layout traps

- **`experiments/wpNN/` (lowercase) is code. `experiments/WP-NN/` (uppercase) is run
  output.** This volume is case-sensitive, so both exist side by side — e.g.
  `experiments/wp12/hawkesn.py` (source, tracked) vs `experiments/WP-12/bridge_results.json`
  (artifacts, gitignored). Check the case before editing; they are different directories.
- **`rigor-port/` is a portable template** for exporting this rigor framework to *other*
  projects. It is not this project's live protocol and its `AGENTS.md` is not in force here.
  The live protocol is `/AGENTS.md` + `docs/`.
- **`.cursor/rules/ocr-research.mdc` is archived research output** (a Gemini Deep Research
  dump on meme OCR), not a rule. It is `alwaysApply: false`. Read it as background if
  working on OCR; do not treat it as instructions.
- **Legacy code predates the current framework.** `memes_base/`, `train_script.py`,
  `test_ocr.py`, `testing.py`, `fakedatagen.py`, `embedding.py`, and the loose scripts at
  the repo root are from the 2023–2024 iteration. Per `docs/02-state.md`, do not assume they
  are correct or current; read before reusing.

### Environment

Every Python invocation goes through `uv run`; every dependency through `uv add` / `uv remove`.
Never system `python`/`pip`, never `uv pip install`. Full conventions in `docs/03-environment.md`.

### Secrets

`.env` holds `OPENAI_API_KEY`. It is gitignored and untracked. Do not print, echo, or copy
its contents into files, logs, or commits.
