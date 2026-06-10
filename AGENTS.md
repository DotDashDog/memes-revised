# Meme Spread Project — Agent Protocol

This is a long-running research project. The source of truth for project state, design decisions, and active work is in `docs/`, **not** in code or scratch files.

This file is the entry point for any AI agent (Cursor, Claude Code, Codex, etc.) working on this codebase. It is intentionally short; everything substantive lives in `docs/`.

## Before doing substantive work

You MUST read, in this order:

1. `docs/README.md` — navigation
2. `docs/00-background.md` — what we're trying to do and why
3. `docs/01-scientific-rigor.md` — non-negotiable experimental standards
4. `docs/02-state.md` — current snapshot, active focus
5. `docs/03-environment.md` — Python/uv conventions (skim if you're already familiar)
6. The specific work package you're picking up in `docs/work-packages/`
7. Any decision records cited by that work package in `docs/decisions/`

If `docs/` does not exist or is malformed, stop and surface the problem.

## Hard rules

- **Do not make big design decisions unilaterally.** A "big decision" is any of:
  - A choice that affects another work package's scope, interface, or assumptions.
  - A choice of model, framework, dataset, embedding pipeline, or hyperparameter regime not already specified by a `status: accepted` decision record.
  - Any change to the project's success metric, validation strategy, or evaluation set.
  - Anything that would invalidate existing experimental results.

  For a big decision, draft a new ADR with `status: proposed` in `docs/decisions/`, then stop and ask the user. Do not implement until the ADR is marked `accepted`.

- **Stay in scope.** Work on one package at a time. If your work suggests changes to another package, log them as open questions or propose a new ADR — do not silently expand scope.

- **Update state as you go.** When you complete a work package or substep, update `docs/02-state.md` and the work package's status header. When you discover new open questions, add them to `docs/open-questions.md`.

- **Scientific rigor is non-negotiable.** Follow `docs/01-scientific-rigor.md`. In particular: define and run baselines, report uncertainty, fix and record random seeds, and file negative results in the work package log.

- **Don't trust embeddings or models silently.** Treat embedding contamination, identifiability, and selection bias as live concerns at every step, not afterthoughts.

- **Python is run through `uv`.** Every Python invocation goes through `uv run`. Every dependency goes through `uv add` / `uv remove`. Never invoke system `python` or `pip` directly, never `uv pip install` outside the lockfile. See `docs/03-environment.md`.

## When asking the user

When you need approval on a proposed decision, present (a) the decision under consideration, (b) the alternatives considered, (c) the evidence so far, and (d) what would change if the decision goes the other way. Don't ask vague questions.
