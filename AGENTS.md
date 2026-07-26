# Meme Spread Project — Agent Protocol

This is a long-running research project. The source of truth for project state, design decisions, and active work is in `docs/`, **not** in code or scratch files.

This file is the entry point for any AI agent (Cursor, Claude Code, Codex, etc.) working on this codebase. It is intentionally short; everything substantive lives in `docs/`.

## Before doing substantive work

You MUST read, in this order:

1. `docs/README.md` — navigation
2. `docs/00-background.md` — what we're trying to do and why
3. `docs/01-scientific-rigor.md` — non-negotiable project-specific experimental standards
4. `docs/method-discipline.md` — general anti-cargo-cult method discipline (how not to fool yourself); high priority
5. `docs/02-state.md` — current snapshot, active focus
6. `docs/03-environment.md` — Python/uv conventions (skim if you're already familiar)
7. The specific work package you're picking up in `docs/work-packages/`
8. Any decision records cited by that work package in `docs/decisions/`

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

- **Scientific rigor is non-negotiable.** Follow `docs/01-scientific-rigor.md` (project standards) **and** `docs/method-discipline.md` (anti-cargo-cult method discipline). In particular: define and run baselines, report uncertainty, fix and record random seeds, file negative results, match the test to the claim, and pre-register principled (not ritual) thresholds.

- **Don't trust embeddings or models silently.** Treat embedding contamination, identifiability, and selection bias as live concerns at every step, not afterthoughts.

- **Python is run through `uv`.** Every Python invocation goes through `uv run`. Every dependency goes through `uv add` / `uv remove`. Never invoke system `python` or `pip` directly, never `uv pip install` outside the lockfile. See `docs/03-environment.md`.

## Verifying scientific claims (anti-self-certification)

An agent can follow the letter of `docs/01-scientific-rigor.md` — even perform its pre-registration ritual — and still certify a bogus result. The author of a test is the worst-placed party to grade it. These rules are structural for that reason.

- **No self-certified passes.** An execution agent may compute metrics and *propose* a verdict, but must not write a PASS / CONFIRMED conclusion into the project record (`docs/02-state.md`, a work-package Outcome, or an ADR). The verdict is rendered by the orchestrating/planning agent or a separate review agent **after inspecting the underlying artifact** (the plot and raw numbers), not the summary.

- **Surface the criterion before the expensive run** whenever a diagnostic's result will enter the project record or inform a substantial planning choice. State the metric, the threshold, and *why that threshold would falsify the claim*, and stop for approval before computing. A threshold you cannot derive is not a test. (This gate does not apply to throwaway exploratory checks that won't be recorded or relied on.)

- **Make the result legible to the grader, and prefer plots.** A plot carries far more context than a scalar; a numerical diagnostic must clear a high bar before it is trusted over what a plot would show. For any recorded diagnostic, produce a plot and — *before the result is known* — write what it should look like if the hypothesis is true and what failure looks like. Route the plot plus that description to the user or a stronger review agent for the verdict. A number that "passes" while the plot disagrees is an open finding, not a pass. Be realistic about an agent's ability to read plots: when there is any doubt, escalate the plot to a human or a more capable reviewer rather than self-grading.

## When asking the user

When you need approval on a proposed decision, present (a) the decision under consideration, (b) the alternatives considered, (c) the evidence so far, and (d) what would change if the decision goes the other way. Don't ask vague questions.
