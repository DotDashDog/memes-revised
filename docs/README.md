# Project Documentation

This folder is the **source of truth** for the meme-spread project's intent, design decisions, and current state. Files outside `docs/` (Python modules, notebooks, datasets, scratch) are *artifacts* — they implement decisions recorded here but they do not define them.

## Onboarding sequence (for any new agent)

Read in this order:

1. **`00-background.md`** — Project vision, scientific framing, prior work, key assumptions. Required.
2. **`01-scientific-rigor.md`** — Project-specific experimental and reporting standards. Non-negotiable. Required.
3. **`method-discipline.md`** — General anti-cargo-cult method discipline (how not to fool yourself). High-priority. Required.
4. **`02-state.md`** — Live snapshot: what's done, what's active, where data and models live.
5. **`03-environment.md`** — Python/uv conventions. Skim if familiar; consult before running any code.
6. **`work-packages/README.md`** — The work plan, dependency graph, and status table.
7. **`decisions/README.md`** — The ADR protocol and index of design decisions.
8. **`open-questions.md`** — Unresolved questions tracked over time.

Skim everything once. Re-read the specific work package and any cited decisions before each task.

## Folder layout

| Path | Purpose |
|---|---|
| `00-background.md` | Project vision, why SIR, prior work, key risks |
| `01-scientific-rigor.md` | Project-specific: baselines, validation, contamination, failure modes |
| `method-discipline.md` | General anti-cargo-cult method discipline: identifiability, pre-registration, plots-before-metrics, debugging |
| `02-state.md` | Live snapshot — updated after every substantive change |
| `03-environment.md` | Python/uv setup, commands, conventions |
| `decisions/` | Architecture Decision Records (ADRs). One file per decision. |
| `work-packages/` | Self-contained units of work. One file per package. |
| `research-notes/` | Distilled exploratory research not yet an ADR or work package. |
| `open-questions.md` | Questions surfaced but not yet resolved |

## Maintenance protocol

These rules apply to all agents (including the one that created this folder):

- **State stays current.** After completing any substantive work, update `02-state.md` and the relevant work package status. Stale state is worse than no state.
- **Decisions are explicit.** Any nontrivial design choice gets an ADR. Drafts are `status: proposed` — only the user moves an ADR to `accepted`.
- **Open questions are surfaced, not silently answered.** If you encounter a question, log it in `open-questions.md` even if you have a guess.
- **Atomicity.** A work package should be doable in a focused session and have a clear "done" condition. If yours grows, propose splitting it via an ADR.
- **No silent scope creep.** If work suggests changing another package, write it down — don't act.

## Conventions

- **File naming.** ADRs: `NNNN-short-slug.md` (zero-padded sequential). Work packages: `WP-NN-short-slug.md`.
- **Cross-references.** Use relative paths (e.g., `decisions/0003-reddit-only.md`).
- **Code citations.** When linking code from these docs, use `path:line-line` form so the IDE can jump.
- **Status fields.** Every ADR and work package has a `Status:` line in its header. Update it; don't leave it stale.
