# Decisions (ADRs)

This folder contains Architecture Decision Records — one Markdown file per nontrivial design decision.

## Protocol

- **Anyone (agent or user) can draft a new ADR.** Drafts start with `Status: proposed`.
- **Only the user can mark an ADR `accepted`.** Agents may *propose* and *implement* but must wait for the user's explicit acceptance before treating a decision as binding.
- **`accepted` ADRs are binding.** If you disagree with one, draft an amending or superseding ADR — do not silently work against it.
- **`superseded` ADRs are preserved.** Link the superseding ADR. Never delete history.

## ADR life cycle

```
proposed  →  accepted  →  (later)  superseded
   ↓
rejected
```

- `proposed` — A draft awaiting user approval.
- `accepted` — Binding on all subsequent work.
- `rejected` — User declined; preserved for record.
- `superseded` — Replaced by a later ADR (link it).

## When to write an ADR

Write one whenever you make (or want to make) a decision that:

- Constrains future work packages.
- Selects between meaningfully different alternatives (a different default would lead to different downstream code or experiments).
- Encodes a tradeoff worth remembering.
- Could plausibly be revisited later.

Small implementation choices (variable names, file layout within a package) do not need ADRs.

## Index

| # | Title | Status | Gates |
|---|---|---|---|
| [0001](0001-success-metric.md) | Target success metric | accepted | — |
| [0002](0002-sir-as-empirical.md) | SIR as empirical compression | accepted | — |
| [0003](0003-reddit-only.md) | Reddit-only initial platform | accepted | — |
| [0004](0004-embedding-contamination.md) | Embedding contamination via time-cutoff | accepted (principle) | specific model in 0006 |
| [0005](0005-sir-initial-conditions.md) | SIR initial conditions $S(0)=1-I(0)$, $R(0)=0$ | accepted | — |
| [0006](0006-ocr-vs-vlm.md) | OCR pipeline vs. VLM-native embedding | accepted | WP-01 |
| [0007](0007-engagement-signal.md) | Posts → $I(t)$ signal construction | proposed | — |
| [0008](0008-non-viral-source.md) | Where non-viral memes come from | proposed | — |
| [0009](0009-clustering.md) | Clustering algorithm and granularity | proposed | embeddings exist |
| [0010](0010-sir-parameterization.md) | SIR parameterization for prediction | accepted | WP-02 |
| [0011](0011-train-test-split.md) | Train/test/eval split strategy | proposed | embeddings + clusters exist |
| [0012](0012-baselines.md) | Canonical baseline set for first model run | proposed | embeddings + targets exist |
| [0013](0013-jax-primary.md) | JAX as the primary numerical backend | accepted | — |

## Template

Copy this into a new file `NNNN-short-slug.md`:

```markdown
# NNNN — <Title>

**Status:** proposed
**Date:** YYYY-MM-DD
**Gates:** <work packages or other ADRs this depends on, if any>

## Context

<What problem prompted this decision? What constraints apply? Cite relevant background.>

## Decision

<The decision in one or two sentences, then any necessary elaboration.>

## Alternatives considered

- **Option A:** <description>. Pros: ... Cons: ...
- **Option B:** <description>. Pros: ... Cons: ...
- **Option C (chosen):** <description>. Rationale: ...

## Consequences

<What does this decision unlock or constrain? What follow-up work is required?>

## Open sub-questions

<Things left undecided by this ADR that may need a future one.>
```
