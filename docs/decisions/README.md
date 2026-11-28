# Decisions (ADRs)

This folder contains Architecture Decision Records — one Markdown file per nontrivial design decision.

## Protocol

- Anyone can draft an ADR; drafts start as `proposed`. Only the user marks one `accepted`.
- `accepted` ADRs are binding. To change one, draft an amending or superseding ADR.
- `rejected` and `superseded` ADRs stay in the folder, with a link to whatever replaced them.

Write one for a decision that constrains later work packages or chooses between
alternatives that lead to different downstream code or experiments. Implementation details
do not need one.

## Index

| # | Title | Status | Gates |
|---|---|---|---|
| [0001](0001-success-metric.md) | Target success metric | accepted | — |
| [0002](0002-sir-as-empirical.md) | SIR as empirical compression | accepted | — |
| [0003](0003-reddit-only.md) | Reddit-only initial platform | accepted | — |
| [0004](0004-embedding-contamination.md) | Embedding contamination via time-cutoff | accepted (principle) | specific model in 0006 |
| [0005](0005-sir-initial-conditions.md) | SIR initial conditions $S(0)=1-I(0)$, $R(0)=0$ | accepted | — |
| [0006](0006-ocr-vs-vlm.md) | OCR pipeline vs. VLM-native embedding | accepted, amended by 0017 and 0018 | WP-01 |
| [0007](0007-engagement-signal.md) | Posts → $I(t)$ signal construction | proposed | — |
| [0008](0008-non-viral-source.md) | Where non-viral memes come from | proposed | — |
| [0009](0009-clustering.md) | Clustering algorithm and granularity | proposed | embeddings exist |
| [0010](0010-sir-parameterization.md) | SIR parameterization for prediction | accepted | WP-02 |
| [0011](0011-train-test-split.md) | Train/test/eval split strategy | proposed | embeddings + clusters exist |
| [0012](0012-baselines.md) | Canonical baseline set for first model run | proposed | embeddings + targets exist |
| [0013](0013-jax-primary.md) | JAX as the primary numerical backend | accepted | — |
| [0014](0014-hawkes-track-scope.md) | Neural-Hawkes track: scope & architecture-critical decisions | proposed | informs WP-12…WP-15 |
| [0015](0015-storage-layout.md) | Storage layout: NVMe working set, portable-drive archive + tar shards | accepted | — |
| [0016](0016-multi-subreddit-corpus.md) | Multi-subreddit corpus with cross-community normalization | proposed | three preconditions in the ADR |
| [0017](0017-no-title-in-clustering-embedding.md) | Post title/body excluded from the meme-identity embedding | accepted | — |
| [0018](0018-separate-image-and-text-channels.md) | Separate image and text channels; sentence model for the text | accepted | — |

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
