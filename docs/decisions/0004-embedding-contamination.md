# 0004 — Embedding-contamination strategy via time-cutoff

**Status:** accepted (principle); model-specific choice deferred to ADR 0006
**Date:** 2026-05-19
**Gates:** —

## Context

Modern embedding models are trained on web data that includes the post-meme internet. Even a meme that predates an embedder's training cutoff is described in articles, discussions, and aggregators that the embedder has read. The embedder therefore likely encodes some signal about which memes were viral, independent of their content. This was identified as failure mode #4 of the prior iteration and is more severe now than in 2023.

Mitigations considered:

1. Train an embedding model from scratch on a contamination-controlled corpus. **Rejected** as impractical.
2. Use a model trained on a known-clean corpus (e.g., Wikipedia-only). **Open** — requires literature survey, may be viable.
3. Use an older embedding model with a known training cutoff, and evaluate exclusively on memes that originated *after* that cutoff. **Accepted** as the principle.

## Decision

Adopt the **time-cutoff principle**: the project's evaluation set will consist exclusively of memes that originated after the chosen embedding model's training cutoff. Training data may span pre- and post-cutoff; evaluation must not.

Specific model selection — and the resulting cutoff date — are deferred to ADR 0006 after WP-01 completes.

Operational requirements:

- Every meme cluster must have a defensible "origin date" (first observation in the corpus, or earlier if known).
- The training/evaluation split is by *meme*, not by *post*, with the evaluation set strictly post-cutoff.
- Reported metrics include the train-vs-eval gap as a contamination signal.
- Where feasible, run a **contamination probe**: train a simple model to predict spread parameters from the embedding *without* the model ever seeing the $I(t)$ data. High accuracy from such a probe is evidence the embedding alone carries the virality signal.

## Alternatives considered

- **Contamination-controlled embedder from scratch.** Rejected — too expensive, corpora insufficient.
- **Wikipedia-only or other small-corpus embedders.** Open — listed as a research direction in `open-questions.md`.
- **Adversarial debiasing** at training time. Speculative — could be revisited as a research direction.

## Consequences

- The eval set is bounded by the embedding model's cutoff date. Older = smaller eval window.
- Older embedders are likely lower-quality. We accept a quality/contamination tradeoff.
- ADR 0011 (split strategy) must be consistent with this principle.

## Open sub-questions

- Are there contamination-controlled modern embedders worth evaluating? Tracked in `open-questions.md`.
- Is contrastive / adversarial debiasing viable for this project? Speculative, deferred.
- What is the "right" cutoff date given availability of post-cutoff memes? Tied to WP-01 / ADR 0006.
