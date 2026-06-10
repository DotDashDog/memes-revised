# 0011 — Train / test / eval split strategy

**Status:** proposed
**Date:** 2026-05-19
**Gates:** embeddings + clusters exist; embedding model chosen (ADR 0006)

## Context

`01-scientific-rigor.md` §2 mandates that the split be decided before any modeling. The split interacts strongly with ADR 0004 (contamination via time cutoff) and ADR 0009 (clustering, which defines "a meme").

Key constraints:

- Evaluation set must be entirely **post-embedding-cutoff** (ADR 0004).
- Split must be **at the meme/cluster level**, not the post level, to avoid leakage.
- Validation set (for hyperparameter tuning) needs to exist separately from the held-out test set.

## Decision (proposed)

Three-way split:

- **Train:** clusters whose origin date is `< T_train_max`.
- **Validation:** clusters whose origin date is in `[T_train_max, T_cutoff)` — pre-cutoff but later than training; used for hyperparameter selection and model checkpoint selection.
- **Test (the official eval set):** clusters whose origin date is `>= T_cutoff`.

`T_cutoff` is the chosen embedding model's known training cutoff. `T_train_max` is set so that the validation set has enough mass to be useful (likely 6–12 months before cutoff).

The test set is touched **once per real claim**. Hyperparameter sweeps run against validation.

## Alternatives considered

- **Random meme-level split.** Doesn't address contamination at all.
- **Pure time-based 80/10/10.** Doesn't anchor on the embedding cutoff; the eval set wouldn't be cleanly post-cutoff.
- **Per-cluster k-fold cross-validation.** Pros: more efficient use of small data. Cons: complicates contamination accounting.

## Consequences

- Need to define "origin date" precisely per cluster (first-post timestamp is the natural choice; revisit if it biases toward singletons).
- Test set size is bounded by post-cutoff meme volume; older embedders have smaller eval sets.

## Open sub-questions

- Should the validation set use rolling-time CV instead of a single block? Tied to total data volume.
- Are there confounds beyond date (e.g., subreddit shift) that need to be accounted for in the split? Probably worth checking.
