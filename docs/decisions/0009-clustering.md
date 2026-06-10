# 0009 — Clustering algorithm and granularity

**Status:** proposed
**Date:** 2026-05-19
**Gates:** embeddings produced (ADR 0006 / WP-01 outcome)

## Context

Clusters in embedding space define what counts as "a meme" for this project. Different clustering algorithms — and different hyperparameters within an algorithm — give different granularities, with cascading effects on $I(t)$ shape and SIR parameters.

This is one of the highest-leverage and least-constrained choices in the pipeline.

## Decision (proposed)

Use a **UMAP → HDBSCAN** pipeline as the primary clustering approach, run at **multiple granularities** (HDBSCAN `min_cluster_size` values) and reported in parallel through downstream experiments. Validation:

- A **hand-labeled gold set** of ~50 known memes (~30 posts each) — built before clustering is finalized.
- **Cluster purity** and **recall** measured against the gold set.
- **Eyeball inspection** of random clusters at each granularity (see `01-scientific-rigor.md` §8).

Pick the granularity that maximizes a defined balance of purity, recall, and downstream signal strength. **Don't pick a single granularity blindly** — keep the multi-granularity ablation as part of the project.

## Alternatives considered

- **k-means.** Bad fit for variable-density data in high-d; rejected as primary.
- **Direct HDBSCAN on raw embeddings.** Slow and unstable in high-d; UMAP preprocessing standard.
- **Hierarchical clustering with explicit cut levels.** Could be a useful complement; treat as a secondary algorithm to try.
- **Embedding-model-native clustering** (some models expose this). Worth checking once a model is chosen.

## Consequences

- The pipeline supports running with a configurable granularity parameter, not a fixed clustering.
- Storage and compute footprint increases ~linearly with the number of granularities evaluated.
- The downstream prediction model must be retrained for each granularity studied.

## Open sub-questions

- How exactly to construct the gold set? Tracked in a future work package.
- Should we report final results at one chosen granularity or as a function of granularity? Likely both.
- Does the holistic point-process approach in `open-questions.md` obviate the need for explicit clustering? Long-term question.
