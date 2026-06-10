# 0007 — Engagement-signal construction (posts → $I(t)$)

**Status:** proposed
**Date:** 2026-05-19
**Gates:** none structural, but should be informed by data inspection

## Context

We need to turn a cluster's stream of timestamped Reddit posts (each with an upvote count and a comment count) into an interest-over-time curve $I(t)$ that we then fit with SIR.

Open design choices:

- **Time-bin size** for $I(t)$ (hour? day? adaptive?).
- **Upvote weighting**: linear, log, rank-normalized, or none.
- **Comment weighting**: as engagement signal, or as separate input to embedding, or both.
- **Subreddit-size normalization**: r/memes grew from ~hundreds-of-thousands to tens-of-millions over the corpus span. Raw post counts conflate meme intrinsic spread with sub growth.
- **Reposts / duplicates**: same image posted multiple times by different users — treat as multiple infections, or deduplicate, or upweight?

## Decision (proposed)

Define $I(t)$ as a **share-of-activity** signal:

$$\hat I(t) = \frac{\sum_{i \in \text{cluster}} w(\text{upvotes}_i, \text{comments}_i) \cdot \mathbf{1}[t_i \in \text{bin}(t)]}{N(\text{bin}(t))}$$

where $N(\text{bin}(t))$ is total subreddit activity (all posts in the same time window). Specific choices:

- Time bins: daily, with later aggregation to weekly for fitting if SIR fits are noisy.
- $w(u, c)$: provisionally $\log(1 + u) + \alpha \log(1 + c)$ with $\alpha$ small or zero.
- Duplicate posts: kept as separate observations (they *are* re-spread events on Reddit).

## Alternatives considered

- **Raw post count.** Conflates meme intrinsic with sub growth.
- **Upvotes-summed.** Heavily skewed by 1-2 viral posts; biased by sub size and karma dynamics.
- **Posts per million sub-members.** Requires reliable historical subscriber counts, which Reddit doesn't expose perfectly.

## Consequences

- A "subreddit total activity" time series must be precomputed.
- The fitted SIR amplitude $s$ now means "share-of-activity scale," not "fraction of population infected." This is fine under ADR 0002.
- Reposts increase signal — which we want, but it also means meme-templates with many derivative posts will look bigger than single-instance virals.

## Open sub-questions

- The exact $w(u, c)$ form should be informed by data inspection; this ADR proposes a default to start.
- Should we run a sensitivity analysis across different $w$ choices? Yes, planned as part of the SIR-fit work package when written.
- Should comments contribute *only* to the embedding (as audience reaction text) and not to the signal? An ablation will tell us.
