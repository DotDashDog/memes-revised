# 0008 — Non-viral baseline data source

**Status:** proposed
**Date:** 2026-05-19
**Gates:** —

## Context

Every meme in r/memes is already selected for at least minimal virality (someone posted it, someone upvoted enough to keep it visible). The "non-viral counterfactual" is missing. Without low-virality examples in our training set, we cannot learn what content fails to spread — only how content varies among the already-successful.

## Decision (proposed)

Compose the corpus from three sources to ensure low-virality coverage:

1. **r/memes (primary):** the bulk of the dataset, including low-upvote and singleton posts. Singleton clusters are *expected* and represent failed memes.
2. **Adjacent meme subs (optional):** r/dankmemes, r/wholesomememes, etc., to broaden style coverage. Decision deferred — see open sub-question in ADR 0003.
3. **Non-meme image-posting subs (controls):** subreddits with a similar posting cadence but where content is not selected for spread (e.g., r/pics, r/mildlyinteresting). These provide near-zero-virality content for the model to contrast against.

A cluster from a control sub that has only one post — or a small handful — is a legitimate training example with low $(\beta, s)$.

## Alternatives considered

- **r/memes only, with low-upvote inclusion.** Less data variety; still subject to selection.
- **Broad sweep across top-N subs.** More noise; harder to interpret what "spread" means across very different posting cultures.
- **Synthetic non-viral examples** (e.g., randomly-paired images and text). Unrepresentative of real failure modes.

## Consequences

- The clustering step must be applied across the combined corpus.
- Posts from non-meme subs may form many tiny clusters; this is fine.
- The embedding pipeline must work on non-meme images, not just text-on-image memes.
- The interest-over-time signal (ADR 0007) must normalize per-subreddit if controls have very different activity levels.

## Open sub-questions

- Which control subreddits, specifically? Should be chosen by activity-cadence similarity to r/memes, not by content.
- Do we treat clusters spanning multiple subs as one meme (true if same content), or per-sub? Tied to ADR 0009.
