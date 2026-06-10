# 0003 — Reddit as the initial (and currently only) platform

**Status:** accepted
**Date:** 2026-05-19
**Gates:** —

## Context

Memes spread across many platforms (Twitter/X, TikTok, Instagram, Discord, Reddit, etc.). Modeling cross-platform spread would be the scientifically ideal target. However, bulk-data access for those other platforms is impractical at the volumes this project needs.

## Decision

Restrict the v1 of this project to **Reddit only**, starting with **r/memes** and expandable to additional subreddits if needed. We have a complete post archive available; we know its shape and quirks; and the data is contiguous over time.

We acknowledge that "Reddit spread" is not the same as "internet spread." That is a tradeoff we accept in exchange for tractable data.

## Alternatives considered

- **Twitter/X.** Vastly larger meme signal, but bulk access is restricted post-2023 and prohibitively expensive at scale.
- **Multi-platform aggregation** (e.g., Reddit + Hacker News + 4chan archives). Could be added later but introduces normalization issues.
- **Cross-platform spread modeling.** Worthwhile if data appears; not currently feasible.

## Consequences

- The trained model will predict **Reddit-shaped spread** specifically. Generalization to other platforms is out of scope.
- The interest-over-time signal is "posts in cluster across r/memes-style subreddits," not "true meme exposure."
- Subreddit-specific phenomena (mod rules, repost culture, karma farming, subreddit size growth) are confounds we must address. See ADR 0007.
- If at some future point bulk data from another platform becomes accessible, an ADR will revisit this.

## Open sub-questions

- Should the v1 dataset include subs beyond r/memes (e.g., r/dankmemes, r/wholesomememes) to broaden meme coverage? Or does that complicate clustering by mixing community-specific styles?
- For the non-viral baseline (ADR 0008), we likely need non-meme subs. Which ones?
