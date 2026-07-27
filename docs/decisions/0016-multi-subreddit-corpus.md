# 0016 — Multi-subreddit corpus: composition, normalization, and split design

**Status:** proposed
**Date:** 2026-07-26
**Gates:** measurements in `research-notes/supplement-subreddits.md`; the three diagnostics in §Preconditions
**Amends:** ADR 0003 (resolves its "subs beyond r/memes?" sub-question), ADR 0005 (normalization basis), ADR 0011 (split structure — itself still `proposed`)
**Bears on:** WP-08, WP-16, ADR 0014

## Context

r/memes fell **97.6%** from its Aug 2020 peak (12,158 posts/day) to Jul 2026 (287/day). The
post-embedding-cutoff eval window ADR 0004 requires therefore exists but is thin, and thin in a
compounding way: fewer posts per meme means noisier $I(t)$, fewer clusters large enough for an
identifiable fit (cf. WP-02's low-count failure regime), and a much smaller viral tail.

Measurement (exact 24-hour Arctic Shift scans, 2026-07-20; full tables in the research note)
establishes the shape of the problem:

- **Activity migrated, it did not vanish.** r/dankmemes also fell ~98%, but r/meirl held flat,
  r/shitposting grew 2.4×, and the two largest general meme communities today — r/whenthe (312/day)
  and r/196 (167/day) — did not meaningfully exist in 2020.
- **Pooling Tier 1 gives ≈1,337 image posts/day vs r/memes' 248 — 5.4×** — but still ~8× below
  r/memes alone in Aug 2020 (≈10,456 image posts/day).
- The two subs ADR 0003 proposed by name both score poorly: r/dankmemes has collapsed *harder* than
  r/memes, and r/wholesomememes is at 9 posts/day.
- Every candidate is present in the Academic Torrents dump, so adding a subreddit adds its **entire
  history** — changing corpus composition across the whole timeline, not just the thin years.

The naive move — pool the biggest current subs — creates a worse problem than it solves. A corpus
that is r/memes-dominated in 2020 and r/whenthe-dominated in 2026 confounds **year with community**,
and the confound sits entirely inside the eval window.

**But the single-subreddit status quo is not confound-free either.** Communities drift internally as
well, more smoothly than they differ from one another. With one community, calendar time and
internal drift are perfectly confounded and there is no contrast that separates them. Several
communities drifting on different schedules make them separable — drift appears at different
calendar times per community, era appears simultaneously across all. Expanding the corpus does not
introduce this confound; it is what makes the existing one estimable. That is the central argument
of this ADR, and it inverts the reading the migration table invites.

## Decision (proposed)

Expand beyond r/memes, with four structural mitigations adopted together. They are a package: 1 and
3 without 4 still leaves the eval window composition-drifted, and 4 without 1 still pools
incommensurable magnitudes.

**1. Backbone + extension corpus.**

- **Backbone** — subreddits present across the full period: **r/memes, r/dankmemes, r/meirl,
  r/comedyheaven**. Composition-matched over time, so a temporal split on the backbone is clean.
- **Extension** — post-2020 arrivals: **r/whenthe, r/196, r/shitposting**, plus the remaining Tier 1
  subs. Reserved for community-generalization evaluation, *not* mixed into the backbone.
- Tier 2 (topic-locked: r/HistoryMemes, r/ProgrammerHumor, …) is **excluded from v1**. Their memes
  are template reuse within a niche, which is a different spread process, and admitting them makes
  ADR 0003's style-mixing cost acute for ~268 image posts/day.

**2. Population enters as an exposure term.** Counts are converted to fractions of $N_{c,t}$ before
fitting, where $N_{c,t}$ is the count of unique authors active in community $c$ over the meme's
observation window. This *preserves* ADR 0005's normalized $S(0) = 1 - I(0)$ rather than breaking
it: the pooling hazard is not that ADR 0005 is wrong, but that a raw count of 100 denotes a
different population fraction in each community-month. `author` is already in the column set WP-16
preserves, so $N_{c,t}$ is a by-product of the re-ingest.

**3. Hierarchical partial pooling.** Per-community parameters under a shared prior — not one pooled
fit, not fully separate fits. Communities differ by more than an order of magnitude in size; partial
pooling shrinks thin communities toward the population mean while letting large ones speak for
themselves.

**4. The split becomes factorial** (amending ADR 0011's single time cut):

| | same community | new community |
| --- | --- | --- |
| **same era** | train | community generalization |
| **later era** | temporal generalization | deployment condition |

The bottom-right cell is the honest deployment condition and should carry the headline number. The
two off-diagonal cells decompose *why* it degrades. An observational confound cannot be removed, but
each leg can be measured separately. This answers ADR 0011's own open sub-question ("are there
confounds beyond date, e.g. subreddit shift?") in the affirmative.

**5. Primary targets should be scale-free** where the science permits: branching ratio, decay
timescale, peak time, normalized trajectory shape. Absolute peak height is not scale-free and must
carry an explicit $N_{c,t}$ term if predicted at all.

### Preconditions — none of these have been run

This ADR should not be accepted until:

1. **Unique authors per community-month** are computed (free from WP-16), confirming $N_{c,t}$ is
   well-behaved — no zero or near-zero denominators, no discontinuities from mod-policy changes.
2. **Cross-posting overlap** is measured across Tier 1 for at least one day with
   `experiments/wp01/dhash.py`. If overlap is large, naive pooling double-counts; the same
   measurement determines whether the within-meme identification in §Consequences is viable.
3. **A community-shortcut audit** is run: can a classifier predict subreddit from meme features
   alone? Strong performance means "community" is a leak available to the model and the factorial
   split's off-diagonal cells will read optimistically.

## Alternatives considered

- **Stay r/memes-only.** Pros: no composition change; every existing result stays comparable; ADR
  0005 and ADR 0011 untouched. Cons: the eval window is ~8× thinner than the training era, time and
  community drift remain inseparable in principle, and the thinness hits exactly the regime where
  fits stop being identifiable. Rejected — the confound is present either way, and this variant is
  the one where it cannot be measured.
- **Pool all Tier 1 subs uniformly, largest-first.** Pros: maximum volume (5.4×), simplest to
  implement. Cons: maximally composition-drifted — the eval window would be dominated by
  communities absent from training. Rejected as the specific failure this ADR exists to prevent.
- **Backbone only, discard the new arrivals.** Pros: cleanest possible temporal comparison. Cons:
  discards the largest current communities and roughly half the available recent volume, and gives
  up any measurement of community generalization. Rejected as over-correction; the extension set
  costs nothing to *hold out* and answers a question the backbone cannot.
- **Composition re-weighting / subsampling to match mixes across eras.** Pros: directly equalizes
  composition. Cons: impossible where a community did not exist (r/whenthe in 2020 cannot be
  up-weighted from zero), and discards data in the period that is already thinnest. Rejected as
  strictly dominated by the backbone/extension design, which achieves the same matching without
  discarding.
- **Fixed per-community effects instead of hierarchical pooling.** Pros: simpler; no prior to
  justify. Cons: thin communities get noisy unregularized estimates, which is precisely the regime
  most of the extension set sits in. Rejected on the size distribution.

## Consequences

- **Cross-posting becomes an identification strategy, not only a nuisance.** The same image in two
  communities is the same content in two populations — a within-meme estimate of the community
  effect, free of content confounding. This reframes the double-counting hazard, contingent on
  precondition 2 showing enough overlap.
- **A minimum-cluster-size filter becomes a selection-bias hazard.** It differentially drops small
  communities and recent years, and that bias reads as a community effect. Use one absolute
  threshold everywhere and report per-community retention rates, or make the threshold a function of
  community size and test sensitivity.
- **Never report a pooled metric alone.** A pooled number is dominated by whichever community
  contributed most data. Report per-community results plus both weighted and unweighted averages.
- **WP-16 gains a deliverable**: per-community-month unique-author counts. This is cheap but must be
  specified before the ingest runs, not bolted on.
- **The Hawkes track (ADR 0014) is structurally advantaged** under this ADR — $\mu$ absorbs
  community activity, the branching ratio is already dimensionless — while SIR needs mitigation 2 to
  survive pooling. That is a reason to weight the tracks differently, not a reason to drop SIR, but
  it should be stated rather than discovered later.
- **Existing r/memes-only baselines are not comparable** to anything computed on the pooled corpus.
  WP-01's conclusions stand on the data they used; any re-run re-baselines and must say so.
- **Ingest and storage cost rise roughly 3–4×** over r/memes alone. Under ADR 0015 this is
  additional time-partitioned shards, not a rebuild.

## Open sub-questions

- **Is $N_{c,t}$ best measured as unique authors, unique commenters, or total posts?** Unique authors
  is the closest analogue to a susceptible population and is free from the submissions ingest, but
  the "susceptible" set arguably includes lurkers who could reshare and never do. Untested.
- **What observation window defines $N_{c,t}$ per meme** — the meme's own span, or a fixed window?
  A meme's span is endogenous to its own spread, which risks circularity.
- **Does the backbone have enough recent volume on its own?** Backbone image posts/day in Jul 2026
  ≈ 519 against r/memes' 2020 ≈ 10,456. The factorial design is only useful if the
  same-community/later-era cell is populated enough to fit.
- **Should the extension set contribute to training at all** once community generalization has been
  measured once? Tempting for volume, but it spends the only clean test of that axis.
- Tier 2 is excluded from v1, not rejected permanently. Revisit if backbone volume proves
  insufficient — but as a separate population with its own $N_{c,t}$, never merged.
