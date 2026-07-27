# Supplement subreddits for the thin recent years

**Date:** 2026-07-26
**Status:** research note — feeds the ADR 0003 sub-question ("add other meme-style subs?") and the
`open-questions.md` entry on ADR 0011's split. **No decision is made here.**

## Why

r/memes fell 97.4% from its Aug 2020 peak. The post-training-cutoff eval window ADR 0004 needs is
therefore real but thin. If supplements are needed, they should be chosen on measured evidence, not
on listicles or recollection.

## Method

Two independent sources, neither of them our own dumps:

1. **The torrent manifest** of the 2025-12 Academic Torrents release — 39,981 subreddits with
   submissions files. Establishes *availability*: whether a candidate has downloadable history at
   all. (Lifetime file size is a poor volume proxy — most of r/memes' 2.82 GB is 2019–21 — so it was
   used only for availability and for surfacing candidates by name.)
2. **Arctic Shift** (`arctic-shift.photon-reddit.com`), Reddit-API-collected. Used for **exact
   24-hour post counts** and content-type classification on 2026-07-20.

Content type is derived from `post_hint` and `url` — the only usable fields (`is_self`, `is_video`,
`domain`, `is_gallery` are rejected by the API). "image%" counts `image` + `gallery`.

**Method caveat, learned the hard way:** an earlier pass sampled four one-hour windows and scaled
×6. That over-estimated by ~34% (r/memes: 384/day estimated vs **287** exact) because 12:00 and
18:00 UTC are peak hours. All figures below are exact full-day scans. A page cap also silently
truncated two 2020 counts at ~3,000 before being caught; those were re-run uncapped.

## Availability

Every candidate tested is present in the dump, so adding a subreddit adds its **entire history**,
not just recent years — which changes corpus composition across the whole timeline, not only in the
thin period.

## Measured: 2026-07-20, exact 24-hour counts

### Tier 1 — general image-meme, viable now

| Subreddit | posts/day | image % | image posts/day |
| --- | ---: | ---: | ---: |
| r/whenthe | 312 | 66.3% | 207 |
| **r/memes** *(baseline)* | **287** | **86.4%** | **248** |
| r/shitposting | 169 | 82.8% | 140 |
| r/196 | 167 | 81.4% | 136 |
| r/meirl | 147 | 91.2% | 134 |
| r/meme | 113 | 98.2% | 111 |
| r/me_irl | 102 | 77.5% | 79 |
| r/okbuddychicanery | 90 | 76.7% | 69 |
| r/antimeme | 80 | 95.0% | 76 |
| r/dankmemes | 77 | 96.1% | 74 |
| r/comedyheaven | 65 | 96.9% | 63 |

**Tier 1 excluding r/memes: ~1,089 image posts/day — 4.4× r/memes alone.**

### Tier 2 — topic-locked (own template populations)

r/HistoryMemes 75 (92.0%) · r/Animemes 60 (86.7%) · r/ProgrammerHumor 46 (93.5%) ·
r/lotrmemes 40 (87.5%) · r/ShitPostCrusaders 34 (82.4%) · r/goodanimemes 23 (91.3%) ·
r/MemePiece 23 (87.0%). Combined ~268 image posts/day.

These are real image-meme communities, but their memes are template reuse *within a niche*. Whether
that is a feature (crisp, well-separated clusters) or a confound (different spread dynamics,
different effective population) is untested.

### Excluded, with reasons

| Subreddit | posts/day | Why excluded |
| --- | ---: | --- |
| r/SipsTea | 190 | 49.5% image — mostly video |
| r/funny | 184 | 46.2% image — video + text heavy |
| r/Unexpected | 48 | **0% image** — entirely video |
| r/HolUp | 26 | 69.2% image, video-heavy for its size |
| r/ich_iel | 55 | German; the text pipeline (MiniLM/BGE/E5) is English |
| r/PoliticalCompassMemes | 42 | Event-driven political content — `open-questions.md` flags event-driven memes as having different dynamics |
| r/AnarchyChess | 38 | 65.8% image, in-joke dependent |
| r/madlads, r/terriblefacebookmemes, r/Memes_Of_The_Dank, r/DeepFriedMemes, r/suddenlycaralho | 0 | Dead |
| r/surrealmemes, r/trippinthroughtime | 3 | Effectively dead |

## The more interesting finding: activity migrated, it did not vanish

Exact 24-hour counts at three points (`experiments/WP-16/subreddit_migration.png`):

| Subreddit | Aug 2020 | Aug 2023 | Jul 2026 | 2026 vs 2020 |
| --- | ---: | ---: | ---: | ---: |
| r/memes | 12,158 | 1,531 | 287 | **2.4%** |
| r/dankmemes | 3,608 | 192 | 77 | **2.1%** |
| r/comedyheaven | 662 | 13 | 65 | 9.8% |
| r/meirl | 144 | 231 | 147 | **102%** |
| r/shitposting | 69 | 789 | 169 | **245%** |
| r/196 | — | 616 | 167 | new since 2020 |
| r/whenthe | — | 57 | 312 | new since 2020 |

The two classic giants collapsed ~98%. r/meirl held flat, r/shitposting grew 2.4×, and the two
largest contributors today (r/whenthe, r/196) **did not meaningfully exist in 2020**.

This matters more than the raw volume numbers: supplements are not merely "more data," they are
disproportionately data *where r/memes is weakest*. Reddit meme posting did not collapse — it
redistributed, and a r/memes-only corpus mistakes redistribution for extinction.

## What pooling does and does not buy

- Tier 1 pooled ≈ **1,337 image posts/day** vs r/memes' 248 → **5.4×**.
- But r/memes alone in Aug 2020 was ≈ 10,456 image posts/day. Pooling all of Tier 1 today still
  leaves recent years **~8× below** the 2020 baseline.

So pooling substantially relieves the ADR 0011 tension without dissolving it. It should be evaluated
as "does this clear the minimum viable cluster size for an identifiable SIR fit," not as "does this
restore 2020."

## Open hazards — none of these are resolved

1. **Cross-posting is unquantified and could be large.** These communities share content heavily; the
   same image often appears in several. For a *spread* model this is arguably signal — observing one
   meme's trajectory across communities is exactly the phenomenon of interest — but naive pooling
   would double-count it as independent observations. **Measurable now** with the project's existing
   `experiments/wp01/dhash.py`: sample one day across Tier 1, compute pairwise near-duplicate
   overlap. This should happen *before* any pooling decision.
2. **Pooling breaks SIR's population assumption.** ADR 0005 fixes $S(0) = 1 - I(0)$ on a normalized
   population. Subreddits differ by more than an order of magnitude in size and activity, so a pooled
   corpus has a heterogeneous, unmodelled $N$. Bears on ADR 0005, WP-08.
3. **Community style mixing** is the cost ADR 0003 already named. Tier 2 makes it acute.
4. **Adding a subreddit changes the whole timeline**, not just the thin years, so any existing split
   or baseline computed on r/memes alone is not comparable to one computed on a pooled corpus.
5. **Composition drift over time.** If the corpus is r/memes-dominated in 2020 and
   r/whenthe-dominated in 2026, then "year" and "community" are confounded — and the eval window
   sits entirely in the drifted region. This is a direct threat to any train/test comparison and is
   arguably the strongest argument for choosing supplements that existed across the whole period
   (r/meirl, r/dankmemes, r/comedyheaven) over the largest ones today (r/whenthe, r/196).

That last point cuts against the naive "pick the biggest current subs" reading of the table above,
and should be weighed before an ADR is drafted.

## Mitigations for cross-community heterogeneity

Added 2026-07-26 after discussion. These address hazards 1, 2 and 5 above. They are **options, not
decisions** — items 1, 3 and 4 touch ADR 0005, ADR 0011 and WP-08 and are carried into
`decisions/0016-multi-subreddit-corpus.md` (`proposed`).

**Reframe first.** Communities also drift internally, more smoothly than they differ from each
other. With a *single* community, calendar time and internal drift are perfectly confounded — there
is no contrast that separates "memes spread differently in 2026" from "r/memes became a different
place." Several communities drifting on different schedules make the two separable: drift shows up
at different calendar times per community, era shows up simultaneously across all of them. So
multi-subreddit does not introduce the confound of hazard 5; it is the only thing that makes the
*existing* one estimable. This weakens hazard 5 considerably, though it does not remove the
composition problem in the eval window.

1. **Population as an exposure term, not a constant.** ADR 0005 normalizes $S(0) = 1 - I(0)$, so
   fits are already in population *fractions*; the breakage under pooling is only that a raw count
   of 100 is a different fraction in each community-month. Convert counts to fractions of
   $N_{c,t}$ before fitting, with $N_{c,t}$ = unique authors active in community $c$ over the
   meme's window. `author` is already in the preserved column set (WP-16 §Method 4), so
   per-community-month author counts are a free by-product of the re-ingest. This is a
   normalization change, not an ADR 0005 rewrite.
2. **Prefer scale-free targets.** Branching ratio, decay timescale, peak *time*, and normalized
   trajectory shape are invariant to community size; absolute peak height is not. Note the
   asymmetry this creates: the Hawkes track (ADR 0014, WP-12–15) is structurally more robust here
   than SIR, since background intensity $\mu$ absorbs community activity and the branching ratio is
   already dimensionless. Caveat: normalizing a curve by its own peak discards magnitude, which may
   itself be a prediction target under ADR 0001.
3. **Hierarchical partial pooling** — per-community parameters under a shared prior, rather than
   one pooled fit or fully separate fits. Small communities shrink toward the population mean, large
   ones dominate their own estimates. This is the standard treatment for groups differing by an
   order of magnitude, and it is strictly better than either extreme on thin per-community data.
4. **Backbone + extension corpus split.** Use the subreddits present across the whole period
   (r/memes, r/dankmemes, r/meirl, r/comedyheaven) as a composition-matched **backbone**, so the
   temporal split is clean. Hold the post-2020 arrivals (r/whenthe, r/196, r/shitposting) out as a
   separate **community-generalization** eval rather than mixing them into the backbone. ADR 0011
   then becomes factorial instead of a single time cut:

   | | same community | new community |
   | --- | --- | --- |
   | **same era** | train | community generalization |
   | **later era** | temporal generalization | deployment condition |

   An observational confound cannot be removed, but each leg can be measured separately. This also
   answers ADR 0011's own open sub-question ("are there confounds beyond date, e.g. subreddit
   shift?") in the affirmative.
5. **Cross-posting is an identification strategy, not only a nuisance.** The same image in two
   communities is the same content observed in two populations — a *within-meme* estimate of the
   community effect, free of content confounding. This reframes hazard 1, and makes the dhash
   overlap measurement dual-purpose: it sizes the double-counting risk *and* determines whether the
   natural experiment has enough samples to be usable.

**Selection-bias trap to avoid.** Filtering clusters by a minimum size differentially drops small
communities and recent years, and that bias reads as a community effect. Apply one absolute
threshold everywhere and report per-community retention rates, or make the threshold a function of
community size and test sensitivity to it.

### Cheap diagnostics, in dependency order

| Diagnostic | Cost | Answers |
| --- | --- | --- |
| Unique authors per community-month | free from WP-16 | every $N_{c,t}$ for mitigation 1 |
| dhash near-duplicate overlap across Tier 1, one day | ~1 h, code exists (`experiments/wp01/dhash.py`) | hazard 1 magnitude; viability of mitigation 5 |
| Community-shortcut audit — predict subreddit from meme features | small | whether "community" is a leak the model can exploit |

None of these has been run.
