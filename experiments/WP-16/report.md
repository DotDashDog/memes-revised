# WP-16 — Full submissions re-ingest: report

**Date:** 2026-07-26
**Archive:** `memes_submissions.zst`, 2025-12 Academic Torrents cut (infohash `3e3f64de…`)
**Output:** `dataset_saves/reddit/submissions-2025-12/submissions.parquet` (586 MB, zstd, 48 row groups)
**Code:** `experiments/wp16/ingest.py` · **Machine-readable stats:** `ingest_report.json` alongside the parquet

## 1. Rows in, rows out, rows dropped

| | count |
| --- | ---: |
| Raw ndjson lines read | **11,782,247** |
| Rows written | **11,782,247** |
| Rows dropped | **0** |

`rows_in == rows_out + rows_dropped` is asserted by the ingester; a run that cannot reconcile exits non-zero.

**Why zero drops is the correct outcome here, not a red flag.** Every input line is a submission record and every one is retained. Filtering — non-image posts, deleted authors, low scores — is a downstream modelling choice, not an ingestion one, and doing it here would make the corpus's selection criteria invisible. The drop taxonomy (`bad_json`, `created_utc_missing`, `created_utc_unparseable`) exists and is counted; it simply never fired.

Independently corroborated three ways:

1. The count matches the **11,782,247** established earlier by full streaming decompress, computed by different code.
2. Re-counting submissions-per-month *from the parquet* and comparing against the raw byte-scan of the `.zst`: **188 months compared, 0 mismatched.**
3. Date range **2008-07-11 → 2025-12-31**, matching the archive.

### Coercions applied (counted, not assumed away)

| Coercion | count | note |
| --- | ---: | --- |
| `created_utc` from `float` | 761,217 | 2023+ records |
| `created_utc` from `str` | 32,179 | 2014–15 records |
| `url` was null | 14,843 | concentrated in 2022; stored as null `img_url`, not `"None"` |
| `url` unparseable | 241 | malformed brackets — `urlsplit` rejects as "Invalid IPv6 URL" |

The `created_utc` drift is worse than WP-16 anticipated: **three** types, not two — `int` (2008–13, 2016–22), `str` (2014–15), `float` (2023+) — with *mixed* types inside 2015 and 2023. A parser branching on year rather than value type would still break. Probe: `experiments/wp16/probe_schema.py`.

## 2. Submissions per month

Plot: `submissions_per_month.png` (also the coverage-completeness gate artifact — see WP-16 §"Gate result", PASSED and user-certified 2026-07-26).

Yearly totals, now spanning the full archive rather than stopping at 2020-03:

| Year | Submissions | | Year | Submissions |
| --- | ---: | --- | --- | ---: |
| 2008–2016 | 118,835 | | 2021 | 2,161,804 |
| 2017 | 84,835 | | 2022 | 821,276 |
| 2018 | 422,133 | | 2023 | 463,933 |
| 2019 | 2,958,995 | | 2024 | 310,545 |
| **2020** | **4,305,354** | | 2025 | 139,537 |

The corpus is severely front-loaded: **2019+2020 alone are 61.7%** of all submissions ever made to r/memes. This is the volume profile ADR 0011 must choose its split against — the naive "hold out the last 20% of the timeline" would yield an eval set of ~450k posts against a 7.3M-post training era, and those two regimes differ in kind, not just size.

## 3. Image URLs by host and year — input to the WP-17 link-rot estimate

| Year | Rows | Image-like | img % | i.redd.it | imgur | self/reddit | other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2011 | 1,245 | 262 | 21.0% | 0 | 331 | 40 | 704 |
| 2012 | 18,573 | 4,636 | 25.0% | 0 | 6,562 | 289 | 8,775 |
| 2013 | 29,944 | 11,553 | 38.6% | 0 | 15,934 | 321 | 9,240 |
| 2014 | 18,202 | 8,714 | 47.9% | 0 | 12,897 | 0 | 2,768 |
| 2015 | 15,522 | 5,973 | 38.5% | 0 | 10,796 | 0 | 2,459 |
| 2016 | 30,338 | 10,644 | 35.1% | 4,263 | 11,614 | 0 | 11,734 |
| 2017 | 84,835 | 62,328 | 73.5% | 55,115 | 11,055 | 0 | 15,867 |
| 2018 | 422,133 | 394,591 | 93.5% | 376,147 | 24,672 | 1,902 | 15,007 |
| 2019 | 2,958,995 | 2,792,286 | 94.4% | 2,746,160 | 53,911 | 121,294 | 29,924 |
| 2020 | 4,305,354 | 4,158,597 | 96.6% | 4,109,595 | 54,329 | 73,362 | 55,363 |
| 2021 | 2,161,804 | 1,904,363 | 88.1% | 1,872,129 | 30,764 | 173,718 | 73,101 |
| 2022 | 821,276 | 631,121 | 76.8% | 616,826 | 13,774 | 71,184 | 36,879 |
| 2023 | 463,933 | 255,175 | 55.0% | 250,765 | 3,750 | 81,997 | 58,181 |
| 2024 | 310,545 | 194,489 | 62.6% | 193,309 | 738 | 65,288 | 42,018 |
| 2025 | 139,537 | 122,590 | 87.9% | 121,905 | 315 | 793 | 13,850 |

Host buckets come from the top-15 hosts per year, which cover ≥97% of rows from 2017 on (84–91% in the sparse pre-2017 years).

**The headline for WP-17 is good news.** **10,346,214 image rows — 88% of the entire corpus — are `i.redd.it`**, Reddit's own CDN, where link rot is mild. The high-rot hosts are almost entirely historical: imgur totals **251,442** rows across all time and is concentrated before 2018, in years that contribute 0.6% of the corpus between them. Combined `imgur + other ≈ 627k`.

So the link-rot risk WP-17 was scoped around is **an order of magnitude smaller than the corpus**, and it correlates with exactly the years that matter least by volume. What it *does* correlate with is age — which is the selection hazard ADR 0015 names: dropping failed fetches silently would bias the corpus toward recent, surviving, Reddit-hosted posts. `fetch_status` remains mandatory.

**Two caveats.** "Image-like" is a host+extension heuristic, not a verified fetch — the true count is knowable only after WP-17 runs. And the 2023–24 dip to 55–63% is not link rot but composition: `self/reddit` rows (text posts) rise to 82k/65k in those years while `other` grows, i.e. r/memes carried proportionally more non-image content as volume collapsed. That interacts with ADR 0016's thin-eval-window problem and makes it slightly worse than the raw post counts suggest.

## 4. What changed versus the previous ingestion

| | 2025-01 run | this run |
| --- | ---: | ---: |
| Rows | 4,121,376 | **11,782,247** |
| Range | 2017-02-13 → 2020-03-11 | **2008-07-11 → 2025-12-31** |
| Internal gaps | 2 (118,314 rows) | **0** |
| Reconciled | no | **yes** |

**2.86× more data**, and the recovered span is precisely the region the project needs most: everything after 2020-03 — including the entire post-embedding-cutoff window ADR 0004 depends on.

The previous run's shortfall is fully diagnosed in WP-16 §"Step 1 result": it processed input chunks 0–2 of 10, and the legacy `save()` then split that partial result into all ten output files, leaving no trace of failure. The legacy ingester was rewritten rather than repaired.

## 5. Schema

Eight columns preserved exactly for `sample.py` / `run_semantic_embedding_cache.py`:
`is_comment, img_url, created_utc, author, score, link, title, text`

Five added — additive only, so no downstream reader can break:

| Column | Why |
| --- | --- |
| `post_id` | required to key ADR 0015's shard index; without it WP-17 cannot build one |
| `subreddit` | required by ADR 0016's multi-subreddit corpus |
| `num_comments` | an ADR 0007 engagement-signal candidate |
| `removed_by_category` | present from 2019; the only field that can address whether the 2020 peak was inflated by later-removed spam — the residual gap the coverage gate left open |
| `over_18` | cheap filter flag |

`author` was already in the preserved set, so **ADR 0016's $N_{c,t}$ exposure term is now computable** from this parquet with a group-by — no re-ingest needed.

## 6. Not done here

- **Comments** (`memes_comments.zst`, 106,464,277 records) — out of scope; whether to ingest at all is tied to ADR 0007.
- **Image fetching** — WP-17.
- **The old parquet is retained** at `dataset_saves/reddit/test_reddit_dataset/`. WP-01's results trace to it and it must not be deleted until anything depending on it is re-baselined.
- **Supplement subreddits are not ingested.** ADR 0016 is `proposed`, not accepted; this run covers r/memes only. The ingester takes the archive path as an argument and needs no changes to run against another subreddit's dump.
