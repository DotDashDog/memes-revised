# WP-16 — Full submissions re-ingest

**Status:** active
**Date opened:** 2026-07-26
**Owner:** unassigned
**Gates:** ~~the 2025-12 archive download~~ done 2026-07-26; ~~coverage-completeness gate~~ **PASSED** 2026-07-26
**Unlocks:** WP-05, WP-06, WP-07; a corpus-wide view for the ADR 0011 split and the ADR 0009 clustering granularity question

## Goal

Replace the partial 2025-01 ingestion with a complete one, **against the 2025-12 archive cut rather than the 2023-12 one**, so the archive is ingested exactly once.

Context: the local 2023-12 `memes_submissions.zst` is verified complete (11,332,165 submissions, 2008-07-11 → 2023-12-31), but only 4,121,376 rows ending 2020-03-11 were ever processed. Rather than re-ingest that and then repeat the work when the newer cut lands, the newer cut is being fetched first (decision 2026-07-26, superseding the earlier "re-ingest now, upgrade later" sequencing).

Metadata only. Images are explicitly out of scope (see below).

## Coverage-completeness check (required, not optional)

The 2025-12 r/memes files are only **7–9% larger** than the 2023-12 ones (submissions 2.63 → 2.82 GB, comments 8.33 → 9.11 GB) despite covering two additional years. If posting volume had held at its 2022–2023 rate, two more years should add substantially more than that.

The likely cause is that **Reddit revoked Pushshift's API access in May 2023**, so post-lockdown records are collected by a different route and may be systematically less complete. This is not a curiosity — it is a direct threat to the project:

- If capture rates drop after mid-2023, a 2024–2025 meme looks less viral than an identical 2019 meme purely because fewer of its posts were recorded. That corrupts $I(t)$ (ADR 0007) and any comparison across the boundary.
- ADR 0011's train/test split, and ADR 0004's contamination cutoff, both want to put the eval window in exactly the post-2023 region where this risk lives.

**So the submissions-per-month histogram is a gate, not a deliverable.** If monthly volume shows a step change at mid-2023 that is not plausibly organic, the post-lockdown period must not be treated as interchangeable with earlier data, and ADR 0004/0011 need revisiting before an eval set is built on it.

### Gate result (2026-07-26) — **PASSED**

**Verdict: PASS. Certified by the user 2026-07-26**, after inspecting `experiments/WP-16/submissions_per_month.png` and the counts below. Per `AGENTS.md` §"Verifying scientific claims" the agent was the proposer here (no subagent in play), so the user is the grader; this is not a self-certification.

**What the pass licenses:** the post-2023 period may be treated as interchangeable with earlier data for the purposes of ADR 0004's cutoff and ADR 0011's split. The gate's blocking condition — a non-organic step change at the May 2023 API revocation — did not occur.

**What it does not license:** the 2020 peak remains independently unverified (see the cross-check's limits below), and the *thinness* finding stands as a separate, unresolved problem — passing this gate says the recent data is real, not that there is enough of it.

The check was run ahead of the ingest, since both archives were already local. **Measurements** (`experiments/WP-16/submissions_per_month.png`):

1. **The overlap is exact.** Every year 2008–2023 has *identical* counts in the 2023-12 and 2025-12 cuts. The newer dump is a faithful superset; nothing was retroactively recollected.
2. **No discontinuity at the API lockdown.** Monthly volume across May 2023: Mar 36,627 · Apr 33,487 · **May 31,279** · Jun 31,948 · Jul 40,867 · Aug 41,251. Volume *rose* through the second half of 2023.
3. **The decline is long-running and predates the lockdown by ~2.5 years.** Peak Aug 2020 at 415,953/month; yearly totals 2020: 4,305,354 → 2021: 2,161,804 (−49.8%) → 2022: 821,276 (−62.0%) → 2023: 463,933 (−43.5%) → 2024: 310,545 (−33.1%) → 2025: 139,537 (−55.1%). The steepest drops are *before* May 2023, not after.

**Reading (certified — see verdict above):** the collection-completeness worry is not supported. The shortfall is organic decline in r/memes posting, not a data-collection artifact. On the log axis a constant percentage decline is a straight line, and the curve is close to one throughout, with no step at the lockdown.

### Independent cross-check (2026-07-26)

The above was measured entirely from the dumps, which is circular: **the no-step-at-May-2023 test rules out a sudden collection failure but not a gradual one**, and gradual capture decay is indistinguishable from organic decline when you only have the archive. So the decline was re-checked against an outside source.

Source: **Arctic Shift** (`arctic-shift.photon-reddit.com`), whose documentation states *"All data is retrieved through the official reddit API,"* standardized from **2023-04** onward, with earlier material referred to as "previous pushshift dumps (2023-03 and earlier)." Samples on or after 2023-04 are therefore genuinely independent of our dumps; earlier ones share lineage and are **not** evidence. Method: **exact full-day post counts** per reference day. Reddit's own JSON endpoints return 403 and were unusable; subredditstats.com stopped collecting in 2023-12 and now serves empty series.

| Date | Dump (posts/day) | Arctic Shift (posts/day) | Ratio | Independent? |
| --- | --- | --- | --- | --- |
| 2020-08-15 | 12,158 | 12,158 | 1.00 | no — same lineage |
| 2022-08-15 | 1,690 | 1,690 | 1.00 | no — same lineage |
| **2024-08-15** | **714** | **714** | **1.00** | **yes** |
| **2025-08-15** | **378** | **378** | **1.00** | **yes** |
| 2026-07-20 | *beyond dump* | 287 | — | yes |

All four overlapping days agree **exactly**, both independent ones included. Reproduce with `experiments/wp16/arctic_shift.py`; plot: `experiments/WP-16/decline_crosscheck.png`.

**Read the exactness carefully.** Four-for-four agreement to the post is *too* clean for two genuinely separate collection efforts — it more likely means Arctic Shift's post-2023-04 pipeline recovers essentially everything the API exposes, and so does the dump, than that two independent samplers coincidentally matched. It is strong evidence against the under-collection hypothesis (the failure mode the gate was built to catch), and it is *not* evidence that either source captures posts Reddit itself never served — deletions before either crawl are invisible to both.

**Method correction (2026-07-26).** An earlier pass sampled four one-hour windows (00/06/12/18 UTC) and scaled ×6, giving ratios of 1.07 and 0.81 and a 2026 rate of 384/day. That method over-estimates by ~34% because 12:00 and 18:00 UTC are peak hours. The exact scans above supersede it, and they are a considerably stronger result — 1.00 rather than "within sampling error." A page cap in the same tooling also silently truncated two 2020 counts at ~3,000 before being caught by comparison against a known dump value; those were re-run uncapped.

**What this does and does not establish.** It rules out the specific failure mode that mattered — the dump is not under-collecting recent years, so the low recent counts are real. It does **not** independently verify the 2020 peak, since no non-pushshift source for that period was reachable. That residual gap is small, because collection failures produce under-counts, not over-counts; the one way the peak could be inflated is mass bot/spam submissions later removed, which the dump would capture at creation time. That remains unexamined.

A bonus datapoint: 2026-07-20 sits at **287 posts/day**, *below* Aug 2025's 378 — the decline is still running, not flattening. (An earlier version of this entry claimed it had flattened, on the strength of the discredited 384 figure. Withdrawn.)

**But the measurement surfaces a different problem, which is arguably worse for the project.** r/memes has declined **97.4%** from its Aug 2020 peak (415,953/month) to Dec 2025 (10,796/month). 2024 and 2025 together hold 450,082 submissions against 4.3M in 2020 alone. The post-training-cutoff eval window ADR 0004 needs therefore *exists but is thin*, and thin in a way that compounds: fewer posts per meme means noisier $I(t)$, fewer clusters reaching the size where an SIR fit is identifiable at all (cf. WP-02's low-count failure regime), and a much smaller viral tail to learn from. ADR 0011 should choose its split with this volume profile in hand rather than assuming recent years resemble 2020.

## Definition of done

1. A complete parquet dataset covering the full archive, with row count reconciled against the 11,332,165 records counted in the raw stream — any shortfall explained, not merely noted.
2. A short report at `experiments/WP-16/report.md` covering:
   - Rows in, rows out, rows dropped, and **why** each drop category exists.
   - Submissions-per-month histogram across the full 2008–2023 range. This is the first corpus-wide view the project has had, and it directly informs ADR 0011's split boundaries.
   - How many rows carry a usable `img_url`, broken down by host (`i.redd.it` vs. imgur vs. other) and by year. This is the input to the link-rot estimate that WP-17 depends on.
3. `docs/02-state.md` data inventory updated with the new coverage.
4. The ingestion is **re-runnable against a different archive file** — the 2025-12 cut is coming, and this should not need rewriting when it does.

## Inputs / prerequisites

- `reddit_data/reddit/subreddits23/memes_submissions.zst` (verified complete, clean zstd stream).
- `zstandard` (already added to the project deps).
- `memes_base/reddit_dataset.py` — the legacy ingester that produced the existing parquet. **Read it before reusing it.** Per `02-state.md`, legacy code is not assumed correct; in particular, establish *why* the original run stopped at 2020-03 before repeating it. If the cause was a silent failure rather than a deliberate `process_chunks` choice, the same bug will bite again.

## Method

1. **Diagnose the truncation first.** Determine what stopped the 2025-01 run at 2020-03-11. Deliberate partial run, OOM, or a schema change mid-archive? This is cheap to establish and determines whether the legacy path is reusable at all.
2. **Handle the known schema hazard.** `created_utc` is a **string** in old records and a **float** in newer ones (`'1247100541'` vs `1704067195.0`). Any parser that assumes one type will fail partway through the archive's history. Check for other fields that drift the same way rather than assuming this is the only one.
3. **Stream, don't load.** 31.5 GB decompressed. Process incrementally to bounded parquet row groups.
4. **Preserve the existing column set** — `is_comment, img_url, created_utc, author, score, link, title, text` — so `sample.py` and `run_semantic_embedding_cache.py` keep working unchanged. If the schema must change, that is a separate decision, not a silent one.
5. **Write to the NVMe**, not `/mnt/d`. This is working-set data per ADR 0015, and the parquet is small (~1 GB).
6. **Do not delete the old parquet** until the new dataset is reconciled. WP-01's results are traceable to it.

## Risks and ambiguities

- **The legacy ingester may be the reason for the truncation.** Rewriting from scratch against the raw ndjson may be cheaper than debugging it. Either is acceptable; decide after step 1 and record which and why.
- **Row-count reconciliation may not be exact.** Deleted/removed posts, malformed JSON lines, and non-image posts are all legitimate drop reasons — but each needs a count, not a shrug. An unexplained gap is a finding, not rounding.
- **This changes the corpus WP-01 sampled from.** WP-01 is closed and its conclusions stand on the data it used, but any *re-run* of the clustering audit will draw from a different, larger pool. Note it rather than silently re-baselining.
- Extending coverage to 2023 also changes what the ADR 0004 contamination cutoff can be set to. That is an improvement, but ADR 0011 should be revisited with the new range in hand rather than inheriting an assumption made when the data stopped at 2020.

## Out of scope

- **Image fetching** — deferred by explicit decision until the 2025-12 archive cut is local, so the expensive irreversible step happens once. That is WP-17.
- **Comments** (`memes_comments.zst`, 100,181,097 records) — whether to ingest them at all is an open question tied to ADR 0007.
- **Downloading the 2025-12 cut** — tracked separately in `open-questions.md`.
- Any change to the embedding or clustering pipeline.

## Log

<!-- Newest at top. -->

- 2026-07-26 — **Coverage gate PASSED (user-certified).** Cross-check numbers replaced with exact full-day counts (1.00 / 1.00, superseding the sampled 1.07 / 0.81); the "decline has flattened" claim withdrawn. Package moved to `active`; ingest started.
- 2026-07-26 — Package opened. Archive verified complete by full streaming decompress; partial ingestion identified.
