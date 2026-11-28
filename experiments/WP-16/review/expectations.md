# Review of the r/memes submissions ingest: expectations written before running the checks

Written 2026-11-27 by the reviewing agent, before any check was run against the archive.
I had read the ingest code (`experiments/wp16/ingest.py`), the author's report, and the
parquet's schema/metadata (13 columns, 11,782,247 rows, 48 row groups). Nothing else.

The claim: `dataset_saves/reddit/submissions-2025-12/submissions.parquet` is a complete and
faithful copy of the 13 ingested fields of every record in
`/mnt/d/reddit_archive/subreddits25/memes_submissions.zst`.

This is a claim of exact identity between two finite data sets, so the test is exact
equality over the whole population, not a tolerance on a sample. Any nonzero mismatch count
is a discrepancy; the only judgement is whether a discrepancy is a documented, deliberate
transformation or an undocumented one. The seeded sample (check 4) is kept because it was
asked for and because it compares against untouched raw lines; the full-population
comparison (check 3) is the stronger test.

Method: one streaming pass over the archive using the `zstd` command-line tool (not the
Python `zstandard` binding the author used for both the ingest and the "independent"
monthly count) and Python's own line iteration (not the author's chunk-split-and-carry
code). The pass writes a reviewer's extract that keeps each field as its raw JSON token
(so null / missing / "" / int / float / string stay distinguishable), plus a seeded
reservoir sample of 5,000 untouched raw lines.

| # | Check | If the ingest is faithful | Counts as failure |
|---|---|---|---|
| 0 | The archive I read is the archive the author read (they used a copy under `~/reddit_dl`, I am pointed at `/mnt/d`) | Same byte size; same SHA-256 if the home copy still exists | Different size or hash: then every later mismatch is ambiguous and I say so |
| 1 | Decompression completeness: zstd CLI exits 0, number of frames, decompressed bytes | Exit 0. If the file has more than one zstd frame, the author's reader (which by default stops at the end of the first frame) would have silently truncated, so the line count below would exceed the parquet's | Non-zero exit (truncated/corrupt archive); or my line count > parquet rows |
| 2 | Line count: non-blank lines in the decompressed stream, blank lines, lines that are not a JSON object | Non-blank lines = 11,782,247 = parquet rows; 0 blank; 0 non-object; 0 unparseable | Any difference in the count. Blank or unparseable lines > 0 while the author reports 0 drops would mean the author's accounting is wrong |
| 3 | Full-population field comparison, row i of parquet against line i of archive, all 13 columns | Row order is archive order and every value equals the raw value under the documented rules: `img_url` = raw `url` verbatim (null stays null); `created_utc` = raw value as integer seconds UTC; `author`, `post_id`, `subreddit`, `removed_by_category` verbatim; `score`, `num_comments` verbatim integers; `link` = `permalink`; `title`; `text` = `selftext`; `over_18` verbatim; `is_comment` false everywhere | Any row where a value differs. I expect, from reading the code, three classes of *undocumented* alteration and will count each: (a) null/missing `title`, `selftext`, `permalink` stored as "" (null and empty become indistinguishable); (b) non-integer `score`/`num_comments` silently truncated or nulled; (c) fractional `created_utc` truncated. Each is a fidelity loss if its count is > 0, small or not. Anything outside these classes is an unexplained mismatch and a failure |
| 4 | Seeded sample: 5,000 raw lines chosen by reservoir sampling (seed 20261127), parsed fresh, matched to the parquet by post id, every column compared | 5,000 of 5,000 ids found exactly once; 0 mismatching cells out of 65,000, apart from the classes in 3(a-c) | An id missing from the parquet, found more than once, or any cell mismatch outside 3(a-c). Also string checks: equal length in characters (no truncation), equal code points (no unicode normalisation or mojibake) |
| 5 | Monthly counts recomputed from the raw values with numpy datetime arithmetic (not `time.gmtime`), compared with (i) counts from the parquet's `created_utc`, (ii) the author's `ingest_report.json` | Identical in every month; 188 months; total 11,782,247 | Any month differing by even 1 |
| 6 | Time zone: are months/dates UTC? | parquet `created_utc` equals the raw epoch seconds exactly, so there is no time-zone conversion at all; min/max correspond to 2008-07-11 and 2025-12-31 in UTC | Any constant offset (e.g. 3600 s) between parquet and raw |
| 7 | Duplicate post ids; null or empty ids | 0 duplicated ids in the archive and in the parquet (a Reddit post id is unique). If the archive itself has duplicates the parquet should have the same ones (the ingest promises no filtering) but downstream users must know | Duplicates in the parquet not in the archive (ingest bug). Duplicates in both: not an ingest failure, but an open finding about the archive |
| 8 | Range and ordering: timestamps outside 2008-07..2025-12, negative or zero; subreddit other than `memes`; is the file time-ordered | All timestamps inside the claimed range; one subreddit; roughly time-ordered | Rows outside the range; other subreddits (would matter for downstream group-bys) |
| 9 | Type drift the author did not report: types of every ingested field in the raw data (not only `created_utc`) | Types as the code assumes: `score`/`num_comments` int, `over_18` bool, strings are strings or null | Any field arriving with a type the code coerces silently: e.g. `over_18` non-boolean becomes null without a counter; `score` boolean; `author`/`title` non-string (pyarrow would have raised, so I expect none) |
| 10 | How `img_url` was derived | It is the raw `url` field and nothing else. So it is expected NOT to be an image URL for text posts, galleries, videos, crossposts and link posts. I will count: rows where `img_url` is a reddit gallery link (the actual images live in `media_metadata`, which was not ingested); rows where it is the post's own permalink (text posts); rows where `url` is null but the record has another URL-bearing field | Not an ingest failure by itself, but if a material share of image posts have no usable image URL in the parquet, the "88% i.redd.it" headline and the downstream image fetch need a caveat |
| 11 | Author's reported counters, recomputed: `created_utc` from float 761,217; from string 32,179; `url` null 14,843; yearly totals; i.redd.it rows 10,346,214 | Exactly reproduced | Any difference |
| 12 | Could the ingest lose or alter records without tripping its own assertion? (code reading plus the data facts from the pass) | The assertion `rows_in == rows_out + drops` is an identity on three counters incremented in the same loop; it cannot detect lines never yielded by the reader. So it only protects against a `continue` without a counter. Completeness really rests on checks 1-2 | A concrete path by which a record is lost or changed with the assertion still passing, that the data actually exercises |

Blind spots I can name in advance:

- Checks 3-5 share one decompression and one JSON parser (Python's `json`) with each other,
  and the JSON parser with the author. A record that Python's `json` mis-parses the same way
  twice would not be caught. I regard that as negligible.
- None of this says anything about whether the archive is complete relative to Reddit. That
  is a separate question; I will comment on the author's two-day cross-check but not test it.
- Score and comment counts in the archive are snapshots taken at some collection time. A
  faithful ingest reproduces the snapshot; it does not make the snapshot meaningful.
