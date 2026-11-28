# Review: is the r/memes submissions table a complete and faithful copy of the archive?

Reviewer: independent agent, 2026-11-27 (finishing a review a previous reviewer started the
same day; the machine shut down before that reviewer finished). Nothing here was produced
by the author of the ingest. Status of the ingest after this review: **agent-reviewed**.

**What was reviewed.** In July an agent converted the r/memes submissions archive
(`/mnt/d/reddit_archive/subreddits25/memes_submissions.zst`: one compressed file, 2.82 GB,
33.3 GB once decompressed, one JSON record per line) into a table,
`dataset_saves/reddit/submissions-2025-12/submissions.parquet` (13 columns). The author
reported 11,782,247 rows, none dropped, posts dated 2008-07-11 to 2025-12-31 (the "submissions
re-ingest", work package WP-16). The code is `experiments/wp16/ingest.py`.

## Verdict: faithful, with caveats

Every one of the 11,782,247 records in the archive is in the table, once, in archive order,
and every value in every one of the 13 columns equals what the archive holds (under the
documented renamings). This was shown twice over the full population, by two different
decompressors and two different JSON parsers. No record was dropped, added, reordered or
altered. Timestamps are the archive's own UTC epoch seconds, unconverted.

The caveats are not ingest errors. They are things a user of the table must know:

1. **One post id appears twice** (`xfn9vg`, a "meme quiz" prediction-tournament post),
   because it appears twice in the archive, with creation times one month apart
   (2022-10-09 and 2022-11-09) and different scores. Anything keyed on post id must decide
   which copy to keep.
2. **Monthly volume halves in one step between January and February 2025** (20,694 to
   11,135 posts) and stays at the lower level. It is in the archive, not introduced by the
   ingest, but it looks like a step rather than gradual decline, and the earlier
   coverage check said there was no step change. Open finding, below.
3. **The ingest code has latent paths that would lose or blur data on a different archive**
   (multi-frame files truncated silently; null text stored as empty string; fractional
   timestamps truncated; a non-boolean `over_18` nulled without a counter). None of them
   fired on this archive (each counted at zero), so this table is unaffected; the code
   should not be reused on other archives without fixing the first one.
4. **The previous reviewer's reference extract contains one corrupted byte** (a single
   flipped bit). It made the full comparison report one false mismatch, which I traced to
   the extract, not the table. A single-bit error in a file written on this machine is
   itself worth knowing about.

## Checks and raw numbers

Expectations for these checks were written before any were run, by the previous reviewer:
`experiments/WP-16/review/expectations.md`. I kept them as the pass/fail criteria. Numbering
below follows that file.

| # | Check | Result | Pass? |
|---|---|---|---|
| 0 | Archive reviewed = archive the author ingested | `/mnt/d` copy and the author's `~/reddit_dl` copy: both 2,820,041,962 bytes, both SHA-256 `cf742ba0…0c87` | pass |
| 1 | Decompression complete | Hand-walk of the zstd container (no zstd library): 1 frame, 253,866 blocks, ends exactly at end of file, on both copies. Frame carries no content checksum. Decompressed size 33,274,645,062 bytes from python-zstandard (previous reviewer) and from pyzstd (mine): identical | pass |
| 2 | Line count | 11,782,247 newline bytes, 11,782,247 non-blank lines, 0 blank, 0 unparseable (Python `json` and `orjson` both), 0 non-object lines, no unterminated last line. Table rows: 11,782,247 | pass |
| 3a | Full-population comparison, all 13 columns, row *i* vs line *i* (previous reviewer's `compare.py` against their extract; run to completion twice, same result) | Value and type must both match; JSON null and missing key both count as null. 13 columns × 11,782,247 rows: `post_id`, `author`, `subreddit`, `removed_by_category`, `score`, `num_comments`, `over_18`, `link`, `title`, `text`, `is_comment`: 0 mismatches each; `created_utc`: 0 rows differ, no offsets; 0 table rows beyond the archive; `img_url`: **1 mismatch** (`img_url`, row 11,280,838), traced to a corrupted byte in the extract, not the table (see Discrepancies) | pass after resolution |
| 3b | Full-population comparison, independent route (mine: `independent_pass.py`) | For every line: decompress with pyzstd, parse with orjson, apply the ingest's documented rules (reimplemented), hash the expected 13 values; compare with the hash of the actual table row at the same position. **0 of 11,782,247 rows differ**; 0 `created_utc` differ. Negative control: same comparison shifted by one row matches 0 of 20,000; with `score`+1 matches 0 of 20,000 | pass |
| 4 | Seeded sample, 5,000 raw lines (seed 20261127), looked up by post id | 5,000/5,000 ids found exactly once; all 5,000 at the expected row position; 65,000 cells, 0 mismatches; 24,990 string cells compared code point for code point, 447 containing non-ASCII | pass (reused) |
| 5 | Monthly counts | 188 months with posts; raw vs table: 0 months differ; raw vs author's `ingest_report.json`: 0 months differ; totals 11,782,247 in all three. See plot | pass |
| 6 | Time zone | Table `created_utc` equals the raw epoch seconds in every row (0 differ, both routes), so no conversion of any kind took place. The three raw types (int 10,988,851; string 32,179; float 761,217) were checked for a clock offset against each other (`tz_check.py`): in post-id order, consecutive posts that switch from int to float are a median 79 s apart (23,012 switches) and float to int 75 s (23,011), against 55 s float-to-float and 7 s int-to-int; no hour-sized jump. Int/string switches occur only twice (1,870 s and 679 s, in sparse 2015). No float timestamp has a fractional part | pass |
| 7 | Duplicate / null / empty post ids | Table: 11,782,246 distinct ids in 11,782,247 rows: **1 id twice** (`xfn9vg`, rows 10,729,411 and 10,782,798). Archive: the same, 1 id twice. 0 null, 0 empty ids. Copies differ in `created_utc` (1665336048 vs 1668020416), `score` (35,268 vs 447,898), `num_comments` (99 vs 3,226) | not an ingest fault; open finding |
| 8 | Range, subreddit, filtering | All 11,782,247 rows have `subreddit` = `memes`. Earliest 2008-07-11 08:04:26 UTC, latest 2025-12-31 23:58:52 UTC; 0 timestamps zero or negative; 0 outside July 2008 to December 2025. Nothing was filtered: the table holds every archive line. The file is roughly but not strictly in time order (51,696 rows are earlier than the row before them, by at most 18.5 hours), which matches the archive's own order | pass |
| 9 | Type drift | Raw JSON types over all 11,782,247 records: `author`, `title`, `selftext`, `permalink`, `id`, `subreddit` always strings; `score`, `num_comments` always integers; `over_18` always boolean; `url` string or null (null 14,843); `removed_by_category` string 4,825,291, null 4,565,223, key absent 2,391,733; `created_utc` int / string / float as in check 6. So none of the ingest's silent coercions had anything to act on | pass |
| 10 | What `img_url` holds | It is the raw `url` field, verbatim, in every row. It is not always an image: i.redd.it direct image 10,346,214 (87.8%); other link 787,554; the post's own page (text post) 298,984; other direct image (imgur or image file extension) 210,221; v.redd.it video 123,253; null 14,843; reddit gallery page 1,178. In 2022–2024, 25–45% of each year's posts are not direct image links (second plot). Correctly ingested, but the column name overstates what it holds | caveat (not an ingest fault) |
| 11 | Author's counters recomputed | `created_utc` from float 761,217, from string 32,179, `url` null 14,843, `url` unparseable 241, i.redd.it rows 10,346,214, yearly totals: all reproduced exactly. imgur: author 251,442, mine 252,239 (author summed the top 15 hosts per year only; mine counts every imgur host), a 797-row difference in a descriptive figure | pass |
| 12b | Do timestamps agree with post-id order? (added) | Post ids are a global counter assigned at creation. Comparing each post's `created_utc` with the median of the 50 posts either side in id order (`id_time_outliers.py`): 4 posts (5 rows) are more than a day off: `lm7wbx` (2 days early), `uc6gaq` (20 days late) and both copies of `xfn9vg` (23 and 54 days late). 5 rows more than an hour off. `compare.py`'s own metric for this (137,539 rows more than a day earlier than some lower id) is misleading: it compares against a running maximum, so these few late outliers make every later-id row look early | pass; 4 odd posts noted |
| 12 | Could the ingest lose or alter records without tripping its own assertion? | Yes in principle, four ways; none was exercised by this archive. See below | pass for this table |

## Plot

`experiments/WP-16/review/monthly_counts_parquet_vs_raw.png`: posts per month, counted
independently from the raw archive (blue line) and from the table (orange circles), with
the difference on a second panel. If the ingest were faithful, every circle would sit on
the line and every point in the lower panel at zero. That is what it shows: 188 of 188
months at zero difference. The line joins across 22 empty months in 2008–2010 (only 11
posts exist before 2011).

`experiments/WP-16/review/img_url_content_by_year.png`: what the `img_url` column points to,
by year. If the column held only image links, every bar would be blue and orange. From 2016
to 2021 it nearly is (87–95% direct images from 2018); in 2022–2024 a quarter to almost half
of each year is text posts, videos and other links. This is the archive's `url` field
faithfully copied; it matters for anyone fetching images from it.

## Every discrepancy found, however small

1. **One false mismatch, caused by a flipped bit in the reference extract.** The full
   comparison found the table's `img_url` at row 11,280,838 (post `1829zgu`) to be
   `…/comments/1829zgu/anyone_know_what_this_means/` while the previous reviewer's extract
   said `…/comments/1829zg5/…`. I read that line straight from the archive (pyzstd + orjson,
   `show_raw_line.py`, output `raw_line_11280838.json`): the archive says `1829zgu`, matching
   the table, the post's own id and its permalink. `u` is byte 0x75 and `5` is 0x35: one bit
   apart. The extract was written by Python code that copies the value verbatim, so the
   likeliest cause is a single-bit memory or storage error on this machine during the
   previous reviewer's pass (the machine later shut down unexpectedly). My integrity check
   of the extract (row counts, contiguous line numbers, byte totals reconciling exactly to
   the decompressed size, sample agreement) could not catch this, because a flipped bit
   changes no length. The table itself was confirmed against the archive by a separate
   pass with different software (check 3b, 0 differences), so this does not touch the
   verdict. It does mean the extract is not a trustworthy reference for anything beyond
   this review, and that single-bit errors on this machine are not hypothetical.
2. **Duplicate post id `xfn9vg`**: in the archive and therefore in the table (check 7).
3. **`removed_by_category` missing versus null.** The key is absent in 2,391,733 records
   and explicitly null in 4,565,223 others; the table stores both as
   null. This loses only the "key was absent" distinction, which carries no information
   about the post. Not documented by the author.
4. **The author's description of timestamp types is slightly wrong.** The report says
   `int` covers 2008–13 and 2016–22 and `float` 2023 onward. In fact ints also occur in
   2015 (1,545), 2023 (131,734), 2024 (15,686) and 2025 (5,378). Harmless, because the
   ingest branches on the value's type, not the year.
5. **The author's "independent" cross-checks were not independent of the ingest.** The
   monthly re-count (`count_monthly.py`) used the same decompression library and call
   pattern (which would stop at the first frame, as the ingest does), the same
   `int(float(…))` conversion and the same `time.gmtime` month bucketing; and the 11,782,247
   line count it was matched against came from the same library. They agreed with the
   ingest, and the archive has one frame, so nothing was missed; but they could not have
   caught a shared failure.

## Could the ingest lose or alter records without its own check noticing?

The ingest's check is `rows_in == rows_out + rows_dropped`. All three are incremented
inside the same loop over the lines the reader yields, so the check can only catch a
`continue` that forgot its counter. It cannot see lines the reader never yields, and it
says nothing about values. Paths by which data could be lost or changed while it passes:

| Path | Would it be caught? | Exercised by this archive? |
|---|---|---|
| `zstandard.stream_reader` without `read_across_frames=True` stops at the end of the first zstd frame; a multi-frame archive would be silently truncated | No (lines never yielded are never counted) | No: 1 frame, verified by container walk and by line count equality |
| `title`, `selftext`, `permalink`: null or missing stored as `""` (`x or ""`) | No | No: 0 null or missing in 11,782,247 records (orjson pass counters) |
| `created_utc` with a fractional part truncated by `int(float(x))` | No | No: 0 fractional values |
| `over_18` that is not a boolean silently set to null, not counted | No | No: 0 non-boolean |
| `score` / `num_comments` not an integer: coerced (counted) or nulled | Counted only | No: 0 coerced (author's counter and mine) |
| Blank lines become `bad_json` drops | Yes (counted) | No: 0 blank lines |
| A line that is valid JSON but not an object, or a value pyarrow cannot store | Crash, not silent | No |
| Duplicate keys inside one JSON object (last wins) | No | Not checked directly; both parsers used (Python `json`, orjson) take the last value, so a disagreement would not show. Considered negligible |

So the assertion protects very little, and completeness of this table rests on the external
checks above, not on it.

## Open findings (numbers pass, but something looks odd)

None of these is an ingest error; all are properties of the archive that the table
faithfully carries.

1. **Step down in February 2025.** Posts per month: 2024-11 20,985; 2024-12 21,752;
   2025-01 20,694; **2025-02 11,135**; then 10,500–12,000 for the rest of 2025. A 46% fall
   in one month, then flat, is the shape of a change in collection or in subreddit rules,
   not of the gradual decline seen 2021–2024. A smaller drop also occurs 2024-01 41,732 to
   2024-02 30,947. The July coverage check (which the user certified) concluded there was
   no step change, but it looked for one at May 2023; this one is later. Its single
   outside comparison day in the affected period (2025-08-15, 378 posts per day, matched by
   the Arctic Shift service) is after the step, so it cannot say whether the drop is real.
   Comparing a few days either side of 2025-02-01 against an outside source would settle
   it. This affects any evaluation window placed in 2025.
2. **41% of posts carry a removal label.** `removed_by_category` is set on 4,825,291 rows:
   `deleted` 2,375,057, `moderator` 2,354,354, `reddit` 95,211, and under 1,000 each for
   six other labels. Faithfully ingested; flagged because a researcher who assumes the
   table is mostly live posts would be wrong, and because the July report named this
   column as the way to check whether the 2020 peak was inflated by removed spam.
3. **Duplicate post with a shifted creation time** (`xfn9vg`, check 7). The two copies are
   31 days apart in `created_utc`. For a model of spread over time, a post whose creation
   time is not stable is worth knowing about, even if it is one of 11.8 million.

4. **Text is stored as the archive has it, which is not clean text.** Titles contain HTML
   escapes (`&amp;`, `&lt;`, `&gt;`) in 31,272 rows; 2 titles contain a NUL character;
   40,269 `img_url` values contain `&amp;` where a fetchable URL needs `&`. In `text` (the
   post body), 8,165,227 rows are empty, 3,227,792 are `[deleted]` and 381,648 are
   `[removed]`. Downstream text embedding and image fetching should unescape and filter
   deliberately.

## What I reused, what I redid, what I added

Reused (read critically, judged sound):
- `extract_raw.py` and its outputs (`ref_extract/`, `raw_pass_summary.json`,
  `sample_raw.jsonl`). I checked the extract before using it (`verify_extract.py` →
  `verify_extract_results.json`): 24 shards, all readable, no leftover temporary files,
  line numbers 0 to 11,782,246 with no gap or repeat, byte lengths plus newlines summing
  exactly to the decompressed size (33,274,645,062), 0 lines without an id or timestamp,
  and the 5,000 sampled lines agreeing with the extract in length and id. It is complete;
  it is not wholly intact (one flipped bit, found later).
  One deviation from the expectations file: it promised decompression with the `zstd`
  command-line tool, but the script used the same Python library as the ingest (with a
  different, frame-following API). The tool is not installed; my pass (3b) supplies the
  independent decompressor instead.
- `sample_check.py` and `sample_check_results.json` (check 4): read the code; the
  comparison is strict (value and type) and the positional cross-check is right.
- `zstd_frames.py`: reran it on the `/mnt/d` copy (`zstd_frames_mntd.json`).
- `compare.py`: reran in full (see below). The first rerun completed the whole cell-by-cell
  comparison (log: `compare_run1_killed_after_cell_comparison.log`, monthly counts:
  `monthly_counts_run1.csv`) and was then killed by something outside it, with no error
  and no out-of-memory record, before writing its summary file. I reran it detached from
  the shell; the second run finished (exit 0) with identical cell results and an identical
  monthly table, and wrote `compare_results.json`. I also wrote `summary_stats.py` as a
  faster cross-check of its descriptive figures; the two agree on every shared number.
- `plots.py`: kept, with the legend moved off the data and the empty early months labelled.

Discarded: `compare_results.json` from the previous reviewer (partial, first 1,000,000
lines). Renamed `STALE_compare_results_first1M.json` together with
`STALE_monthly_counts_first1M.csv` and `STALE_compare_interrupted.log`, so nothing reads
them by accident.

Added:
- `independent_pass.py` (check 3b): the one extra full pass over the archive, with a
  different decompressor (pyzstd, libzstd 1.6.0) and parser (orjson 3.13), via
  `uv run --with`, so the project lockfile is unchanged. Also recorded the SHA-256 of the
  decompressed stream (`f4abf3c4…1d4f`) for any future reviewer.
- `independent_negative_control.py`: shows the comparison in 3b can fail.
- SHA-256 of the author's home copy (`sha256_homecopy.txt`).
- `tz_check.py` (check 6), `ids_check.py` and `dup_detail.py` (check 7), `id_time_outliers.py` (check 12b),
  `find_url_mismatch.py` and `show_raw_line.py` (discrepancy 1).

## What I could not check

- Whether the archive itself is complete relative to what was posted on Reddit. That is a
  property of the archive, not of the ingest (see open finding 1).
- Whether the archive is byte-identical to what its publisher released: the zstd frame has
  no checksum, and I did not compare the file against the torrent's piece hashes.
- JSON objects with duplicate keys (both parsers keep the last one).
- Whether `score` and `num_comments` are meaningful: they are snapshots taken at collection
  time, which varies by year. The table reproduces the snapshot faithfully.

## Paths

Outputs, `experiments/WP-16/review/`: `review.md` (this file), `expectations.md`,
`monthly_counts_parquet_vs_raw.png`, `img_url_content_by_year.png`, `summary_stats.json`, `monthly_counts_run1.csv`,
`compare.log`, `compare_results.json`, `independent_pass_stage1.json`,
`independent_pass_stage2.json`, `independent_negative_control.json`,
`verify_extract_results.json`, `tz_check_results.json`, `ids_check_results.json`,
`duplicate_id_detail.json`, `id_time_outliers.json`, `img_url_mismatch.json`, `raw_line_11280838.json`,
`sample_check_results.json`, `raw_pass_summary.json`, `zstd_frames_mntd.json`,
`zstd_frames_homecopy.json`, `sha256_homecopy.txt`.

Code, `experiments/wp16/review/`: `extract_raw.py`, `verify_extract.py`, `compare.py`,
`independent_pass.py`, `independent_negative_control.py`, `sample_check.py`, `tz_check.py`,
`ids_check.py`, `dup_detail.py`, `id_time_outliers.py`, `summary_stats.py`, `find_url_mismatch.py`, `show_raw_line.py`,
`zstd_frames.py`, `plots.py`.

Nothing in `docs/`, the table or the archive was changed; nothing was committed. The record
(WP-16 log, `docs/02-state.md`) still says UNCERTIFIED and should be updated by whoever
owns it.
