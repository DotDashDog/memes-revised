# Image-pipeline throughput — engineering note

**Status:** living note, updated as WP-04 executes
**Purpose:** WP-04 deliverable (f). Records measured download / OCR / embedding rates, the
settings that achieved them, and every rate-limit or failure mode encountered. Required
input for WP-17 (full-corpus fetch, ~10M images) and WP-05 (full-corpus embedding).
**Hardware:** WSL2 on Windows; RTX 3090 (24 GB); NVMe working set; `/mnt/d` = 5400 rpm USB
drive (~95 MB/s sequential, ~100 IOPS) per ADR 0015.

## 1. Downloads (WP-04 pool fetch, 2026-07-27)

Fetcher: `experiments/wp04/fetch.py` — asyncio + aiohttp, per-host lanes, single
sequential tar-shard writer (32 MB buffer) to `/mnt/d/memes/` per ADR 0015.

### Settings that worked

- **i.redd.it lane** (92.6% of pool URLs): start 4 workers, ramp +2 per 60 s toward 16
  while rolling 60 s error rate < 1%; jittered per-worker pacing floor 0.1 s; keep-alive
  connection reuse (`aiohttp.TCPConnector`, DNS cache 600 s).
- **Other-host lane** (imgur, imgflip, …): 2 workers, ~1 s jittered pacing.
- Honest static User-Agent naming the project repo; timeouts connect 10 s / total 30 s;
  3 retries with 1 s/4 s/16 s exponential backoff for timeouts and 5xx.
- Throttle tripwire: ≥3 × 429/403 within 60 s → lane full-stops for 300 s and halves its
  workers. **Never triggered in the pilot** (zero 429/403 from i.redd.it at ~112 req/s).
- Imgur page URLs rewritten `imgur.com/{id}` → `i.imgur.com/{id}.jpg`; albums/galleries,
  `v.redd.it`, `www.reddit.com` (galleries/permalinks), youtube → recorded as
  `unsupported_url`, not fetched. ~4.6% of the pool (23/500 pilot).

### Measured (pilot, 500 posts / 477 URLs)

| metric | value |
|---|---|
| throughput, 4 unpaced workers, warm CDN edge | ~112 req/s, ~13 MB/s |
| throughput, 4 unpaced workers, cold edge (first run) | ~22 req/s |
| 429/403 events | 0 |
| HTTP 404/410 (2019 i.redd.it links) | **43%** (207/477) |
| imgur `removed.png` redirects | 2/8 |
| median image size (2019, ok) | 85 KB (mean 205 KB, max 3.7 MB) |

(Full-run numbers appended below when the 50k fetch completes.)

### Failure modes hit (so far)

1. **aiohttp truncated-read bug (ours):** `StreamReader.read(n)` returns *up to* n bytes —
   the first buffered chunk (often exactly 16384 B). Every pilot image was silently
   truncated; PIL `Image.open` still reported plausible headers, so only the
   size-vs-dimensions mismatch exposed it. Fix: `await resp.read()` (read-everything),
   cap enforced afterwards. **WP-17 lesson: validate `len(body)` against Content-Length
   and fully decode a sample before trusting any fetcher.**
2. **Link rot is content-correlated, not noise:** 43% of 2019 i.redd.it URLs 404 even
   though only 8% of 2019 posts are flagged `removed_by_category`. Deleted-by-user posts
   lose their images without carrying the flag. Recorded per-post in `fetch_status`
   (ADR 0015's R2 selection-bias requirement); WP-04 §2 must check whether 404s correlate
   with family size before deriving n.
3. **Warm-edge speedup:** rerunning the same URL set is ~5× faster (CDN edge cache). Any
   throughput measured on re-fetched URLs overstates cold-corpus rates — WP-17 estimates
   must use cold numbers.

### Extrapolation to WP-17 (to revisit with full-run numbers)

At a sustained polite ~60 req/s with ~57% live links and ~205 KB mean size: ~10M URLs ≈
2 days of wall-clock fetch, ~1.2 TB — /mnt/d capacity (3.7 TB) is fine, and the write
path (sequential tar, ≤ ~12 MB/s) is far below the drive's ~95 MB/s ceiling. The
bottleneck is politeness, not disk or bandwidth.

## 2. OCR (to be measured in WP-04 §5)

- EasyOCR on the 3090: batch detector/recognizer; measure images/s at several batch sizes.

## 3. Embedding (to be measured in WP-04 §5)

- OpenCLIP B/16 image+text encoders on the 3090: measure images/s vs batch size, fp16 vs
  fp32.
