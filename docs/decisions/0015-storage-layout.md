# 0015 — Storage layout: tiered devices, sharded image corpus

**Status:** accepted
**Date:** 2026-07-26
**Accepted:** 2026-07-26 — user chose the tier split, then the tar-shard container format
**Gates:** measurements in `03-environment.md` §"Storage: two tiers, measured"

## Context

The project will eventually store an image corpus far larger than the ~20 GB now on disk — plausibly 1M+ meme images (≈175 GB at the measured 175 KB/image average), plus embedding caches and OCR outputs. A 4 TB USB drive was connected for this purpose and mounted at `/mnt/d`.

Benchmarks on this box (full table in `03-environment.md`) establish the constraint:

| Operation | Portable `/mnt/d` | NVMe `/` |
| --- | --- | --- |
| Sequential write | 95 MB/s | 1.4 GB/s |
| Write 2296 small files | 15.5 s (148 files/s) | 0.33 s |
| Read 2296 small files | 9.1 s (252 files/s) | 0.02 s (cached) |
| Re-read, 2nd pass | 9.1 s (no gain) | free |

Three facts drive the decision:

1. **Capacity is not the constraint.** 3.7 TB holds ~21M images. Per-file *overhead* is the constraint.
2. **drvfs reads are not page-cached** — the second read pass was no faster than the first, so a loose-file training loop pays full cost every epoch.
3. **The platter, not 9p, is the floor.** 95 MB/s sequential identifies a 5400 rpm HDD, whose random small reads are seek-bound near 100 IOPS. Reformatting to ext4 would sacrifice cross-platform readability and not fix the small-file case.

The user has endorsed a tiered split. This ADR records it and adds the consequence the benchmarks force: the corpus must not be stored as millions of loose files on the portable drive.

## Decision

**Two tiers, with the image corpus stored as packed shards rather than loose files.**

1. **Portable drive (`/mnt/d`) — archive and shard store.** Raw compressed dumps (`memes_*.zst`), derived shard files, and cold artifacts. Access pattern must be *sequential streaming*, never per-image random reads.
2. **NVMe (`~`, ext4) — working set.** The repo, active image caches, embedding stores, per-run experiment outputs, and anything memory-mapped or randomly accessed. 931 GB free is ~5M images' worth of headroom, so the hot set does not need to be small.
3. **The image corpus is stored as uncompressed tar shards plus a parquet index**, not as one file per image.

### Shard format (normative)

```
/mnt/d/memes/
  images/
    memes-{period}-{nnnnn}.tar     # uncompressed, ~1.5 GB, ~8600 images
  index/
    shard_index.parquet            # post_id | shard_id | offset | length | fetch_status
```

- **Uncompressed `.tar`, never `.tar.gz`.** JPEGs are already compressed, and gzip would destroy seekability — which is the whole point of the index.
- **Member name is `{post_id}.jpg`**, so a shard is self-describing even if the index is lost.
- **The index is the random-access path.** `open(shard).seek(offset).read(length)` returns the exact JPEG bytes. `offset` is the *data* offset, one 512-byte USTAR header past the member header.
- **`fetch_status` is a first-class column**, not an absence. Dead links, 404s, and retries are recorded rather than silently dropped — see the selection-bias note in Consequences.
- **Shards are immutable once closed.** New data — later time periods, or the 2025-12 archive cut — arrives as *new shards*. Nothing is ever rewritten.
- **Shards are time-partitioned**, so a later archive cut appends cleanly instead of forcing a rebuild.
- **Writers must use a large output buffer** (≥16 MB). This is not a micro-optimization: measured below, it is a 3.2× difference on this drive.

### Measured validation (2026-07-26)

Built 4.31 GB of real shards (24,300 meme JPEGs) on `/mnt/d` and measured against criteria fixed *before* the run:

| Metric | Criterion | Measured | |
| --- | --- | --- | --- |
| Sequential stream | ≥76 MB/s (80% of the 95 MB/s raw ceiling) | **77 MB/s**, 435 img/s | pass |
| Random single-image seek | ≤50 ms typical | **0.6 ms** median, 1.4 ms p90, 8.7 ms max | pass |
| Write throughput | — | **117 MB/s**, 660 img/s | — |
| Loose-file baseline | — | 252 files/s read, 148 files/s write | 1.7× / 4.5× worse |

All 200 randomly-seeked payloads verified as valid JPEGs, confirming the index arithmetic.

**Caveat on the seek number:** at 4.3 GB on a 62 GB machine, those reads were served from the Windows file cache. 0.6 ms is far below a 5400 rpm platter's ~11 ms seek, so treat it as an upper bound on performance — what it establishes is that the *mechanism* is correct and adds no meaningful software overhead, not that cold seeks are sub-millisecond. Cold random access should be expected at roughly one seek per image (~10–15 ms), still comfortably inside the criterion. The 77 MB/s stream figure, by contrast, is genuinely disk-bound and not a cache artifact.

Ancillary: **never memory-map or run a sqlite/LMDB-style store on `/mnt/d`.** Those workloads are unreliable over drvfs and pathological on a seek-bound platter. This is not hypothetical — the legacy pipeline already uses one (`dataset_saves/reddit/test_reddit_dataset_embeds.lmbd`, read via `read_from_lmdb` in `memes_base/reddit_dataset.py`), so any revived or successor embed store must stay on the NVMe.

## Alternatives considered

- **Loose per-image files on the portable drive.** Pros: simplest to implement; each image independently addressable; trivially inspectable. Cons: measured ~1.9 h of pure filesystem overhead to write 1M images, and ~66 min per read pass at the optimistic 252 files/s (realistically several hours once files are scattered), repeated every epoch because drvfs reads do not cache. Rejected on the measurements.
- **Reformat the portable drive to ext4** via `wsl --mount --bare`. Pros: native block device, correct POSIX semantics, no 9p layer. Cons: does not address the actual bottleneck — a 5400 rpm platter caps random small reads near 100 IOPS regardless of filesystem — and costs Windows/macOS readability on a drive whose purpose is portability. Rejected as high-cost, low-benefit.
- **Everything on the NVMe.** Pros: fastest possible; no tiering complexity. Cons: 931 GB free caps the corpus at ~5M images and leaves no room for growth, and it puts the only copy of the archive on the same device as the working set. Rejected as a long-term plan, though it is fine for the current 11 GB.
- **Tiered split with sharded corpus (chosen).** Rationale: turns the portable drive's one strength (95 MB/s sequential, ample capacity) into the access pattern actually used, and keeps random-access workloads on the device that handles them. ~31 min per streaming pass over 175 GB, predictable, and the 47× small-file write penalty disappears.

## Consequences

- **The ingestion pipeline must emit shards.** The `.zst` → sample pipeline (currently nonexistent; see below) should write tar shards directly, never a directory of loose JPEGs. This constrains WP-05 (full-corpus embedding) and any future image-fetching code.
- **`experiments/wp01/io_utils.py` and the image-fetch path assume loose files under `images/`.** That is fine at pilot scale on the NVMe, but does not scale to the corpus and will need a shard-aware reader.
- Existing per-run artifacts stay where they are; nothing needs migrating today, since the current 11 GB fits comfortably on the NVMe.
- **Individual images stay cheaply addressable** via the index, so debugging and inspection workflows are not sacrificed — but they must go through the index rather than a filesystem path. Any code doing `open(f"images/{post_id}.jpg")` needs a shim.
- **`fetch_status` carries methodological weight, not just bookkeeping.** Link rot is severe on pre-2018 posts pointing at external hosts and mild on Reddit-hosted `i.redd.it`. Silently dropping failed fetches would bias the corpus toward posts that survived, which correlates with popularity — the R2 selection problem entering through the back door. Recording every attempt makes that bias measurable instead of invisible.

### Why this layout suits the actual access pattern

Worth stating explicitly, because it is what rules out a heavier random-access store: per the WP roadmap, pixels are consumed **once** by WP-05 to produce embeddings, and everything downstream — WP-06 clustering, WP-08 fitting, WP-10's `embedding → params` predictor, and ADR 0014's frozen-PCA marks — operates on embeddings, not images. Realistic image access is therefore a handful of lifetime sequential passes (one per embedder candidate in WP-03, one per OCR variant), plus small random reads for WP-04's gold set and cluster inspection. There is no repeated pixel-level training epoch unless the embedder is later fine-tuned — which is a parked open question, and would justify revisiting this ADR if it is ever taken up.

## Open sub-questions

- ~~Shard format specifics~~ **Resolved 2026-07-26:** uncompressed tar + parquet index (user-chosen over parquet-with-binary-column and Lance). Rationale: plain tar needs no dependency to read (`tarfile` is stdlib), appends without rewriting, stays framework-agnostic — which matters under ADR 0013's JAX-primary stance, since WebDataset tooling is torch-centric while the *format* is not — and the index recovers the random access that tar alone lacks. Lance was the technically strongest fit but adds a young dependency for a benefit (fast random access) the measured access pattern barely needs.
- ~~Shard size~~ **Resolved 2026-07-26:** **1.5 GB** (~8600 images). At the measured 77 MB/s that is ~20 s to stream one shard, fine-grained enough that an interrupted fetch loses little, while a 1M-image corpus is only ~117 files. The benchmark did not discriminate between 1 and 2 GB, so this is a reasoned choice within a flat region, not a measured optimum.
- **Where does the corpus of record live?** The portable drive is a single point of failure with no redundancy. If the fetched image corpus is expensive to re-download, it needs a backup plan — this ADR does not provide one.
- **The submissions archive is already ingested to loose parquet, and its coverage stops at 2020-03-11.** `dataset_saves/reddit/test_reddit_dataset/*.parquet` (4.1M rows) was produced from `memes_submissions.zst` by `memes_base/reddit_dataset.py`. Any re-ingestion to extend coverage is the natural moment to adopt the shard format decided here — but the shard-format sub-question above should be settled first, so the re-ingest is done once. That work is upstream of this ADR; see `open-questions.md` §"Data ingestion".
