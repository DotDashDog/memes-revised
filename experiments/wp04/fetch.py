"""WP-04 Method §1: polite async image fetcher writing ADR 0015 tar shards.

Design (see WP-04 "Fetch discipline" and ADR 0015):
- Corpus: uncompressed tar shards `/mnt/d/memes/images/memes-2019-{nnnnn}.tar`,
  member `{post_id}.{ext}`, cut at ~1.5 GB, written sequentially through a
  single writer with a 32 MB buffer. Shards are immutable once closed.
- Index: one parquet per closed shard under `/mnt/d/memes/index/parts/`
  (post_id | url | host | shard_id | offset | length | fetch_status |
  http_status | content_type | sha256 | ts). Failures carry a null shard_id.
  `compact_index()` folds parts into `index/shard_index.parquet`.
- Resume: on start, read all index parts; post_ids with a terminal status are
  skipped. Temporary failures (timeout/5xx) are retried on the next run.
- Politeness: per-host lanes. i.redd.it (a CDN) ramps 4 -> 16 workers, +2
  every 60 s while the rolling error rate stays under 1%; any burst of
  429/403 (>=3 in 60 s) halts the lane for 5 min and halves its workers.
  Other hosts get 2 workers and ~1 req/s pacing. Honest static User-Agent,
  jittered pacing, keep-alive reuse.
- URLs that cannot be a direct image (reddit galleries, v.redd.it, youtube)
  are recorded as `unsupported_url` — dead/unfetchable links are selection
  data, not noise.

Usage:
    uv run python -m experiments.wp04.fetch --limit 500   # pilot
    uv run python -m experiments.wp04.fetch               # full pool
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import random
import re
import tarfile
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

POOL = Path("/home/linus/memes-revised/experiments/WP-04/pool.parquet")
METRICS = Path("/home/linus/memes-revised/experiments/WP-04/fetch_metrics.jsonl")
OUT_ROOT = Path("/mnt/d/memes")
IMAGES_DIR = OUT_ROOT / "images"
INDEX_PARTS = OUT_ROOT / "index" / "parts"
SHARD_PREFIX = "memes-2019"
SHARD_TARGET_BYTES = 1_500_000_000
WRITE_BUFFER = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 30 * 1024 * 1024
USER_AGENT = (
    "meme-spread-research/0.1 "
    "(+https://github.com/DotDashDog/memes-revised; academic research on meme diffusion)"
)
FETCH_ORDER_SEED = 7  # fixed shuffle so pilot prefix == full-run prefix

TERMINAL = {
    "ok",
    "http_404",
    "http_410",
    # 403 from these image hosts is a permanent content block, not rate
    # limiting — retrying it on every resume just trips the throttle tripwire
    "http_403",
    "removed",
    "unsupported_url",
    "non_image",
    "too_large",
    "empty_body",
}

MAGIC = [
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"RIFF", "webp"),  # checked with WEBP at offset 8
]


def sniff_ext(body: bytes) -> str | None:
    for magic, ext in MAGIC:
        if body.startswith(magic):
            if ext == "webp" and body[8:12] != b"WEBP":
                continue
            return ext
    return None


def normalize_url(url: str) -> tuple[str | None, str]:
    """Return (fetchable_url or None, lane). None => unsupported_url."""
    url = url.strip()
    m = re.match(r"https?://([^/]+)(/.*)?$", url)
    if not m:
        return None, "unsupported"
    host, path = m.group(1).lower(), m.group(2) or "/"
    if host == "i.redd.it":
        return "https://i.redd.it" + path, "iredd"
    if host == "i.imgur.com":
        return "https://i.imgur.com" + path, "other"
    if host in ("imgur.com", "www.imgur.com", "m.imgur.com"):
        if re.match(r"^/(a|gallery|t)/", path):
            return None, "unsupported"
        stem = path.split("?")[0].rstrip("/")
        if not re.match(r"^/[A-Za-z0-9]+$", stem):
            return None, "unsupported"
        return f"https://i.imgur.com{stem}.jpg", "other"
    if host in ("v.redd.it", "www.reddit.com", "reddit.com", "old.reddit.com",
                "youtu.be", "www.youtube.com", "youtube.com", "gfycat.com",
                "www.gfycat.com", "v.gfycat.com"):
        return None, "unsupported"
    # generic host: only attempt direct-image-looking URLs
    if re.search(r"\.(jpe?g|png|gif|webp)(\?.*)?$", path, re.I):
        return url, "other"
    return None, "unsupported"


@dataclass
class Job:
    url: str
    fetch_url: str
    host: str
    post_ids: list[str]
    lane: str = "other"


@dataclass
class Result:
    job: Job
    status: str
    http_status: int | None = None
    content_type: str | None = None
    body: bytes | None = None
    ext: str | None = None
    latency: float | None = None


class ShardWriter:
    """Single-threaded sequential tar-shard writer with per-shard index parts."""

    def __init__(self) -> None:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_PARTS.mkdir(parents=True, exist_ok=True)
        # a .part shard from a crashed run has no index rows -> discard; its
        # post_ids will be refetched (resume-by-index makes this safe)
        for stale in IMAGES_DIR.glob(f"{SHARD_PREFIX}-*.tar.part"):
            print(f"discarding unindexed partial shard {stale.name}")
            stale.unlink()
        existing = sorted(IMAGES_DIR.glob(f"{SHARD_PREFIX}-*.tar"))
        self.shard_no = (
            max(int(p.stem.rsplit("-", 1)[1]) for p in existing) + 1 if existing else 0
        )
        self.rows: list[dict] = []
        self._open_shard()

    def _open_shard(self) -> None:
        self.shard_id = f"{SHARD_PREFIX}-{self.shard_no:05d}"
        self.path = IMAGES_DIR / f"{self.shard_id}.tar.part"
        self.fobj = open(self.path, "wb", buffering=WRITE_BUFFER)
        self.tar = tarfile.open(fileobj=self.fobj, mode="w")

    def _close_shard(self) -> None:
        empty = self.tar.offset == 0
        self.tar.close()
        self.fobj.flush()
        import os

        os.fsync(self.fobj.fileno())
        self.fobj.close()
        if empty:
            self.path.unlink()  # don't leave zero-member shards behind
        else:
            self.path.rename(self.path.with_suffix(""))  # strip .part -> .tar
        self.flush_index()

    def flush_index(self) -> None:
        if not self.rows:
            return
        part = INDEX_PARTS / f"{self.shard_id}-{time.time_ns()}.parquet"
        pq.write_table(pa.Table.from_pylist(self.rows, schema=INDEX_SCHEMA), part)
        self.rows = []

    def add_image(self, res: Result) -> None:
        body = res.body
        assert body is not None and res.ext is not None
        digest = hashlib.sha256(body).hexdigest()
        member = f"{res.job.post_ids[0]}.{res.ext}"
        info = tarfile.TarInfo(name=member)
        info.size = len(body)
        info.mtime = int(time.time())
        self.tar.addfile(info, io.BytesIO(body))
        # addfile copies the TarInfo, so offset_data isn't set on ours; the
        # member's data ends at tar.offset minus its 512-padded length
        padded = ((len(body) + 511) // 512) * 512
        offset = self.tar.offset - padded
        for pid in res.job.post_ids:
            self.rows.append(self._row(pid, res, self.shard_id, offset, len(body), digest))
        if self.tar.offset >= SHARD_TARGET_BYTES:
            self._close_shard()
            self.shard_no += 1
            self._open_shard()

    def add_failure(self, res: Result) -> None:
        for pid in res.job.post_ids:
            self.rows.append(self._row(pid, res, None, None, None, None))
        if len(self.rows) >= 5000:
            self.flush_index()

    @staticmethod
    def _row(pid, res: Result, shard_id, offset, length, digest) -> dict:
        return {
            "post_id": pid,
            "url": res.job.url,
            "host": res.job.host,
            "shard_id": shard_id,
            "offset": offset,
            "length": length,
            "fetch_status": res.status,
            "http_status": res.http_status,
            "content_type": res.content_type,
            "sha256": digest,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    def close(self) -> None:
        self._close_shard()


INDEX_SCHEMA = pa.schema(
    [
        ("post_id", pa.string()),
        ("url", pa.string()),
        ("host", pa.string()),
        ("shard_id", pa.string()),
        ("offset", pa.int64()),
        ("length", pa.int64()),
        ("fetch_status", pa.string()),
        ("http_status", pa.int32()),
        ("content_type", pa.string()),
        ("sha256", pa.string()),
        ("ts", pa.string()),
    ]
)


class Lane:
    """Per-host(-group) worker pool with ramp-up and throttle full-stop."""

    def __init__(self, name: str, start_workers: int, max_workers: int,
                 min_interval: float, ramp_every: float | None):
        self.name = name
        self.queue: asyncio.Queue[Job | None] = asyncio.Queue()
        self.target_workers = start_workers
        self.max_workers = max_workers
        self.min_interval = min_interval  # per-worker pacing floor, jittered
        self.ramp_every = ramp_every
        self.throttle_events: deque[float] = deque()
        self.recent: deque[tuple[float, bool]] = deque()  # (ts, is_error)
        self.paused_until = 0.0
        self.stop_events = 0
        self.max_seen_workers = start_workers
        self.last_ramp = time.monotonic()

    def record(self, is_error: bool) -> None:
        now = time.monotonic()
        self.recent.append((now, is_error))
        while self.recent and now - self.recent[0][0] > 60:
            self.recent.popleft()

    def error_rate(self) -> float:
        if not self.recent:
            return 0.0
        return sum(1 for _, e in self.recent if e) / len(self.recent)

    def on_throttle(self) -> None:
        now = time.monotonic()
        self.throttle_events.append(now)
        while self.throttle_events and now - self.throttle_events[0] > 60:
            self.throttle_events.popleft()
        if len(self.throttle_events) >= 3 and now >= self.paused_until:
            self.paused_until = now + 300
            self.target_workers = max(1, self.target_workers // 2)
            self.stop_events += 1
            print(f"[{self.name}] 429/403 burst -> FULL STOP 300s, "
                  f"workers -> {self.target_workers}")

    def maybe_ramp(self) -> None:
        if self.ramp_every is None:
            return
        now = time.monotonic()
        if (now - self.last_ramp >= self.ramp_every and now >= self.paused_until
                and self.target_workers < self.max_workers
                and len(self.recent) >= 20 and self.error_rate() < 0.01):
            self.target_workers = min(self.max_workers, self.target_workers + 2)
            self.max_seen_workers = max(self.max_seen_workers, self.target_workers)
            self.last_ramp = now
            print(f"[{self.name}] ramp up -> {self.target_workers} workers")


async def fetch_one(session: aiohttp.ClientSession, job: Job) -> Result:
    t0 = time.monotonic()
    delay = 1.0
    last_exc: str | None = None
    for attempt in range(3):
        try:
            async with session.get(job.fetch_url, allow_redirects=True) as resp:
                if str(resp.url).endswith("/removed.png"):
                    return Result(job, "removed", resp.status, latency=time.monotonic() - t0)
                if resp.status in (404, 410):
                    return Result(job, f"http_{resp.status}", resp.status,
                                  latency=time.monotonic() - t0)
                if resp.status in (429, 403):
                    return Result(job, f"http_{resp.status}", resp.status,
                                  latency=time.monotonic() - t0)
                if resp.status != 200:
                    last_exc = f"http_{resp.status}"
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status)
                if (resp.content_length or 0) > MAX_IMAGE_BYTES:
                    return Result(job, "too_large", 200, resp.content_type,
                                  latency=time.monotonic() - t0)
                # NB: StreamReader.read(n) returns *up to* n bytes (first
                # buffered chunk) — resp.read() is the read-everything call
                body = await resp.read()
                if len(body) > MAX_IMAGE_BYTES:
                    return Result(job, "too_large", 200, resp.content_type,
                                  latency=time.monotonic() - t0)
                if not body:
                    return Result(job, "empty_body", 200, resp.content_type,
                                  latency=time.monotonic() - t0)
                ext = sniff_ext(body)
                if ext is None:
                    return Result(job, "non_image", 200, resp.content_type,
                                  latency=time.monotonic() - t0)
                return Result(job, "ok", 200, resp.content_type, body, ext,
                              time.monotonic() - t0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # timeout, conn reset, 5xx
            last_exc = type(exc).__name__
            if attempt < 2:
                await asyncio.sleep(delay + random.random())
                delay *= 4
    return Result(job, "error_temp", None, last_exc, latency=time.monotonic() - t0)


async def worker(lane: Lane, session: aiohttp.ClientSession,
                 out: asyncio.Queue, wid: int) -> None:
    while True:
        if wid >= lane.target_workers:  # ramped down; park until allowed again
            await asyncio.sleep(5)
            continue
        now = time.monotonic()
        if now < lane.paused_until:
            await asyncio.sleep(lane.paused_until - now + random.random())
            continue
        try:
            job = lane.queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        if job is None:
            return
        res = await fetch_one(session, job)
        throttled = res.status in ("http_429", "http_403")
        if throttled:
            lane.on_throttle()
        lane.record(res.status in ("error_temp", "http_429", "http_403"))
        lane.maybe_ramp()
        await out.put(res)
        pace = lane.min_interval * (0.5 + random.random())
        if pace > 0:
            await asyncio.sleep(pace)


def load_done() -> set[str]:
    done: set[str] = set()
    parts = list(INDEX_PARTS.glob("*.parquet"))
    for p in parts:
        t = pq.read_table(p, columns=["post_id", "fetch_status"])
        for pid, st in zip(t["post_id"].to_pylist(), t["fetch_status"].to_pylist()):
            if st in TERMINAL:
                done.add(pid)
    return done


def compact_index() -> None:
    parts = sorted(INDEX_PARTS.glob("*.parquet"))
    if not parts:
        return
    tables = [pq.read_table(p) for p in parts]
    merged = pa.concat_tables(tables)
    df = merged.to_pandas()
    # keep the newest row per post_id (retries supersede error_temp rows)
    df = df.sort_values("ts").drop_duplicates(subset="post_id", keep="last")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   OUT_ROOT / "index" / "shard_index.parquet")
    print(f"compacted index: {len(df):,} post_ids -> index/shard_index.parquet")


async def run(limit: int | None) -> None:
    pool = pd.read_parquet(POOL)
    order = np.random.default_rng(FETCH_ORDER_SEED).permutation(len(pool))
    pool = pool.iloc[order].reset_index(drop=True)
    if limit:
        pool = pool.iloc[:limit]

    done = load_done()
    writer = ShardWriter()

    # group by normalized URL; unsupported recorded immediately
    jobs_by_url: dict[str, Job] = {}
    n_skip = n_unsupported = 0
    for r in pool.itertuples():
        if r.post_id in done:
            n_skip += 1
            continue
        # pool host column is NaN (a float, and truthy!) where the URL regex
        # didn't match — sanitize before it reaches the string index column
        host = r.host if isinstance(r.host, str) else ""
        fetch_url, lane_name = normalize_url(r.img_url)
        if fetch_url is None:
            res = Result(Job(r.img_url, "", host, [r.post_id]), "unsupported_url")
            writer.add_failure(res)
            n_unsupported += 1
            continue
        if fetch_url in jobs_by_url:
            jobs_by_url[fetch_url].post_ids.append(r.post_id)
        else:
            jobs_by_url[fetch_url] = Job(r.img_url, fetch_url, host,
                                         [r.post_id], lane_name)

    lanes = {
        # pilot: 4 unpaced workers hit ~112 req/s with zero 429s — faster than
        # needed; the 0.1 s jittered floor keeps the ramped lane near 60 req/s
        "iredd": Lane("iredd", start_workers=4, max_workers=16,
                      min_interval=0.1, ramp_every=60),
        "other": Lane("other", start_workers=2, max_workers=2,
                      min_interval=1.0, ramp_every=None),
    }
    for job in jobs_by_url.values():
        lanes[job.lane].queue.put_nowait(job)

    total_jobs = len(jobs_by_url)
    print(f"pool rows: {len(pool):,}  already done: {n_skip:,}  "
          f"unsupported: {n_unsupported:,}  urls to fetch: {total_jobs:,} "
          f"(iredd {lanes['iredd'].queue.qsize():,} / other {lanes['other'].queue.qsize():,})")

    out: asyncio.Queue[Result] = asyncio.Queue()
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(limit=32, ttl_dns_cache=600)
    status_counts: Counter[str] = Counter()
    bytes_dl = 0
    t_start = time.monotonic()
    window: deque[tuple[float, int]] = deque()  # (ts, bytes) for rolling rate
    metrics_f = open(METRICS, "a")

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers={"User-Agent": USER_AGENT}
    ) as session:
        tasks = [
            asyncio.create_task(worker(lanes["iredd"], session, out, wid))
            for wid in range(lanes["iredd"].max_workers)
        ] + [
            asyncio.create_task(worker(lanes["other"], session, out, wid))
            for wid in range(lanes["other"].max_workers)
        ]

        processed = 0
        last_report = time.monotonic()
        while processed < total_jobs:
            if all(t.done() for t in tasks) and out.empty():
                break  # queues drained
            try:
                res = await asyncio.wait_for(out.get(), timeout=5)
            except asyncio.TimeoutError:
                res = None
            if res is not None:
                processed += 1
                status_counts[res.status] += 1
                if res.status == "ok":
                    bytes_dl += len(res.body)
                    window.append((time.monotonic(), len(res.body)))
                    writer.add_image(res)
                else:
                    writer.add_failure(res)
            now = time.monotonic()
            while window and now - window[0][0] > 30:
                window.popleft()
            if now - last_report >= 30:
                last_report = now
                rate = len(window) / 30
                mbs = sum(b for _, b in window) / 30 / 1e6
                line = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "processed": processed,
                    "total": total_jobs,
                    "req_per_s_30s": round(rate, 2),
                    "mb_per_s_30s": round(mbs, 2),
                    "iredd_workers": lanes["iredd"].target_workers,
                    "iredd_err_rate_60s": round(lanes["iredd"].error_rate(), 4),
                    "status_counts": dict(status_counts),
                    "bytes_total": bytes_dl,
                }
                metrics_f.write(json.dumps(line) + "\n")
                metrics_f.flush()
                print(f"{processed:,}/{total_jobs:,}  {rate:.1f} req/s  {mbs:.1f} MB/s  "
                      f"workers={lanes['iredd'].target_workers}  "
                      f"err60s={lanes['iredd'].error_rate():.1%}  {dict(status_counts)}")

        for t in tasks:
            t.cancel()

    writer.close()
    elapsed = time.monotonic() - t_start
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "run_summary",
        "processed": processed,
        "elapsed_s": round(elapsed, 1),
        "avg_req_per_s": round(processed / elapsed, 2) if elapsed else None,
        "bytes_total": bytes_dl,
        "avg_mb_per_s": round(bytes_dl / elapsed / 1e6, 2) if elapsed else None,
        "status_counts": dict(status_counts),
        "iredd_max_workers_reached": lanes["iredd"].max_seen_workers,
        "iredd_full_stops": lanes["iredd"].stop_events,
        "other_full_stops": lanes["other"].stop_events,
    }
    metrics_f.write(json.dumps(summary) + "\n")
    metrics_f.close()
    print("RUN SUMMARY:", json.dumps(summary, indent=2))
    compact_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot: only the first N pool rows (fixed shuffled order)")
    args = ap.parse_args()
    asyncio.run(run(args.limit))


if __name__ == "__main__":
    main()
