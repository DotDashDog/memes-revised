"""WP-16: stream a Reddit submissions .zst dump into parquet.

Replaces `memes_base/reddit_dataset.py`, which was rewritten rather than debugged -- see
WP-16 §"Step 1 result" for the five defects, the worst being that it silently captured 36%
of the archive and reported success.

Design rules, each one a direct response to a defect in the legacy path:

- **Nothing is dropped silently.** Every input line lands in exactly one bucket: written,
  or one of the named drop reasons. `rows_in == rows_out + sum(drops)` is asserted at the
  end. A run that cannot reconcile exits non-zero.
- **Bounded memory.** Rows stream into fixed-size parquet row groups; nothing accumulates.
- **Type drift is coerced explicitly and counted.** `created_utc` is `int` (2008-13,
  2016-22), `str` (2014-15) and `float` (2023+), mixed within 2015 and 2023. `url` is null
  in 14,843 records. Every coercion and null is counted, not assumed away.
- **Re-runnable against a different archive.** The cut is a CLI argument, not a constant.

    uv run python experiments/wp16/ingest.py \
        --archive ~/reddit_dl/reddit/subreddits25/memes_submissions.zst \
        --out dataset_saves/reddit/submissions-2025-12

Schema note: the eight columns the existing pipeline reads (`is_comment, img_url,
created_utc, author, score, link, title, text`) are preserved exactly, so `sample.py` and
`run_semantic_embedding_cache.py` keep working. Five columns are *added*:
`post_id` (required to key ADR 0015's shard index), `subreddit` (required by ADR 0016's
multi-subreddit corpus), `num_comments` (an ADR 0007 engagement-signal candidate),
`removed_by_category` and `over_18`. The change is additive only -- no column is renamed,
retyped or removed -- so it cannot break a downstream reader.
"""
import argparse
import collections
import json
import os
import sys
import time
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

SCHEMA = pa.schema([
    ("is_comment", pa.bool_()),
    ("img_url", pa.string()),
    ("created_utc", pa.int64()),
    ("author", pa.string()),
    ("score", pa.int64()),
    ("link", pa.string()),
    ("title", pa.string()),
    ("text", pa.string()),
    ("post_id", pa.string()),
    ("subreddit", pa.string()),
    ("num_comments", pa.int64()),
    ("removed_by_category", pa.string()),
    ("over_18", pa.bool_()),
])


def stream_lines(path, chunk_size=1 << 26):
    """Yield raw ndjson lines. The tail of each read is carried into the next."""
    dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
    with open(path, "rb") as fh, dctx.stream_reader(fh) as r:
        tail = b""
        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            lines = (tail + chunk).split(b"\n")
            tail = lines.pop()
            yield from lines
        if tail.strip():
            yield tail


def host_of(url):
    if not url:
        return "<none>"
    try:
        h = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return "<unparseable>"
    if not h:
        return "<none>"
    if h.startswith("www."):
        h = h[4:]
    return h


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".gifv", ".webp", ".bmp")


def looks_like_image(url, host):
    """Host/extension heuristic. Never raises: some records carry malformed URLs
    (e.g. unbalanced brackets, which `urlsplit` rejects as 'Invalid IPv6 URL')."""
    if not url or host == "<unparseable>":
        return False
    if host in ("i.redd.it", "i.imgur.com"):
        return True
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        return False
    return path.endswith(IMAGE_EXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--row-group", type=int, default=250_000)
    ap.add_argument("--expect-rows", type=int, default=None,
                    help="known raw line count; mismatch is a hard failure")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, "submissions.parquet")
    is_comment = "comments" in os.path.basename(args.archive)

    drops = collections.Counter()
    coercions = collections.Counter()
    per_month = collections.Counter()
    host_year = collections.defaultdict(collections.Counter)
    image_year = collections.Counter()
    rows_year = collections.Counter()

    cols = {name: [] for name in SCHEMA.names}
    rows_in = rows_out = 0
    writer = pq.ParquetWriter(out_file, SCHEMA, compression="zstd")
    t0 = time.time()

    def flush():
        nonlocal cols
        if not cols["created_utc"]:
            return
        writer.write_table(pa.table(cols, schema=SCHEMA))
        cols = {name: [] for name in SCHEMA.names}

    for raw in stream_lines(args.archive):
        rows_in += 1
        try:
            obj = json.loads(raw)
        except Exception:
            drops["bad_json"] += 1
            continue

        cu = obj.get("created_utc")
        if cu is None:
            drops["created_utc_missing"] += 1
            continue
        if isinstance(cu, str):
            coercions["created_utc_from_str"] += 1
        elif isinstance(cu, float):
            coercions["created_utc_from_float"] += 1
        try:
            ts = int(float(cu))
        except (TypeError, ValueError):
            drops["created_utc_unparseable"] += 1
            continue

        g = time.gmtime(ts)
        year = g.tm_year

        url = obj.get("url")
        if url is None:
            coercions["url_null"] += 1
        host = host_of(url)
        if host == "<unparseable>":
            coercions["url_unparseable"] += 1
        host_year[year][host] += 1
        if looks_like_image(url, host):
            image_year[year] += 1
        rows_year[year] += 1
        per_month[f"{year:04d}-{g.tm_mon:02d}"] += 1

        score = obj.get("score")
        if not isinstance(score, int):
            coercions["score_coerced"] += 1
            try:
                score = int(float(score))
            except (TypeError, ValueError):
                score = None

        ncom = obj.get("num_comments")
        if ncom is not None and not isinstance(ncom, int):
            coercions["num_comments_coerced"] += 1
            try:
                ncom = int(float(ncom))
            except (TypeError, ValueError):
                ncom = None

        if is_comment:
            link = f"/r/{obj.get('subreddit')}/comments/{(obj.get('link_id') or '____')[3:]}/_/{obj.get('id')}/"
            title, text = "", obj.get("body") or ""
        else:
            link = obj.get("permalink") or ""
            title, text = obj.get("title") or "", obj.get("selftext") or ""

        cols["is_comment"].append(is_comment)
        cols["img_url"].append(url)
        cols["created_utc"].append(ts)
        cols["author"].append(obj.get("author"))
        cols["score"].append(score)
        cols["link"].append(link)
        cols["title"].append(title)
        cols["text"].append(text)
        cols["post_id"].append(obj.get("id"))
        cols["subreddit"].append(obj.get("subreddit"))
        cols["num_comments"].append(ncom)
        cols["removed_by_category"].append(obj.get("removed_by_category"))
        cols["over_18"].append(obj.get("over_18") if isinstance(obj.get("over_18"), bool) else None)
        rows_out += 1

        if len(cols["created_utc"]) >= args.row_group:
            flush()
        if rows_in % 2_000_000 == 0:
            print(f"  {rows_in:,} in / {rows_out:,} out  {time.time()-t0:.0f}s", flush=True)

    flush()
    writer.close()
    elapsed = time.time() - t0

    dropped = sum(drops.values())
    report = {
        "archive": args.archive,
        "output": out_file,
        "elapsed_s": round(elapsed, 1),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped": dropped,
        "drops": dict(drops),
        "coercions": dict(coercions),
        "rows_per_year": dict(sorted(rows_year.items())),
        "rows_per_month": dict(sorted(per_month.items())),
        "image_rows_per_year": dict(sorted(image_year.items())),
        "hosts_per_year": {str(y): dict(c.most_common(15)) for y, c in sorted(host_year.items())},
    }
    with open(os.path.join(args.out, "ingest_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    print(f"\nrows in      {rows_in:,}")
    print(f"rows out     {rows_out:,}")
    print(f"rows dropped {dropped:,}  {dict(drops) or '{}'}")
    print(f"coercions    {dict(coercions)}")
    print(f"elapsed      {elapsed:.0f}s -> {out_file}")

    ok = True
    if rows_in != rows_out + dropped:
        print(f"\nRECONCILIATION FAILED: {rows_in:,} != {rows_out:,} + {dropped:,}")
        ok = False
    if args.expect_rows is not None and rows_in != args.expect_rows:
        print(f"\nLINE COUNT MISMATCH: read {rows_in:,}, expected {args.expect_rows:,}")
        ok = False
    if not ok:
        sys.exit(1)
    print("\nreconciled: rows_in == rows_out + rows_dropped")


if __name__ == "__main__":
    main()
