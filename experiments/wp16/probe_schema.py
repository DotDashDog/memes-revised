"""WP-16 step 2: which fields drift type or go missing across the archive's history?

`created_utc` is known to be a string in old records and a float in newer ones. WP-16
§Method 2 requires checking whether other fields do the same rather than assuming that is
the only one -- a parser that assumes one type fails silently partway through 15 years of
data.

Reports, per year, the observed Python type and missing/null count for every field the
ingest depends on.

    uv run python experiments/wp16/probe_schema.py --archive <path>
"""
import argparse
import collections
import json
import time

import zstandard

FIELDS = ["url", "created_utc", "author", "score", "permalink", "title", "selftext",
          "id", "subreddit", "num_comments", "is_self", "over_18", "removed_by_category"]


def stream_lines(path, chunk_size=1 << 26):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", default="/tmp/wp16_schema.json")
    args = ap.parse_args()

    # year -> field -> typename -> count ; plus year -> field -> "MISSING"/"NULL"
    types = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    per_year = collections.Counter()
    bad_json = 0
    no_ts = 0
    total = 0

    t0 = time.time()
    for raw in stream_lines(args.archive):
        total += 1
        try:
            obj = json.loads(raw)
        except Exception:
            bad_json += 1
            continue
        cu = obj.get("created_utc")
        try:
            year = time.gmtime(int(float(cu))).tm_year
        except (TypeError, ValueError):
            no_ts += 1
            year = 0
        per_year[year] += 1
        for f in FIELDS:
            if f not in obj:
                types[year][f]["<missing>"] += 1
            elif obj[f] is None:
                types[year][f]["<null>"] += 1
            else:
                types[year][f][type(obj[f]).__name__] += 1
        if total % 2_000_000 == 0:
            print(f"  {total:,} lines  {time.time()-t0:.0f}s", flush=True)

    print(f"\ntotal lines {total:,}   bad json {bad_json:,}   unparseable created_utc {no_ts:,}")
    print(f"elapsed {time.time()-t0:.0f}s\n")

    print(f"{'field':22} {'year':>6}  type distribution")
    for f in FIELDS:
        prev = None
        for year in sorted(types):
            dist = types[year][f]
            sig = tuple(sorted(dist))
            if sig != prev:
                shown = "  ".join(f"{k}:{v:,}" for k, v in dist.most_common())
                print(f"{f:22} {year:>6}  {shown}")
                prev = sig

    json.dump({str(y): {f: dict(c) for f, c in d.items()} for y, d in types.items()},
              open(args.out, "w"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
