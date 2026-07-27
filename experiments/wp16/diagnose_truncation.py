"""WP-16 step 1: why did the 2025-01 ingestion stop at 2020-03-11?

Replays the legacy `memes_base/reddit_dataset.py` block-splitting against the 2023-12
archive and shows that processing input chunks 0-2 of 10 reproduces the existing parquet
exactly (4,121,367 predicted vs 4,121,376 actual, 2017-02-13 -> 2020-03-11).

Also enumerates the blocks silently dropped by the strict-inequality bug in `in_bounds`.

    uv run python experiments/wp16/diagnose_truncation.py

Pass 1 streams the whole 31.5 GB archive (~5 min); results cache to --cache.
"""
import argparse
import datetime as dt
import json
import os

import zstandard

ARCHIVE = "/home/linus/memes-revised/reddit_data/reddit/subreddits23/memes_submissions.zst"
KEY = '"created_utc":'


# --- verbatim from memes_base/reddit_dataset.py, so the replay is faithful ---

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def chunk_bounds(l, n):
    bounds = [0]
    for chunk in range(n):
        if chunk < l % n:
            bounds.append(bounds[-1] + l // n + 1)
        else:
            bounds.append(bounds[-1] + l // n)
    return bounds


def in_bounds(i, process_chunks, cb):
    for chunk in process_chunks:
        if i > cb[chunk] and i < cb[chunk + 1]:   # <-- the bug: `>` should be `>=`
            return True
    return False


# ---------------------------------------------------------------------------

def _ts(line):
    i = line.find(KEY)
    if i < 0:
        return None
    j = i + len(KEY)
    while line[j] in ' "':
        j += 1
    k = j
    while line[k] not in ',}"':
        k += 1
    try:
        return int(float(line[j:k]))
    except ValueError:
        return None


def map_blocks(path):
    """Block index -> (row count, first date, last date), using the legacy 2**27 split."""
    out = []
    fh = open(path, "rb")
    reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(fh)
    buffer, i = "", 0
    while True:
        chunk = read_and_decode(reader, 2 ** 27, (2 ** 29) * 2)
        if not chunk:
            break
        lines = (buffer + chunk).split("\n")
        buffer = lines[-1]
        block = lines[:-1]
        stamps = [t for t in (_ts(l) for l in block) if t]
        out.append({
            "i": i, "n": len(block),
            "lo": dt.datetime.fromtimestamp(min(stamps), dt.UTC).strftime("%Y-%m-%d") if stamps else None,
            "hi": dt.datetime.fromtimestamp(max(stamps), dt.UTC).strftime("%Y-%m-%d") if stamps else None,
        })
        i += 1
    reader.close()
    fh.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--cache", default="/tmp/wp16_blocks.json")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--parquet-rows", type=int, default=4_121_376)
    args = ap.parse_args()

    if os.path.exists(args.cache):
        blocks = json.load(open(args.cache))
    else:
        blocks = map_blocks(args.archive)
        json.dump(blocks, open(args.cache, "w"))

    b = {x["i"]: x for x in blocks}
    n_blocks = len(blocks)
    print(f"n_blocks = {n_blocks}   rows = {sum(x['n'] for x in blocks):,}\n")

    cb = chunk_bounds(n_blocks, args.chunks)
    print("bounds:", cb)

    dropped = [i for i in range(n_blocks) if not in_bounds(i, range(args.chunks), cb)]
    lost = sum(b[i]["n"] for i in dropped)
    print(f"\nblocks silently dropped by in_bounds ({len(dropped)}, {lost:,} rows, "
          f"{100*lost/sum(x['n'] for x in blocks):.2f}% of the archive):")
    for i in dropped:
        print(f"  block {i:>3}  {b[i]['n']:>7,} rows  {b[i]['lo']} -> {b[i]['hi']}")

    print("\n--- hypothesis: only input chunks 0,1,2 were processed ---")
    sel = [i for i in range(n_blocks) if in_bounds(i, [0, 1, 2], cb)]
    rows = sum(b[i]["n"] for i in sel)
    print(f"predicted: {rows:,} rows, {b[sel[0]]['lo']} -> {b[sel[-1]]['hi']}")
    print(f"actual   : {args.parquet_rows:,} rows, 2017-02-13 -> 2020-03-11")
    print(f"delta    : {rows - args.parquet_rows:,} rows "
          f"(line-splitting at block edges)")


if __name__ == "__main__":
    main()
