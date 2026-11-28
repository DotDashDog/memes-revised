"""Reviewer's single pass over the raw r/memes submissions archive.

Independent of experiments/wp16/ingest.py: uses zstandard's decompressobj API (not
stream_reader), follows the stream across zstd frames, checks that the last frame is
complete, counts newline bytes separately from the line splitter, and keeps every field of
interest as its raw JSON token so that null / missing / "" / int / float / str stay
distinguishable. Also keeps a seeded reservoir sample of untouched raw lines.

    uv run python experiments/wp16/review/extract_raw.py ARCHIVE OUTDIR
"""
import collections
import hashlib
import json
import os
import random
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

SEED = 20261127
K = 5000
FIELDS = ["url", "created_utc", "author", "score", "permalink", "title", "selftext", "id",
          "subreddit", "num_comments", "removed_by_category", "over_18"]
EXTRA = ["is_self", "domain", "is_gallery", "post_hint", "url_overridden_by_dest",
         "retrieved_on", "retrieved_utc", "crosspost_parent"]
PRESENCE = ["media_metadata", "gallery_data", "preview", "media"]

archive, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

names = (["line_no", "nbytes"] + [f"j_{f}" for f in FIELDS] + [f"x_{f}" for f in EXTRA]
         + [f"has_{f}" for f in PRESENCE])
schema = pa.schema([("line_no", pa.int64()), ("nbytes", pa.int32())]
                   + [(f"j_{f}", pa.string()) for f in FIELDS]
                   + [(f"x_{f}", pa.string()) for f in EXTRA]
                   + [(f"has_{f}", pa.bool_()) for f in PRESENCE])
# Output is sharded (one parquet file per 500k lines, written atomically via rename) so an
# interrupted pass leaves readable partial results rather than a footer-less file.
SHARD_DIR = os.path.join(outdir, "ref_extract")
os.makedirs(SHARD_DIR, exist_ok=True)
for old in os.listdir(SHARD_DIR):
    os.remove(os.path.join(SHARD_DIR, old))
cols = {n: [] for n in names}
shard_no = 0


def flush():
    global cols, shard_no
    if cols["line_no"]:
        tmp = os.path.join(SHARD_DIR, f"part-{shard_no:04d}.parquet.tmp")
        pq.write_table(pa.table(cols, schema=schema), tmp, compression="zstd")
        os.replace(tmp, tmp[:-4])
        shard_no += 1
        cols = {n: [] for n in names}
        with open(os.path.join(outdir, "raw_pass_progress.json"), "w") as fh:
            json.dump({"lines_done": line_no, "shards": shard_no, "elapsed_s": round(time.time() - t0)}, fh)


def tok(obj, key):
    """Raw value as an ASCII JSON token; None (-> parquet null) only if the key is absent."""
    if key not in obj:
        return None
    return json.dumps(obj[key])


rng = random.Random(SEED)
reservoir = []  # (line_no, raw bytes)
stats = collections.Counter()
keys = collections.Counter()
bad_examples, nonobj_examples, first_lines, last_lines = [], [], [], collections.deque(maxlen=3)

sha = hashlib.sha256()
dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
frames = 0
comp_bytes = decomp_bytes = newline_bytes = 0
line_no = 0  # 0-based index over NON-BLANK lines, to align with parquet rows
t0 = time.time()


def handle(raw):
    global line_no
    if not raw.strip():
        stats["blank_lines"] += 1
        return
    i = line_no
    line_no += 1
    if len(first_lines) < 3:
        first_lines.append(raw.decode("utf-8", "replace"))
    last_lines.append(raw)
    # reservoir sampling (algorithm R)
    if len(reservoir) < K:
        reservoir.append((i, raw))
    else:
        j = rng.randint(0, i)
        if j < K:
            reservoir[j] = (i, raw)
    if raw.endswith(b"\r"):
        stats["lines_ending_cr"] += 1
    try:
        obj = json.loads(raw)
    except Exception:
        stats["bad_json"] += 1
        if len(bad_examples) < 50:
            bad_examples.append([i, raw[:2000].decode("utf-8", "replace")])
        obj = None
    if obj is not None and not isinstance(obj, dict):
        stats["non_object"] += 1
        if len(nonobj_examples) < 50:
            nonobj_examples.append([i, raw[:2000].decode("utf-8", "replace")])
        obj = None
    cols["line_no"].append(i)
    cols["nbytes"].append(len(raw))
    if obj is None:
        for n in names[2:]:
            cols[n].append(None)
        if len(cols["line_no"]) >= 500_000:
            flush()
        return
    keys.update(obj.keys())
    for f in FIELDS:
        cols[f"j_{f}"].append(tok(obj, f))
    for f in EXTRA:
        cols[f"x_{f}"].append(tok(obj, f))
    for f in PRESENCE:
        v = obj.get(f)
        cols[f"has_{f}"].append(bool(v))
    if len(cols["line_no"]) >= 500_000:
        flush()


tail = b""
with open(archive, "rb") as fh:
    dobj = dctx.decompressobj()
    frames = 0
    pending = b""
    frame_open = False  # has the current decompressor received any bytes
    while True:
        data = pending or fh.read(1 << 23)
        if not pending:
            if not data:
                break
            sha.update(data)
            comp_bytes += len(data)
        pending = b""
        if not frame_open:
            frames += 1          # a fresh decompressor is starting a new frame
            frame_open = True
        out = dobj.decompress(data)
        if out:
            decomp_bytes += len(out)
            newline_bytes += out.count(b"\n")
            parts = (tail + out).split(b"\n")
            tail = parts.pop()
            before = line_no
            for p in parts:
                handle(p)
            if line_no // 1_000_000 != before // 1_000_000:
                print(f"  {line_no:,} lines  {decomp_bytes/1e9:.1f} GB  {time.time()-t0:.0f}s", flush=True)
        if dobj.eof:
            stats["frames_completed"] += 1
            pending = dobj.unused_data
            dobj = dctx.decompressobj()
            frame_open = False
    last_frame_complete = not frame_open
if tail:
    stats["final_line_without_newline"] += 1
    handle(tail)
flush()

reservoir.sort()
with open(os.path.join(outdir, "sample_raw.jsonl"), "wb") as fh:
    for _, raw in reservoir:
        fh.write(raw + b"\n")
with open(os.path.join(outdir, "sample_line_nos.json"), "w") as fh:
    json.dump([i for i, _ in reservoir], fh)

summary = {
    "archive": archive,
    "sha256_compressed": sha.hexdigest(),
    "compressed_bytes": comp_bytes,
    "decompressed_bytes": decomp_bytes,
    "newline_bytes": newline_bytes,
    "non_blank_lines": line_no,
    "zstd_frames_started": frames,
    "zstd_frames_completed": stats["frames_completed"],
    "last_frame_complete": last_frame_complete,
    "stats": dict(stats),
    "key_counts": dict(keys.most_common()),
    "bad_json_examples": bad_examples,
    "non_object_examples": nonobj_examples,
    "first_lines": [s[:500] for s in first_lines],
    "last_lines": [s.decode("utf-8", "replace")[:500] for s in last_lines],
    "seed": SEED, "sample_size": len(reservoir),
    "elapsed_s": round(time.time() - t0, 1),
}
with open(os.path.join(outdir, "raw_pass_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k not in ("key_counts", "first_lines", "last_lines")}, indent=1))
