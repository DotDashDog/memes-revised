"""Second, independent pass over the raw archive (third reviewer).

The first reviewer's extract used python-zstandard (the same binding as the ingest) and
Python's json (the same parser as the ingest). This pass uses a different zstd binding
(pyzstd, file-level reader that follows all frames) and a different JSON parser (orjson,
strict: rejects NaN/Infinity, invalid UTF-8 and lone surrogates). For every line it applies
the ingest's documented per-field rules, written here from scratch, to produce the row the
parquet *should* contain, and stores an 8-byte digest of that expected row plus the
expected created_utc. Stage 2 recomputes the digest from the actual parquet row at the same
position and compares all 11,782,247 rows.

Also records: SHA-256 of the decompressed stream, decompressed byte and newline counts,
and per-line maximum length.

    uv run --with pyzstd --with orjson python experiments/wp16/review/independent_pass.py ARCHIVE
"""
import collections
import hashlib
import io
import json
import sys
import time

import numpy as np
import orjson
import pyarrow.parquet as pq
import pyzstd

PQ = "dataset_saves/reddit/submissions-2025-12/submissions.parquet"
OUT = "experiments/WP-16/review"
COLS = ["is_comment", "img_url", "created_utc", "author", "score", "link", "title", "text",
        "post_id", "subreddit", "num_comments", "removed_by_category", "over_18"]


def digest(vals):
    return int.from_bytes(hashlib.blake2b(orjson.dumps(vals), digest_size=8).digest(), "little")


def expected_row(o, stats):
    """The ingest's documented mapping, reimplemented independently."""
    cu = o.get("created_utc")
    stats[f"created_utc_type:{type(cu).__name__}"] += 1
    if isinstance(cu, bool) or cu is None:
        raise ValueError("created_utc unusable")
    if isinstance(cu, str):
        cu_f = float(cu)
    else:
        cu_f = cu
    if cu_f != int(cu_f):
        stats["created_utc_fractional"] += 1
    ts = int(cu_f)  # truncation toward zero, as int(float(x)) in the ingest

    def as_int(v, name, allow_none):
        if v is None and allow_none:
            return None
        if isinstance(v, int):  # includes bool, as isinstance does in the ingest
            if isinstance(v, bool):
                stats[f"{name}_bool"] += 1
            return int(v)
        stats[f"{name}_non_int:{type(v).__name__}"] += 1
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    ov = o.get("over_18")
    if not isinstance(ov, bool):
        stats[f"over_18_non_bool:{type(ov).__name__}"] += 1
    for f in ("title", "selftext", "permalink"):
        v = o.get(f, "<<MISSING>>")
        if v == "<<MISSING>>":
            stats[f"{f}_key_missing"] += 1
        elif v is None:
            stats[f"{f}_null"] += 1
        elif not isinstance(v, str):
            stats[f"{f}_non_str:{type(v).__name__}"] += 1
    for f in ("url", "author", "id", "subreddit", "removed_by_category"):
        v = o.get(f)
        if v is not None and not isinstance(v, str):
            stats[f"{f}_non_str:{type(v).__name__}"] += 1
    row = [False, o.get("url"), ts, o.get("author"), as_int(o.get("score"), "score", False),
           o.get("permalink") or "", o.get("title") or "", o.get("selftext") or "",
           o.get("id"), o.get("subreddit"), as_int(o.get("num_comments"), "num_comments", True),
           o.get("removed_by_category"), ov if isinstance(ov, bool) else None]
    return row, ts


def stage1(archive):
    sha = hashlib.sha256()
    stats = collections.Counter()
    digs, tss = [], []
    nbytes = nlines = maxlen = 0
    t0 = time.time()
    # Large raw read buffer: small reads over the WSL /mnt/d mount ran at ~1/6 of disk speed.
    with open(archive, "rb", buffering=1 << 25) as raw_fh, \
            pyzstd.ZstdFile(raw_fh, "rb", level_or_option={pyzstd.DParameter.windowLogMax: 31}) as zf:
        br = io.BufferedReader(zf, buffer_size=1 << 24)
        for line in br:
            nbytes += len(line)
            sha.update(line)
            if line.endswith(b"\n"):
                nlines += 1
                body = line[:-1]
            else:
                stats["final_line_without_newline"] += 1
                body = line
            if not body.strip():
                stats["blank"] += 1
                continue
            maxlen = max(maxlen, len(body))
            try:
                o = orjson.loads(body)
            except orjson.JSONDecodeError:
                stats["orjson_rejects"] += 1
                o = json.loads(body)  # fall back so the row still aligns; counted above
            if not isinstance(o, dict):
                stats["non_object"] += 1
                digs.append(0); tss.append(-1)
                continue
            try:
                row, ts = expected_row(o, stats)
                digs.append(digest(row)); tss.append(ts)
            except (ValueError, TypeError, OverflowError):
                stats["would_be_dropped_or_unencodable"] += 1
                digs.append(0); tss.append(-1)
            if len(digs) % 1_000_000 == 0:
                print(f"  {len(digs):,} lines  {nbytes/1e9:.1f} GB  {time.time()-t0:.0f}s", flush=True)
    summ = {"decompressor": f"pyzstd {pyzstd.__version__} (libzstd {pyzstd.zstd_version})",
            "parser": f"orjson {orjson.__version__}",
            "decompressed_bytes": nbytes, "newlines": nlines, "rows": len(digs),
            "max_line_bytes": maxlen, "sha256_decompressed": sha.hexdigest(),
            "stats": dict(sorted(stats.items())), "elapsed_s": round(time.time() - t0)}
    np.save(f"{OUT}/independent_digests.npy", np.array(digs, dtype=np.uint64))
    np.save(f"{OUT}/independent_ts.npy", np.array(tss, dtype=np.int64))
    json.dump(summ, open(f"{OUT}/independent_pass_stage1.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
    return summ


def stage2():
    digs = np.load(f"{OUT}/independent_digests.npy")
    tss = np.load(f"{OUT}/independent_ts.npy")
    pf = pq.ParquetFile(PQ)
    off, bad, bad_ts, examples = 0, 0, 0, []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=COLS)
        cols = [t.column(c).to_pylist() for c in COLS]
        for i in range(t.num_rows):
            row = [c[i] for c in cols]
            j = off + i
            if j >= len(digs):
                bad += 1
                continue
            if row[2] != tss[j]:
                bad_ts += 1
            if digest(row) != int(digs[j]):
                bad += 1
                if len(examples) < 10:
                    examples.append({"row": j, "post_id": row[8]})
        off += t.num_rows
    res = {"parquet_rows": off, "raw_rows": int(len(digs)),
           "rows_whose_13_values_differ_from_expected": bad,
           "rows_whose_created_utc_differs": bad_ts, "examples": examples}
    json.dump(res, open(f"{OUT}/independent_pass_stage2.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] != "--stage2-only":
        stage1(sys.argv[1])
    stage2()
