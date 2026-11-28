"""WP-18: check the six extra subreddit tables were ingested completely (coordinator request after
the multi-frame zstd truncation risk found in experiments/wp16/ingest.py).

Independent of the ingester's reader: pyzstd's decompressor is fed the raw file and restarted at
every frame boundary, so it reads across frames. Counts non-blank lines, records the last
non-blank record's created_utc and the max created_utc over all records, and compares them with
the parquet row count and timestamps. Also walks the frame structure with
experiments/wp16/review/zstd_frames.py (header-level, no library).

    uv run --with pyzstd --with orjson python experiments/wp18/verify_ingest_frames.py
"""
import json
import subprocess
import sys
from pathlib import Path

import orjson
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyzstd

SRC = Path("/home/linus/reddit_dl_wp18/reddit/subreddits25")
DST = Path("/home/linus/memes-revised/dataset_saves/reddit/submissions-2025-12")
OUT = Path("/home/linus/memes-revised/experiments/WP-18")
SUBS = ["dankmemes", "meirl", "comedyheaven", "whenthe", "196", "shitposting"]


def ts(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def scan(path: Path) -> dict:
    n_lines = n_bad = frames = 0
    last_rec_ts = max_ts = None
    rest = b""
    d = pyzstd.ZstdDecompressor()
    fed = False
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            data = chunk
            while data:
                out = d.decompress(data)
                fed = True
                rest += out
                if d.eof:  # frame ended: start a new decompressor on the unused input
                    frames += 1
                    data = d.unused_data
                    d = pyzstd.ZstdDecompressor()
                    fed = False
                else:
                    data = b""
                *lines, rest = rest.split(b"\n")
                for ln in lines:
                    if not ln.strip():
                        continue
                    n_lines += 1
                    try:
                        t = ts(orjson.loads(ln).get("created_utc"))
                    except orjson.JSONDecodeError:
                        n_bad += 1
                        continue
                    last_rec_ts = t
                    if t is not None and (max_ts is None or t > max_ts):
                        max_ts = t
    tail_unterminated = bool(rest.strip())
    if tail_unterminated:
        n_lines += 1
        try:
            last_rec_ts = ts(orjson.loads(rest).get("created_utc"))
        except orjson.JSONDecodeError:
            n_bad += 1
    return {"nonblank_lines": n_lines, "unparseable_lines": n_bad, "frames_completed": frames,
            "trailing_incomplete_frame": bool(fed and not d.eof),
            "last_record_created_utc": last_rec_ts, "max_created_utc": max_ts,
            "unterminated_tail": tail_unterminated}


def main():
    res = {}
    for sub in SUBS:
        a = SRC / f"{sub}_submissions.zst"
        if not a.exists():
            res[sub] = {"archive": "MISSING"}
            continue
        fw = OUT / f"verify_frames_{sub}.json"
        subprocess.run([sys.executable, "experiments/wp16/review/zstd_frames.py", str(a), str(fw)], check=True,
                       cwd="/home/linus/memes-revised")
        s = scan(a)
        t = pq.read_table(DST / f"{sub}_submissions.parquet", columns=["created_utc"])
        col = t.column("created_utc")
        s.update({"parquet_rows": t.num_rows, "parquet_max_created_utc": int(pc.max(col).as_py()),
                  "parquet_last_row_created_utc": int(col[t.num_rows - 1].as_py()),
                  "frame_walk": json.loads(fw.read_text())})
        s["rows_match"] = s["parquet_rows"] == s["nonblank_lines"]
        s["max_ts_match"] = s["parquet_max_created_utc"] == s["max_created_utc"]
        res[sub] = s
        print(sub, {k: v for k, v in s.items() if k != "frame_walk"}, flush=True)
    (OUT / "verify_ingest_frames.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
