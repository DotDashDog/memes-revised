"""Monthly submission counts straight off the .zst dumps — input to the WP-16 gate plot.

Extracts `created_utc` by byte-scanning rather than `json.loads`, which is ~20x faster and
is all that's needed here. Note the schema hazard this sidesteps: `created_utc` is a
*string* in old records (`"1247100541"`) and a *float* in newer ones (`1704067195.0`), so
the value is parsed as `int(float(...))` and both forms work.

Unparseable lines are counted, not silently dropped — an unexplained `bad` count is a
finding, not rounding.
"""
import json, sys, time, zstandard as zstd
from collections import Counter

KEY = b'"created_utc":'

def monthly(path):
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    counts, tail, bad = Counter(), b"", 0
    with open(path, "rb") as fh, dctx.stream_reader(fh) as r:
        while True:
            chunk = r.read(1 << 24)
            if not chunk: break
            buf = tail + chunk
            lines = buf.split(b"\n")
            tail = lines.pop()
            for ln in lines:
                i = ln.find(KEY)
                if i < 0: bad += 1; continue
                j = i + len(KEY)
                while ln[j:j+1] in (b' ', b'"'): j += 1
                k = j
                while ln[k:k+1] not in (b',', b'}', b'"', b''): k += 1
                try:
                    t = int(float(ln[j:k]))
                except ValueError:
                    bad += 1; continue
                g = time.gmtime(t)
                counts[(g.tm_year, g.tm_mon)] += 1
    return counts, bad

out = {}
for label, path in [("2023cut", "/home/linus/memes-revised/reddit_data/reddit/subreddits23/memes_submissions.zst"),
                    ("2025cut", "/home/linus/reddit_dl/reddit/subreddits25/memes_submissions.zst")]:
    c, bad = monthly(path)
    out[label] = {f"{y:04d}-{m:02d}": n for (y, m), n in sorted(c.items())}
    print(f"{label}: {sum(c.values()):,} parsed, {bad} unparseable", flush=True)

json.dump(out, open("/home/linus/.claude/jobs/b2e9a557/tmp/monthly.json", "w"))
print("written")
