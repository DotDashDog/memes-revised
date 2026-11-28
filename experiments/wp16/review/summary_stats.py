"""Faster, mostly-Arrow recomputation of the descriptive parts of compare.py (raw JSON types
per field, author's counters, what img_url holds by year, ordering, range). Written because
the first full compare.py run was killed after its cell-by-cell comparison but before it
wrote its summary; also serves as a cross-check of compare.py's own summary.

    uv run python experiments/wp16/review/summary_stats.py
"""
import glob
import json
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PQ = "dataset_saves/reddit/submissions-2025-12/submissions.parquet"
OUT = "experiments/WP-16/review"
SH = sorted(glob.glob(f"{OUT}/ref_extract/part-*.parquet"))
FIELDS = ["url", "created_utc", "author", "score", "permalink", "title", "selftext", "id",
          "subreddit", "num_comments", "removed_by_category", "over_18"]


def col(name):
    return pa.chunked_array([pq.read_table(s, columns=[name]).column(0) for s in SH]).combine_chunks()


def jtype(tok):
    """JSON type of raw tokens (json.dumps output) without parsing them."""
    first = pc.utf8_slice_codeunits(tok, 0, 1)
    isnum = pc.is_in(first, value_set=pa.array(list("-0123456789")))
    isfloat = pc.and_(isnum, pc.or_(pc.match_substring(tok, "."),
                                    pc.or_(pc.match_substring(tok, "e"), pc.match_substring(tok, "E"))))
    out = {}
    out["<key missing>"] = tok.null_count
    for lab, mask in [("str", pc.equal(first, '"')), ("null", pc.equal(tok, "null")),
                      ("bool", pc.or_(pc.equal(tok, "true"), pc.equal(tok, "false"))),
                      ("float", isfloat), ("int", pc.and_(isnum, pc.invert(isfloat))),
                      ("list/object", pc.is_in(first, value_set=pa.array(["[", "{"])))]:
        out[lab] = int(pc.sum(mask).as_py() or 0)
    return {k: v for k, v in out.items() if v}


res = {"raw_json_types_per_field": {f: jtype(col(f"j_{f}")) for f in FIELDS}}

t = pq.read_table(PQ, columns=["created_utc", "img_url", "link", "removed_by_category"])
ts = t.column("created_utc").to_numpy()
yr = ts.astype("M8[s]").astype("M8[Y]").astype(int) + 1970
res["range"] = {"min_utc": str(ts.min().astype("M8[s]")), "max_utc": str(ts.max().astype("M8[s]")),
                "nonpositive": int((ts <= 0).sum()),
                "outside_2008-07-01_to_2026-01-01": int(((ts < np.datetime64("2008-07-01", "s").astype(int))
                                                         | (ts >= np.datetime64("2026-01-01", "s").astype(int))).sum()),
                "rows_earlier_than_previous_row_in_file": int((np.diff(ts) < 0).sum()),
                "largest_backward_step_s": int(-np.diff(ts).min())}

# img_url content
urls = t.column("img_url").to_pylist()
links = t.column("link").to_pylist()
is_self = [json.loads(x) if x is not None else None for x in col("x_is_self").to_pylist()]
is_gal = [json.loads(x) if x is not None else None for x in col("x_is_gallery").to_pylist()]
IMG = (".jpg", ".jpeg", ".png", ".gif", ".gifv", ".webp", ".bmp")
kinds = {}
auth = {"url_null": 0, "url_unparseable": 0, "host_i.redd.it": 0, "host_imgur_any": 0}
for i, u in enumerate(urls):
    y = int(yr[i])
    if u is None:
        k = "no URL (null)"; auth["url_null"] += 1
    else:
        try:
            h = (urlsplit(u).hostname or "").lower()
        except ValueError:
            h = ""; auth["url_unparseable"] += 1
        h = h[4:] if h.startswith("www.") else h
        auth["host_i.redd.it"] += h == "i.redd.it"
        auth["host_imgur_any"] += h.endswith("imgur.com")
        path = u.split("?")[0].split("#")[0].lower()
        if h == "i.redd.it":
            k = "i.redd.it direct image"
        elif is_gal[i] is True or ("/gallery/" in u and h.endswith("reddit.com")):
            k = "reddit gallery page (images not in parquet)"
        elif h == "v.redd.it":
            k = "v.redd.it video"
        elif is_self[i] is True or (links[i] and u.endswith(links[i])):
            k = "the post's own page (text post)"
        elif path.endswith(IMG):
            k = "other direct image (imgur / image extension)"
        else:
            k = "other link"
    kinds.setdefault(y, {}).setdefault(k, 0)
    kinds[y][k] += 1
tot = {}
for y in kinds:
    for k, v in kinds[y].items():
        tot[k] = tot.get(k, 0) + v
res["img_url_content"] = {"kinds_total": dict(sorted(tot.items(), key=lambda x: -x[1])),
                          "kinds_by_year": {y: kinds[y] for y in sorted(kinds)}}
res["author_counters_recomputed"] = auth
json.dump(res, open(f"{OUT}/summary_stats.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "img_url_content"}, indent=1))
print(json.dumps(res["img_url_content"]["kinds_total"], indent=1))
