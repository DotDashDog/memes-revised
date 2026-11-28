"""Seeded-sample field check (reused from the first reviewer; changed to read only the
matching rows and to add a positional cross-check).

Seeded-sample field check: 5,000 untouched raw archive lines (reservoir sample, seed
20261127, drawn in extract_raw.py) are parsed here from scratch and compared with the
parquet row(s) carrying the same post id. Lookup is by id, not by position.

    uv run python experiments/wp16/review/sample_check.py
"""
import collections
import json
from decimal import Decimal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PQ = "dataset_saves/reddit/submissions-2025-12/submissions.parquet"
OUT = "experiments/WP-16/review"

raws = [json.loads(l) for l in open(f"{OUT}/sample_raw.jsonl", "rb")]
ids = [r["id"] for r in raws]
# Read only the matching rows (the draft read the whole 13-column table, several GB).
hit = pq.read_table(PQ, filters=pc.field("post_id").isin(pa.array(ids))).to_pylist()
# Positional cross-check: the sampled line numbers should be the parquet row positions.
line_nos = json.load(open(f"{OUT}/sample_line_nos.json"))
pid_col = pq.read_table(PQ, columns=["post_id"]).column(0)
pos_ids = pid_col.take(pa.array(line_nos)).to_pylist()
positional_agree = sum(a == b for a, b in zip(pos_ids, ids))
by = collections.defaultdict(list)
for row in hit:
    by[row["post_id"]].append(row)

MAP = {"img_url": "url", "author": "author", "score": "score", "link": "permalink", "title": "title",
       "text": "selftext", "post_id": "id", "subreddit": "subreddit", "num_comments": "num_comments",
       "removed_by_category": "removed_by_category", "over_18": "over_18"}
found = collections.Counter(); cells = 0
mism = collections.Counter(); ex = collections.defaultdict(list)
strlen = 0; nonascii = 0
for r in raws:
    rows = by.get(r["id"], [])
    found[len(rows)] += 1
    if not rows:
        continue
    # if an id occurs more than once, require at least one parquet copy to match fully
    best = None
    for p in rows:
        bad = []
        for col, f in MAP.items():
            rv = r.get(f)
            if not (p[col] == rv and type(p[col]) is type(rv)):
                cls = "null/missing -> ''" if (rv is None and p[col] == "") else "UNEXPLAINED"
                bad.append((col, cls, rv, p[col]))
        cu = r.get("created_utc")
        exact = Decimal(cu) if isinstance(cu, str) else Decimal(repr(cu))
        if Decimal(p["created_utc"]) != exact:
            bad.append(("created_utc", "fraction dropped" if int(exact) == p["created_utc"] else "UNEXPLAINED", cu, p["created_utc"]))
        if p["is_comment"] is not False:
            bad.append(("is_comment", "UNEXPLAINED", False, p["is_comment"]))
        if best is None or len(bad) < len(best):
            best = bad
    cells += 13
    for col, cls, rv, pv in best:
        mism[f"{col}: {cls}"] += 1
        if len(ex[f"{col}: {cls}"]) < 3:
            ex[f"{col}: {cls}"].append({"id": r["id"], "raw": repr(rv)[:120], "parquet": repr(pv)[:120]})
    for f in ("title", "selftext", "url", "author", "permalink"):
        v = r.get(f)
        if isinstance(v, str):
            strlen += 1
            nonascii += any(ord(c) > 127 for c in v)
out = {"sample_size": len(raws), "seed": 20261127,
       "parquet_rows_found_per_sampled_id": {str(k): v for k, v in sorted(found.items())},
       "sampled_line_number_equals_parquet_row_position": positional_agree,
       "cells_compared": cells, "mismatching_cells": sum(mism.values()),
       "mismatches_by_column_and_class": dict(mism), "examples": dict(ex),
       "string_cells_compared_codepoint_for_codepoint": strlen, "of_which_contain_non_ascii": nonascii}
json.dump(out, open(f"{OUT}/sample_check_results.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
