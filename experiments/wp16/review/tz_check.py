"""Time-zone check for the three raw created_utc types (int, str, float).

The parquet stores raw epoch seconds with no conversion (shown elsewhere), so the remaining
question is whether the *archive's* str- or float-typed timestamps sit on a different clock
from the int-typed ones (e.g. a local-time string stored as if UTC). Reddit post ids are a
global base-36 counter assigned at creation, so in id order timestamps should be
non-decreasing apart from small jitter. At each point where the raw type switches between
neighbours in id order, a time-zone offset would show as a jump of a whole number of hours.

    uv run python experiments/wp16/review/tz_check.py
"""
import glob
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PQ = "dataset_saves/reddit/submissions-2025-12/submissions.parquet"
OUT = "experiments/WP-16/review"
shards = sorted(glob.glob(f"{OUT}/ref_extract/part-*.parquet"))
tok = pa.chunked_array([pq.read_table(s, columns=["j_created_utc"]).column(0) for s in shards])
is_str = pc.starts_with(tok, '"').to_numpy(zero_copy_only=False)
is_float = (pc.match_substring(tok, ".").to_numpy(zero_copy_only=False)
            | pc.match_substring(tok, "e").to_numpy(zero_copy_only=False)) & ~is_str
typ = np.where(is_str, 1, np.where(is_float, 2, 0))  # 0 int, 1 str, 2 float
names = {0: "int", 1: "str", 2: "float"}
t = pq.read_table(PQ, columns=["post_id", "created_utc"])
ts = t.column("created_utc").to_numpy()
idn = np.array([int(x, 36) for x in t.column("post_id").to_pylist()], dtype=np.int64)
o = np.argsort(idn, kind="stable")
ts_o, ty_o = ts[o], typ[o]
d = np.diff(ts_o)
a, b = ty_o[:-1], ty_o[1:]
res = {"type_counts": {names[k]: int((typ == k).sum()) for k in names}}
pairs = {}
for x in names:
    for y in names:
        m = (a == x) & (b == y)
        if m.sum() == 0:
            continue
        v = d[m]
        pairs[f"{names[x]}->{names[y]}"] = {
            "n": int(m.sum()), "median_s": float(np.median(v)),
            "p01_s": float(np.percentile(v, 1)), "p99_s": float(np.percentile(v, 99)),
            "share_backward_more_than_30min": float((v < -1800).mean()),
            "share_forward_more_than_30min": float((v > 1800).mean()),
        }
res["consecutive_in_id_order"] = pairs
# per year: year range of each type (calendar years in UTC)
yr = ts.astype("M8[s]").astype("M8[Y]").astype(int) + 1970
res["years_by_type"] = {names[k]: {int(y): int(c) for y, c in zip(*np.unique(yr[typ == k], return_counts=True))}
                        for k in names}
json.dump(res, open(f"{OUT}/tz_check_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
