"""How many posts have a created_utc far from that of the posts with neighbouring ids?

Reddit ids are a global counter assigned at creation, so a post's created_utc should be
close to its id-neighbours'. compare.py's 'timestamp_vs_id_order' metric (rows more than a
day earlier than the running maximum) is inflated by any single far-future outlier, so this
counts outliers directly: |ts - median ts of the 50 posts either side in id order|.

    uv run python experiments/wp16/review/id_time_outliers.py
"""
import json

import numpy as np
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

OUT = "experiments/WP-16/review"
t = pq.read_table("dataset_saves/reddit/submissions-2025-12/submissions.parquet",
                  columns=["post_id", "created_utc"])
ts = t.column("created_utc").to_numpy()
ids = t.column("post_id").to_pylist()
idn = np.array([int(x, 36) for x in ids], dtype=np.int64)
o = np.argsort(idn, kind="stable")
s = ts[o]
W = 50
pad = np.concatenate([np.full(W, s[0]), s, np.full(W, s[-1])])
med = np.empty(len(s), dtype=np.float64)
CH = 500_000
for a in range(0, len(s), CH):  # chunked to bound memory
    b = min(len(s), a + CH)
    win = sliding_window_view(pad[a:b + 2 * W], 2 * W + 1)
    med[a:b] = np.median(win, axis=1)
dev = s - med
yr = s.astype("M8[s]").astype("M8[Y]").astype(int) + 1970
res = {"window": f"{W} posts either side in id order"}
for lab, thr in [("1h", 3600), ("1d", 86400), ("30d", 30 * 86400)]:
    later, earlier = dev > thr, dev < -thr
    res[f"later_than_neighbours_by_more_than_{lab}"] = int(later.sum())
    res[f"earlier_than_neighbours_by_more_than_{lab}"] = int(earlier.sum())
m = np.abs(dev) > 86400
res["more_than_1d_off_by_year_of_created_utc"] = {int(y): int(c) for y, c in zip(*np.unique(yr[m], return_counts=True))}
idx = np.nonzero(m)[0][:10]
res["examples"] = [{"post_id": ids[o[i]], "created_utc": str(np.datetime64(int(s[i]), "s")),
                    "neighbour_median": str(np.datetime64(int(med[i]), "s"))} for i in idx]
json.dump(res, open(f"{OUT}/id_time_outliers.json", "w"), indent=1)
print(json.dumps(res, indent=1))
