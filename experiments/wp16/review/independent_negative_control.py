"""Negative control for independent_pass.py stage 2: the same digest comparison must flag
rows when (a) positions are shifted by one and (b) a single field is perturbed."""
import json, sys
sys.path.insert(0, "experiments/wp16/review")
import numpy as np, pyarrow.parquet as pq
import independent_pass as ip
digs = np.load(f"{ip.OUT}/independent_digests.npy")
t = pq.ParquetFile(ip.PQ).read_row_group(0, columns=ip.COLS).slice(0, 20000)
cols = [t.column(c).to_pylist() for c in ip.COLS]
rows = [[c[i] for c in cols] for i in range(t.num_rows)]
same = sum(ip.digest(r) == int(digs[i]) for i, r in enumerate(rows))
shift = sum(ip.digest(r) == int(digs[i + 1]) for i, r in enumerate(rows))
pert = 0
for i, r in enumerate(rows):
    r2 = list(r); r2[4] = (r2[4] or 0) + 1  # score +1
    pert += ip.digest(r2) == int(digs[i])
res = {"rows": len(rows), "aligned_matches": same, "shifted_by_one_matches": shift, "score_plus_one_matches": pert}
json.dump(res, open(f"{ip.OUT}/independent_negative_control.json", "w"), indent=1); print(res)
