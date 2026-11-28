"""Show the duplicated post id: its parquet row positions and every column of each copy."""
import json
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
OUT = "experiments/WP-16/review"
PQ = "dataset_saves/reddit/submissions-2025-12/submissions.parquet"
pid = pq.read_table(PQ, columns=["post_id"]).column(0)
vc = pc.value_counts(pid); d = vc.field("values").filter(pa.array(vc.field("counts").to_numpy() > 1))
pos = np.nonzero(pc.is_in(pid, value_set=d).to_numpy(zero_copy_only=False))[0].tolist()
pf = pq.ParquetFile(PQ); t = pf.read().take(pa.array(pos)) if False else None
rows = []
off = 0
for rg in range(pf.metadata.num_row_groups):
    n = pf.metadata.row_group(rg).num_rows
    for p in pos:
        if off <= p < off + n:
            r = pf.read_row_group(rg).slice(p - off, 1).to_pylist()[0]
            rows.append({"row": p, **{k: (v[:200] if isinstance(v, str) else v) for k, v in r.items()}})
    off += n
diff = [k for k in rows[0] if k != "row" and any(r[k] != rows[0][k] for r in rows)]
res = {"duplicated_ids": d.to_pylist(), "rows": rows, "columns_that_differ": diff}
json.dump(res, open(f"{OUT}/duplicate_id_detail.json", "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
