"""Locate the single img_url mismatch reported by compare.py and show the raw bytes."""
import glob, json
import pyarrow.parquet as pq
OUT = "experiments/WP-16/review"
PQ = "dataset_saves/reddit/submissions-2025-12/submissions.parquet"
par = pq.read_table(PQ, columns=["img_url", "post_id"])
pu = par.column("img_url"); pid = par.column("post_id")
off = 0; found = []
for s in sorted(glob.glob(f"{OUT}/ref_extract/part-*.parquet")):
    t = pq.read_table(s, columns=["j_url"]); toks = t.column(0).to_pylist()
    pv = pu.slice(off, len(toks)).to_pylist()
    for i, (tk, p) in enumerate(zip(toks, pv)):
        r = None if tk is None else json.loads(tk)
        if not (r == p and type(r) is type(p)):
            found.append({"row": off + i, "post_id": pid[off + i].as_py(), "raw_token": tk,
                          "raw_repr": repr(r), "parquet_repr": repr(p),
                          "raw_type": type(r).__name__, "parquet_type": type(p).__name__})
    off += len(toks)
json.dump(found, open(f"{OUT}/img_url_mismatch.json", "w"), indent=1)
print(json.dumps(found, indent=1))
