"""Post-id uniqueness and null/empty ids, parquet and raw extract (fast, Arrow-only)."""
import glob, json
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
OUT = "experiments/WP-16/review"
pid = pq.read_table("dataset_saves/reddit/submissions-2025-12/submissions.parquet", columns=["post_id", "subreddit"])
vc = pc.value_counts(pid.column("post_id")); c = vc.field("counts").to_numpy()
rid = pa.chunked_array([pq.read_table(s, columns=["j_id"]).column(0) for s in sorted(glob.glob(f"{OUT}/ref_extract/part-*.parquet"))])
rvc = pc.value_counts(rid); rc = rvc.field("counts").to_numpy()
lens = pc.value_counts(pc.utf8_length(pid.column("post_id")))
res = {"parquet_rows": pid.num_rows, "parquet_distinct_ids": len(vc), "parquet_ids_seen_more_than_once": int((c > 1).sum()),
       "parquet_null_ids": pid.column("post_id").null_count,
       "parquet_empty_ids": int(pc.sum(pc.equal(pid.column("post_id"), "")).as_py() or 0),
       "id_length_distribution": {str(v["values"]): v["counts"] for v in lens.to_pylist()},
       "raw_distinct_id_tokens": len(rvc), "raw_ids_seen_more_than_once": int((rc > 1).sum()),
       "subreddit_values": {str(v["values"]): v["counts"] for v in pc.value_counts(pid.column("subreddit")).to_pylist()}}
json.dump(res, open(f"{OUT}/ids_check_results.json", "w"), indent=1); print(json.dumps(res, indent=1))
