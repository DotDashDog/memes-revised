"""Integrity check of the reviewer's raw extract (ref_extract/), written just before a
machine shutdown. Checks, without trusting raw_pass_summary.json alone:

- every shard opens and its footer reads; no leftover .tmp shards;
- shard row counts sum to the summary's non-blank line count;
- line_no runs 0..N-1 with no gap, repeat or reordering across all shards;
- sum(nbytes) + number of newlines == decompressed byte count in the summary
  (every decompressed byte is accounted for by some extracted line);
- no line failed to parse (all j_* fields null would mean a bad/non-object line);
- every line has an id and a created_utc token;
- the 5,000-line seeded sample (written at the very end of the pass) agrees with the
  extract at the sampled line numbers (byte length and id).

    uv run python experiments/wp16/review/verify_extract.py
"""
import glob
import json
import os

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = "experiments/WP-16/review"
shards = sorted(glob.glob(f"{OUT}/ref_extract/part-*.parquet"))
tmp = glob.glob(f"{OUT}/ref_extract/*.tmp")
summ = json.load(open(f"{OUT}/raw_pass_summary.json"))
res = {"n_shards": len(shards), "leftover_tmp_files": tmp,
       "shard_names_contiguous": [os.path.basename(s) for s in shards] == [f"part-{i:04d}.parquet" for i in range(len(shards))]}

rows, nbytes_sum, expect, problems = 0, 0, 0, []
null_id = null_cu = all_null = 0
line_nos_parts, nbytes_parts, ids_parts = [], [], []
for s in shards:
    t = pq.read_table(s)  # full read: decodes every page, so a corrupt shard raises here
    ln = t.column("line_no").to_numpy()
    if len(ln) and not (ln[0] == expect and np.all(np.diff(ln) == 1)):
        problems.append({"shard": s, "first": int(ln[0]), "expected_first": expect})
    expect = int(ln[-1]) + 1 if len(ln) else expect
    rows += t.num_rows
    nb = t.column("nbytes").to_numpy()
    nbytes_sum += int(nb.sum())
    null_id += t.column("j_id").null_count
    null_cu += t.column("j_created_utc").null_count
    allnull = None
    for c in t.column_names:
        if c.startswith("j_"):
            m = pc.is_null(t.column(c))
            allnull = m if allnull is None else pc.and_(allnull, m)
    all_null += pc.sum(allnull).as_py() or 0
    line_nos_parts.append(ln); nbytes_parts.append(nb); ids_parts.append(t.column("j_id"))
    res.setdefault("rows_per_shard", []).append(t.num_rows)

nb_all = np.concatenate(nbytes_parts)

res.update({
    "rows_total": rows, "summary_non_blank_lines": summ["non_blank_lines"],
    "rows_match_summary": rows == summ["non_blank_lines"],
    "line_no_problems": problems,
    "last_line_no": expect - 1,
    "sum_nbytes": nbytes_sum, "newlines": summ["newline_bytes"],
    "sum_nbytes_plus_newlines": nbytes_sum + summ["newline_bytes"],
    "summary_decompressed_bytes": summ["decompressed_bytes"],
    "bytes_reconcile": nbytes_sum + summ["newline_bytes"] == summ["decompressed_bytes"],
    "rows_missing_id_key": null_id, "rows_missing_created_utc_key": null_cu,
    "rows_with_every_field_null(bad json)": int(all_null),
})

# sample consistency
line_nos = json.load(open(f"{OUT}/sample_line_nos.json"))
raws = open(f"{OUT}/sample_raw.jsonl", "rb").read().split(b"\n")
if raws and raws[-1] == b"":
    raws.pop()
import pyarrow as pa
ids_cat = pa.chunked_array(ids_parts)
ok_len = sum(len(r) == int(nb_all[i]) for r, i in zip(raws, line_nos))
samp_ids = ids_cat.take(pa.array(line_nos)).to_pylist()
ok_id = sum(json.dumps(json.loads(r)["id"]) == sid for r, sid in zip(raws, samp_ids))
res["sample"] = {"lines": len(raws), "line_nos": len(line_nos), "distinct_line_nos": len(set(line_nos)),
                 "byte_length_matches_extract": ok_len, "id_matches_extract": ok_id}
json.dump(res, open(f"{OUT}/verify_extract_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
