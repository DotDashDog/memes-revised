"""Print selected fields of one raw archive line (0-based non-blank line number), decoded with
pyzstd, plus the SHA-256 of that line, to settle a disagreement between data sets.

    uv run --with pyzstd --with orjson python experiments/wp16/review/show_raw_line.py ARCHIVE LINE_NO
"""
import hashlib, io, json, sys
import orjson, pyzstd
archive, target = sys.argv[1], int(sys.argv[2])
with open(archive, "rb", buffering=1 << 25) as fh, \
        pyzstd.ZstdFile(fh, "rb", level_or_option={pyzstd.DParameter.windowLogMax: 31}) as zf:
    for i, line in enumerate(io.BufferedReader(zf, buffer_size=1 << 24)):
        if i == target:
            o = orjson.loads(line)
            out = {"line_no": i, "nbytes": len(line.rstrip(b"\n")), "sha256": hashlib.sha256(line).hexdigest(),
                   **{k: o.get(k) for k in ("id", "url", "permalink", "title", "created_utc", "is_self")}}
            i_url = line.find(b'"url":')
            out["url_bytes_hex"] = line[i_url:i_url + 100].hex()
            print(json.dumps(out, indent=1)); break
