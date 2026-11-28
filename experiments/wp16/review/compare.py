"""Full-population comparison of the ingested parquet against the reviewer's raw extract.

Row i of the parquet is compared with non-blank line i of the archive (the ingest writes
rows in archive order with a single writer, so position is the right key; a lost or
inserted row would show up as a cascade of post_id mismatches). Strict rule: the parquet
value must equal the raw JSON value and have the same Python type (JSON null and a missing
key both map to parquet null; timestamps must equal the raw number exactly). Every
mismatch is classified and counted.

Streams in 500k-row chunks so peak memory stays around 2-3 GB. Writes
compare_results.json and monthly_counts.csv to experiments/WP-16/review/.

    uv run python experiments/wp16/review/compare.py

Rewritten by the second reviewer from the first reviewer's draft. Changes: streaming
instead of whole-column Python lists (the draft needed well over 8 GB); monthly counts for
the raw side now use every raw line, not just the first min(N, M) (the draft would have
hidden extra raw lines from the monthly comparison); string mismatches are sub-classified
(truncation, unicode normalisation, whitespace); author's counters are recomputed.
"""
import collections
import glob
import json
import re
import unicodedata
from decimal import Decimal
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PQ = "dataset_saves/reddit/submissions-2025-12/submissions.parquet"
OUT = "experiments/WP-16/review"
SHARDS = sorted(glob.glob(f"{OUT}/ref_extract/part-*.parquet"))
assert SHARDS, "run extract_raw.py first"

res = {}
pf = pq.ParquetFile(PQ)
N = pf.metadata.num_rows
M = sum(pq.ParquetFile(s).metadata.num_rows for s in SHARDS)
res["parquet_rows"], res["raw_non_blank_lines"] = N, M
res["parquet_row_groups"] = pf.metadata.num_row_groups
res["parquet_schema"] = str(pf.schema_arrow)
res["parquet_key_value_metadata"] = {k.decode(): v.decode()[:300] for k, v in (pf.metadata.metadata or {}).items()}
print("rows parquet", N, "raw", M, flush=True)


def parquet_chunks():
    """Yield parquet tables re-sliced to match each ref shard's length."""
    buf, rg = None, 0
    for s in SHARDS:
        need = pq.ParquetFile(s).metadata.num_rows
        while (buf is None or buf.num_rows < need) and rg < pf.metadata.num_row_groups:
            t = pf.read_row_group(rg)
            rg += 1
            buf = t if buf is None else pa.concat_tables([buf, t])
        if buf is None:
            yield s, None
            continue
        take = min(need, buf.num_rows)
        yield s, buf.slice(0, take)
        buf = buf.slice(take)
    leftover = (buf.num_rows if buf is not None else 0) + sum(
        pf.metadata.row_group(i).num_rows for i in range(rg, pf.metadata.num_row_groups))
    res["parquet_rows_beyond_raw"] = leftover


def tname(v):
    return "<missing>" if v is MISSING else type(v).__name__


MISSING = object()


def dec(tok):
    return MISSING if tok is None else json.loads(tok)


def nn(v):  # missing -> None for value comparison
    return None if v is MISSING else v


def cls_string(r, p):
    if isinstance(p, str) and p == "" and r is MISSING:
        return "raw key missing -> stored as empty string"
    if isinstance(p, str) and p == "" and r is None:
        return "raw null -> stored as empty string"
    if isinstance(r, str) and isinstance(p, str):
        if r.startswith(p):
            return "TRUNCATED"
        if unicodedata.normalize("NFC", r) == unicodedata.normalize("NFC", p):
            return "UNICODE NORMALISED"
        if r.strip() == p.strip():
            return "WHITESPACE CHANGED"
    return "UNEXPLAINED"


def cls_int(r, p):
    if isinstance(r, bool):
        return f"raw bool {r} -> {p!r}"
    if isinstance(r, float):
        return "raw float -> int" if p == int(r) else "UNEXPLAINED"
    if isinstance(r, str):
        return "raw str -> int" if p is not None else "raw str -> null"
    return "UNEXPLAINED"


def cls_bool(r, p):
    if p is None and r not in (None, MISSING) and not isinstance(r, bool):
        return f"raw non-bool ({type(r).__name__}) -> null"
    return "UNEXPLAINED"


PAIRS = [  # parquet column, raw field, classifier
    ("post_id", "id", None), ("author", "author", None), ("subreddit", "subreddit", None),
    ("removed_by_category", "removed_by_category", None),
    ("score", "score", cls_int), ("num_comments", "num_comments", cls_int),
    ("over_18", "over_18", cls_bool),
    ("link", "permalink", cls_string), ("title", "title", cls_string),
    ("text", "selftext", cls_string), ("img_url", "url", cls_string),
]
cmp = {c: {"raw_types": collections.Counter(), "mismatches": 0, "classes": collections.Counter(),
           "examples": collections.defaultdict(list)} for c, _, _ in PAIRS}
cmp["is_comment"] = {"raw_types": collections.Counter(), "mismatches": 0, "classes": collections.Counter(),
                     "examples": collections.defaultdict(list)}

raw_ts = np.full(M, -1, dtype=np.int64)
par_ts_all = np.full(N, -1, dtype=np.int64)
ctype_year = collections.Counter()
frac = collections.Counter()
ts_other = collections.Counter()
ts_mism = 0
ts_offsets = collections.Counter()
ts_examples = []

# author's counters, recomputed
auth = collections.Counter()
host_tot = collections.Counter()
kind_year = collections.defaultdict(collections.Counter)
null_url = collections.Counter()
amp = collections.Counter()
ws = 0
lag = collections.defaultdict(list)
text_extra = collections.Counter()
title_maxlen = text_maxlen_p = text_maxlen_r = 0
nonascii_cells = 0

HOST = re.compile(r"^[a-z][a-z0-9+.\-]*://(?:[^/?#@]*@)?([^/?#:]*)", re.I)
IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".gifv", ".webp", ".bmp")

off = 0
for shard, ptab in parquet_chunks():
    rt = pq.read_table(shard)
    m = rt.num_rows
    n = 0 if ptab is None else ptab.num_rows
    R = {c: rt.column(c).to_pylist() for c in rt.column_names}
    P = {c: ptab.column(c).to_pylist() for c in ptab.column_names} if n else {}
    assert R["line_no"][0] == off, (shard, R["line_no"][0], off)

    # ---- created_utc (all raw lines, even beyond the parquet)
    for i in range(m):
        t = R["j_created_utc"][i]
        v = dec(t)
        if v is MISSING or v is None or isinstance(v, bool):
            ts_other[tname(v)] += 1
            continue
        if isinstance(v, int):
            raw_ts[off + i] = v
            ct = "int"
        else:
            try:
                d = Decimal(v) if isinstance(v, str) else Decimal(t)  # exact, no float round trip
            except Exception:
                ts_other["unparseable " + tname(v)] += 1
                continue
            ct = "str" if isinstance(v, str) else "float"
            if d != d.to_integral_value():
                frac[ct] += 1
            raw_ts[off + i] = int(d.to_integral_value(rounding="ROUND_FLOOR"))
        ctype_year[(ct, int(np.datetime64(int(raw_ts[off + i]), "s").astype("M8[Y]").astype(int) + 1970))] += 1
    if n:
        pts = np.asarray(ptab.column("created_utc").to_numpy(zero_copy_only=False))
        par_ts_all[off:off + n] = pts
        d = pts - raw_ts[off:off + n]
        bad = np.nonzero(d != 0)[0]
        ts_mism += len(bad)
        ts_offsets.update(d[bad].tolist())
        for i in bad[:5 - len(ts_examples)] if len(ts_examples) < 5 else []:
            ts_examples.append({"row": off + int(i), "raw_token": R["j_created_utc"][i], "parquet": int(pts[i])})

    # ---- plain columns
    for col, f, classify in PAIRS:
        st = cmp[col]
        toks = R[f"j_{f}"]
        for i in range(m):
            st["raw_types"][tname(dec(toks[i]))] += 1
        for i in range(n):
            r = dec(toks[i])
            p = P[col][i]
            rv = nn(r)
            if p == rv and type(p) is type(rv):
                continue
            st["mismatches"] += 1
            c = classify(r, p) if classify else "UNEXPLAINED"
            st["classes"][c] += 1
            if len(st["examples"][c]) < 5:
                st["examples"][c].append({"row": off + i, "post_id": P["post_id"][i],
                                          "raw_token": (toks[i] or "<key missing>")[:200], "parquet": repr(p)[:200]})
    for i in range(n):
        if P["is_comment"][i] is not False:
            cmp["is_comment"]["mismatches"] += 1
            cmp["is_comment"]["classes"][repr(P["is_comment"][i])] += 1

    # ---- string extras
    for i in range(n):
        tt, tx = P["title"][i], P["text"][i]
        title_maxlen = max(title_maxlen, len(tt))
        text_maxlen_p = max(text_maxlen_p, len(tx))
        text_extra["title_empty"] += tt == ""
        text_extra["title_contains_NUL"] += "\x00" in tt
        text_extra["title_contains_&amp;/&lt;/&gt;"] += ("&amp;" in tt or "&lt;" in tt or "&gt;" in tt)
        text_extra["text_empty"] += tx == ""
        text_extra["text_[removed]"] += tx == "[removed]"
        text_extra["text_[deleted]"] += tx == "[deleted]"
        nonascii_cells += any(ord(ch) > 127 for ch in tt)
    for tok in R["j_selftext"]:
        v = dec(tok)
        if isinstance(v, str):
            text_maxlen_r = max(text_maxlen_r, len(v))

    # ---- recompute author's counters + what img_url holds
    for i in range(m):
        u = nn(dec(R["j_url"][i]))
        ct = R["j_created_utc"][i]
        cv = dec(ct)
        if isinstance(cv, str):
            auth["created_utc_from_str"] += 1
        elif isinstance(cv, float):
            auth["created_utc_from_float"] += 1
        if u is None:
            auth["url_null"] += 1
        else:
            try:
                h = (urlsplit(u).hostname or "").lower()
            except ValueError:
                auth["url_unparseable"] += 1
                h = None
            if h is not None:
                h = h[4:] if h.startswith("www.") else h
                auth[f"host_is_i.redd.it"] += h == "i.redd.it"
                auth[f"host_is_imgur(any subdomain)"] += h.endswith("imgur.com")
    for i in range(n):
        y = int(np.datetime64(int(par_ts_all[off + i]), "s").astype("M8[Y]").astype(int) + 1970)
        u = P["img_url"][i]
        is_self = nn(dec(R["x_is_self"][i]))
        is_gal = nn(dec(R["x_is_gallery"][i]))
        uod = nn(dec(R["x_url_overridden_by_dest"][i]))
        has_mm = R["has_media_metadata"][i]
        perma = P["link"][i]
        if u is None:
            k = "no URL (null)"
            null_url["total"] += 1
            null_url["of which url_overridden_by_dest is set"] += bool(uod)
            null_url["of which media_metadata present"] += bool(has_mm)
            null_url["of which removed_by_category set"] += P["removed_by_category"][i] is not None
            null_url["of which is_self true"] += is_self is True
        else:
            mm = HOST.match(u.strip())
            h = mm.group(1).lower() if mm else ""
            if h.startswith("www."):
                h = h[4:]
            host_tot[h or "<no host>"] += 1
            if u != u.strip():
                ws += 1
            path = u.split("?")[0].split("#")[0].lower()
            if "&amp;" in u:
                amp[h] += 1
            if h == "i.redd.it":
                k = "i.redd.it direct image"
            elif is_gal is True or ("/gallery/" in u and h.endswith("reddit.com")):
                k = "reddit gallery page (images not in parquet)"
            elif h == "v.redd.it":
                k = "v.redd.it video"
            elif is_self is True or (perma and u.endswith(perma)):
                k = "the post's own page (text post)"
            elif h.endswith("imgur.com") and path.endswith(IMG_EXT) or path.endswith(IMG_EXT):
                k = "other direct image (imgur / image extension)"
            else:
                k = "other link"
        kind_year[y][k] += 1
        if (off + i) % 37 == 0:
            ro = nn(dec(R["x_retrieved_on"][i]))
            ru = nn(dec(R["x_retrieved_utc"][i]))
            r = ro if ro is not None else ru
            if isinstance(r, (int, float)) and not isinstance(r, bool):
                lag[y].append((r - par_ts_all[off + i]) / 86400)
    off += m
    print(f"  compared through row {off:,}", flush=True)
    del R, P, rt, ptab

res["created_utc"] = {
    "raw_types_by_year": {t: {y: c for (tt, y), c in sorted(ctype_year.items()) if tt == t}
                          for t in sorted({t for t, _ in ctype_year})},
    "raw_values_with_fractional_seconds": dict(frac),
    "raw_missing_null_bool_or_unparseable": dict(ts_other),
    "rows_where_parquet_differs_from_raw": ts_mism,
    "distinct_nonzero_offsets_seconds": ts_offsets.most_common(10),
    "examples": ts_examples,
    "raw_min_utc": str(raw_ts[raw_ts >= 0].min().astype("M8[s]")),
    "raw_max_utc": str(raw_ts.max().astype("M8[s]")),
    "parquet_min_utc": str(par_ts_all.min().astype("M8[s]")),
    "parquet_max_utc": str(par_ts_all.max().astype("M8[s]")),
    "parquet_nonpositive": int((par_ts_all <= 0).sum()),
    "parquet_outside_2008-07-01_to_2026-01-01": int(((par_ts_all < np.datetime64("2008-07-01", "s").astype(int))
                                                    | (par_ts_all >= np.datetime64("2026-01-01", "s").astype(int))).sum()),
}
back = np.diff(par_ts_all) < 0
res["ordering"] = {"rows_earlier_than_previous_row": int(back.sum()),
                   "largest_backward_step_seconds": int(-np.diff(par_ts_all).min()) if back.any() else 0}

for col, st in cmp.items():
    res[col] = {"raw_types": dict(st["raw_types"]), "mismatches": st["mismatches"],
                "mismatch_classes": dict(st["classes"]), "examples": dict(st["examples"])}
    print(col, st["mismatches"], dict(st["classes"]), flush=True)


def monthly(ts):
    mo = ts[ts >= 0].astype("M8[s]").astype("M8[M]")
    u, c = np.unique(mo, return_counts=True)
    return {str(k): int(v) for k, v in zip(u, c)}


m_raw, m_par = monthly(raw_ts), monthly(par_ts_all)
m_auth = json.load(open("dataset_saves/reddit/submissions-2025-12/ingest_report.json"))["rows_per_month"]
months = sorted(set(m_raw) | set(m_par) | set(m_auth))
with open(f"{OUT}/monthly_counts.csv", "w") as fh:
    fh.write("month,raw_archive,parquet,author_report\n")
    for mo in months:
        fh.write(f"{mo},{m_raw.get(mo, 0)},{m_par.get(mo, 0)},{m_auth.get(mo, 0)}\n")
first, last = min(m_raw), max(m_raw)
all_months = np.arange(np.datetime64(first, "M"), np.datetime64(last, "M") + 1)
res["monthly"] = {
    "months_with_data_raw": len(m_raw), "months_with_data_parquet": len(m_par), "months_author_report": len(m_auth),
    "months_parquet_differs_from_raw": sum(m_raw.get(mo, 0) != m_par.get(mo, 0) for mo in months),
    "months_author_report_differs_from_raw": sum(m_raw.get(mo, 0) != m_auth.get(mo, 0) for mo in months),
    "max_abs_diff_parquet_vs_raw": max(abs(m_raw.get(mo, 0) - m_par.get(mo, 0)) for mo in months),
    "calendar_months_in_span": len(all_months),
    "empty_calendar_months_in_span": [str(x) for x in all_months if str(x) not in m_raw],
    "empty_calendar_months_from_2011_on": [str(x) for x in all_months if str(x) not in m_raw and str(x) >= "2011"],
    "total_raw": sum(m_raw.values()), "total_parquet": sum(m_par.values()),
}
yr = collections.Counter()
for mo, c in m_raw.items():
    yr[mo[:4]] += c
res["yearly_raw"] = dict(sorted(yr.items()))
auth_report = json.load(open("dataset_saves/reddit/submissions-2025-12/ingest_report.json"))
res["author_counters_recomputed"] = {"reviewer": dict(auth), "author": auth_report["coercions"],
                                     "author_yearly": auth_report["rows_per_year"]}

# ---- duplicates (C++ side, cheap)
pid = pq.read_table(PQ, columns=["post_id"]).column(0)
vc = pc.value_counts(pid)
cnt = vc.field("counts").to_numpy()
res["duplicate_ids_parquet"] = {"distinct_ids": len(vc), "ids_appearing_more_than_once": int((cnt > 1).sum()),
                                "extra_rows": int((cnt[cnt > 1] - 1).sum()),
                                "examples": [str(x) for x in vc.field("values").filter(pa.array(cnt > 1))[:10].to_pylist()]}
res["null_or_empty_ids_parquet"] = int(pid.null_count + pc.sum(pc.equal(pid, "")).as_py() or 0)
rid = pa.concat_arrays([pq.read_table(s, columns=["j_id"]).column(0).combine_chunks() for s in SHARDS])
rvc = pc.value_counts(rid)
rcnt = rvc.field("counts").to_numpy()
res["duplicate_ids_raw"] = {"distinct_tokens": len(rvc), "ids_appearing_more_than_once": int((rcnt > 1).sum()),
                            "extra_rows": int((rcnt[rcnt > 1] - 1).sum())}
dupset = vc.field("values").filter(pa.array(cnt > 1))
if len(dupset):
    rows_idx = np.nonzero(pc.is_in(pid, value_set=dupset).to_numpy(zero_copy_only=False))[0]
    tab = pq.read_table(PQ, filters=pc.field("post_id").isin(dupset))  # dup rows only (order not relied on)
    assert tab.num_rows == len(rows_idx)
    pos = collections.defaultdict(list)
    for r_i, v in zip(rows_idx.tolist(), pid.take(pa.array(rows_idx)).to_pylist()):
        pos[v].append(r_i)
    by = collections.defaultdict(list)
    for row in tab.to_pylist():
        by[row["post_id"]].append(row)
    gaps = [b - a for v in pos.values() for a, b in zip(v, v[1:])]
    res["duplicate_positions"] = {"row_gap_between_copies_min": min(gaps), "median": float(np.median(gaps)),
                                  "max": max(gaps), "examples": dict(list(pos.items())[:10])}
    ident = sum(1 for v in by.values() if all(r == v[0] for r in v))
    differing_cols = collections.Counter(c for v in by.values() for c in v[0] if any(r[c] != v[0][c] for r in v))
    res["duplicate_detail"] = {"ids_examined": len(by), "ids_whose_copies_are_identical_in_all_13_columns": ident,
                               "columns_that_differ_between_copies": dict(differing_cols),
                               "examples": {k: [{kk: (vv[:80] if isinstance(vv, str) else vv) for kk, vv in r.items()}
                                                for r in v] for k, v in list(by.items())[:4]}}

# ---- timestamps vs post-id order (ids are base-36 counters assigned in creation order)
idnum = np.array([int(x, 36) if x else -1 for x in pid.to_pylist()], dtype=np.int64)
order = np.argsort(idnum, kind="stable")
ts_by_id = par_ts_all[order]
run_max = np.maximum.accumulate(ts_by_id)
res["timestamp_vs_id_order"] = {
    "rows_whose_created_utc_is_>1_day_earlier_than_some_lower_id": int((run_max - ts_by_id > 86400).sum()),
    "rows_>30_days": int((run_max - ts_by_id > 30 * 86400).sum()),
}
del idnum, order, ts_by_id, run_max

tot_kind = collections.Counter()
for y in kind_year:
    tot_kind.update(kind_year[y])
res["img_url_content"] = {
    "kinds_total": dict(tot_kind.most_common()),
    "kinds_by_year": {int(y): dict(c) for y, c in sorted(kind_year.items())},
    "top_hosts": dict(host_tot.most_common(20)),
    "urls_containing_&amp;": {"total": sum(amp.values()), "top_hosts": dict(amp.most_common(8))},
    "urls_with_leading_or_trailing_whitespace": ws,
    "null_url": dict(null_url),
}
res["text_extra"] = {**dict(text_extra), "title_max_len_chars": title_maxlen,
                     "text_max_len_chars_parquet": text_maxlen_p, "text_max_len_chars_raw": text_maxlen_r,
                     "titles_with_non_ascii": nonascii_cells}
res["retrieval_lag_days_median_by_year(1-in-37 subsample)"] = {y: round(float(np.median(v)), 1)
                                                               for y, v in sorted(lag.items())}
json.dump(res, open(f"{OUT}/compare_results.json", "w"), indent=1, default=str)
print(json.dumps(res["monthly"], default=str))
print("done")
