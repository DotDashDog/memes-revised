"""WP-18 measurement 2, image-free companion: exact-URL sharing across communities, all history.

A native Reddit crosspost, or a second submission of the same uploaded file, carries the same
i.redd.it / imgur URL. This needs no image fetch, so it covers dead links and every year.
It is a strict LOWER bound on cross-posting (a re-upload gets a new URL).

    uv run python -m experiments.wp18.m2_url_history
"""
import json

import pandas as pd
import pyarrow.parquet as pq

from experiments.wp18.common import OUT, SUBS, parquet_path

frames = []
for sub in SUBS:
    t = pq.read_table(parquet_path(sub), columns=["img_url", "created_utc"]).to_pandas()
    t = t[t.img_url.notna()]
    host = t.img_url.str.extract(r"^https?://([^/]+)", expand=False).str.lower()
    t = t[host.isin(["i.redd.it", "i.imgur.com", "imgur.com"])].copy()
    t["url"] = t.img_url.str.replace(r"^https?://", "", regex=True).str.replace(r"\?.*$", "", regex=True)
    t["year"] = pd.to_datetime(t.created_utc, unit="s", utc=True).dt.year
    t["community"] = sub
    frames.append(t[["url", "year", "community"]])
df = pd.concat(frames, ignore_index=True)
ncomm = df.groupby("url").community.nunique()
df["shared"] = df.url.map(ncomm) >= 2
tab = df.groupby(["year", "community"]).shared.agg(["sum", "size"]).reset_index()
tab["pct"] = 100 * tab["sum"] / tab["size"]
tab.to_csv(OUT / "m2_url_shared_by_year.csv", index=False)
piv = tab[tab.year >= 2018].pivot(index="year", columns="community", values="pct").reindex(columns=SUBS).round(3)
print(piv.to_string())
print(tab[tab.year >= 2018].pivot(index="year", columns="community", values="sum").reindex(columns=SUBS).to_string())
(OUT / "m2_url_shared_summary.json").write_text(json.dumps(
    {"image_host_posts": int(len(df)), "posts_sharing_url_with_another_community": int(df.shared.sum()),
     "pct": float(100 * df.shared.mean()), "pct_by_year_community": piv.to_dict()}, indent=1))
