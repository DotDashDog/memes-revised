"""WP-18 measurement 1, follow-up (NOT pre-registered; added after the primary run showed a
collapse of the deleted-author share in mid-2023 and near-total removal in r/comedyheaven).

Adds per community-month: posts not flagged removed, unique named authors of those posts, and
the size of the step in the named-author count across the May->July 2023 boundary.

    uv run python -m experiments.wp18.m1_extra
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.wp18.common import OUT, SUBS, parquet_path

rows = []
for sub in SUBS:
    t = pq.read_table(parquet_path(sub), columns=["created_utc", "author", "removed_by_category"]).to_pandas()
    t["month"] = pd.to_datetime(t.created_utc, unit="s", utc=True).dt.strftime("%Y-%m")
    t["named"] = t.author.notna() & ~t.author.isin(["[deleted]", "[removed]"])
    t["visible"] = t.removed_by_category.isna()
    g = pd.DataFrame({
        "posts": t.groupby("month").size(),
        "visible_posts": t[t.visible].groupby("month").size(),
        "n_named": t[t.named].groupby("month").author.nunique(),
        "n_named_visible": t[t.named & t.visible].groupby("month").author.nunique(),
        "deleted_author_posts": t[~t.named].groupby("month").size(),
    }).fillna(0).astype(int)
    g.insert(0, "subreddit", sub)
    rows.append(g.reset_index())
m = pd.concat(rows)
m.to_csv(OUT / "m1_extra_monthly.csv", index=False)

out = {}
for sub in SUBS:
    d = m[m.subreddit == sub].set_index("month")
    pre, post = d.loc["2023-03":"2023-05"], d.loc["2023-07":"2023-09"]
    if sub == "196":
        post = d.loc["2023-08":"2023-10"]  # r/196 was closed in July 2023
    out[sub] = {
        "named_authors_per_post_pre_2023-03..05": round(pre.n_named.sum() / pre.posts.sum(), 3),
        "named_authors_per_post_post_2023-07..09": round(post.n_named.sum() / post.posts.sum(), 3),
        "ratio_post_over_pre": round((post.n_named.sum() / post.posts.sum()) / (pre.n_named.sum() / pre.posts.sum()), 3),
        "posts_change_pct": round(100 * (post.posts.sum() / pre.posts.sum() - 1), 1),
        "n_named_change_pct": round(100 * (post.n_named.sum() / pre.n_named.sum() - 1), 1),
        "upper_bound_inflation_pre": round(1 + pre.deleted_author_posts.sum() / pre.n_named.sum(), 3),
        "upper_bound_inflation_post": round(1 + post.deleted_author_posts.sum() / post.n_named.sum(), 3),
        "visible_share_by_year_pct": {y: round(100 * g.visible_posts.sum() / g.posts.sum(), 1)
                                      for y, g in d.groupby(d.index.str[:4]) if int(y) >= 2018},
        "min_n_named_visible_since_2019": [int(d.loc["2019-01":].n_named_visible.min()),
                                           d.loc["2019-01":].n_named_visible.idxmin()],
        "months_n_named_visible_under_100_since_2019":
            d.loc["2019-01":].index[d.loc["2019-01":].n_named_visible < 100].tolist(),
    }
(OUT / "m1_extra_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
