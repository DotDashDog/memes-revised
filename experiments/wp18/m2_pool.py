"""WP-18 measurement 2, step 1: the fetch pool — every submission of the seven candidate
communities in the pre-registered 28-day window, with its URL classified by the project's
fetcher (`experiments.wp04.fetch.normalize_url`).

    uv run python -m experiments.wp18.m2_pool
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pandas as pd
import pyarrow.dataset as ds

from experiments.wp04.fetch import normalize_url
from experiments.wp18.common import OUT, SUBS, parquet_path

T0 = int(datetime(2025, 11, 3, tzinfo=timezone.utc).timestamp())
T1 = int(datetime(2025, 12, 1, tzinfo=timezone.utc).timestamp())  # exclusive


def url_class(url) -> tuple[str, str | None]:
    if not isinstance(url, str) or not url.strip():
        return "no_url", None
    fetch_url, lane = normalize_url(url)
    if fetch_url is not None:
        return "image", fetch_url
    u = url.lower()
    if "/gallery/" in u:
        return "gallery", None
    if "v.redd.it" in u or "youtu" in u:
        return "video", None
    if re.search(r"reddit\.com/r/[^/]+/comments/", u):
        return "self_or_permalink", None
    return "other_link", None


def main() -> None:
    frames = []
    for sub in SUBS:
        d = ds.dataset(parquet_path(sub)).to_table(
            filter=(ds.field("created_utc") >= T0) & (ds.field("created_utc") < T1),
            columns=["post_id", "subreddit", "created_utc", "author", "img_url", "title",
                     "score", "num_comments", "removed_by_category", "over_18"]).to_pandas()
        d["community"] = sub
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    cls = df.img_url.map(url_class)
    df["url_class"] = [c for c, _ in cls]
    df["fetch_url"] = [f for _, f in cls]
    df["host"] = df.img_url.str.extract(r"https?://([^/]+)", expand=False).str.lower().fillna("")
    df["day"] = pd.to_datetime(df.created_utc, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    assert df.post_id.is_unique
    df.to_parquet(OUT / "m2_window_posts.parquet", index=False)
    pool = df[df.url_class == "image"][["post_id", "img_url", "host", "community"]]
    pool.to_parquet(OUT / "m2_pool.parquet", index=False)
    tab = df.groupby(["community", "url_class"]).size().unstack(fill_value=0).reindex(SUBS)
    tab["total"] = tab.sum(axis=1)
    print(tab.to_string())
    print("days:", df.day.nunique(), " pool rows:", len(pool), " unique fetch urls:",
          df.fetch_url.dropna().nunique())
    (OUT / "m2_pool_meta.json").write_text(json.dumps({
        "window_utc": ["2025-11-03T00:00:00Z", "2025-12-01T00:00:00Z (exclusive)"],
        "t0": T0, "t1": T1, "counts": tab.to_dict(orient="index"),
        "removed_by_category_in_image_posts":
            df[df.url_class == "image"].groupby("community").removed_by_category
            .apply(lambda s: s.fillna("<none>").value_counts().to_dict()).unstack(fill_value=0)
            .to_dict(orient="index"),
    }, indent=1, default=int))


if __name__ == "__main__":
    main()
