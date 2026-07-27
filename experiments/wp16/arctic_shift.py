"""Arctic Shift client — the independent cross-check source for WP-16.

Arctic Shift (`arctic-shift.photon-reddit.com`) documents that "All data is retrieved
through the official reddit API", standardized from 2023-04 onward. Samples on or after
that date are genuinely independent of the Academic Torrents / Pushshift dumps; earlier
ones share lineage and are NOT independent evidence.

Two hazards, both hit during WP-16 and both fixed here rather than left as footguns:

1. **Never estimate a daily rate from sampled hours.** An earlier pass sampled four
   one-hour windows and scaled x6; it over-estimated by ~34% because 12:00 and 18:00 UTC
   are peak posting hours. `day_scan` walks the entire day.
2. **`max_pages` silently truncates.** At the old default of 30 (= 3,000 posts) the 2020
   counts capped out and looked like ~3,000/day against a true 12,158. The default here is
   300 and `day_scan` returns a `capped` flag — check it, do not ignore it.

The API rejects `is_self`, `is_video`, `domain` and `is_gallery`; `FIELDS` is the usable set.
"""
import datetime as dt
import json
import time
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"
FIELDS = "created_utc,id,url,post_hint"


def get(params, tries=4):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r).get("data") or []
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(2 * (attempt + 1))


def kind(p):
    """Classify a post from `post_hint` + `url` only — the other type fields are rejected."""
    h = p.get("post_hint") or ""
    u = (p.get("url") or "").lower()
    if "video" in h or "v.redd.it" in u or "youtube.com" in u or "youtu.be" in u:
        return "video"
    if "gallery" in u:
        return "gallery"
    if h == "image":
        return "image"
    if any(u.endswith(e) for e in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return "image"
    if "i.redd.it" in u or "i.imgur" in u:
        return "image"
    if "/comments/" in u:
        return "text"          # self-post: url points back at itself
    return "other"


def day_scan(sub, day, max_pages=300):
    """Exact post count for one UTC day. Returns (n_posts, kind_counts, capped)."""
    start = dt.datetime.strptime(day, "%Y-%m-%d")
    end = start + dt.timedelta(days=1)
    seen, kinds, cur, pages = {}, {}, end, 0
    while pages < max_pages:
        rows = get({"subreddit": sub,
                    "after": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "before": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 100, "fields": FIELDS, "sort": "desc"})
        pages += 1
        if rows is None:
            return None, None, True
        if not rows:
            break
        new = [r for r in rows if r["id"] not in seen]
        for r in rows:
            if r["id"] not in seen:
                seen[r["id"]] = 1
                k = kind(r)
                kinds[k] = kinds.get(k, 0) + 1
        if len(rows) < 100 or not new:
            break
        cur = dt.datetime.fromtimestamp(
            min(r["created_utc"] for r in rows), dt.UTC
        ).replace(tzinfo=None) + dt.timedelta(seconds=1)
        time.sleep(0.35)
    return len(seen), kinds, pages >= max_pages
