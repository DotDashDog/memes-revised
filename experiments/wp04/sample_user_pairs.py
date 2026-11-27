"""Draw the pairs the user judges, per the labelling protocol's validation section.

Strata (labeller A's labels; seed fixed):
  same      40  pairs A put in one family; a family is drawn uniformly, then a pair,
                so large families do not swamp the stratum
  hard_tpl  25  same template, different family
  hard_ref  15  share a stage-1 referent string, different family and different template
  random    20  uniform random pairs
Order is shuffled; the page shows no labels. Writes pairs.json (with strata, kept
out of the page) and thumbnails under <sample>/user_check/img/.
"""
from __future__ import annotations

import json
import random
import re
from itertools import combinations
from pathlib import Path

import pandas as pd
from PIL import Image

S = Path("/home/linus/memes-revised/experiments/WP-04/sample-001")
SEED = 20261127


def main() -> None:
    rng = random.Random(SEED)
    lab = pd.read_parquet(S / "labels.parquet").set_index("post_id")
    desc = {d["id"]: d for d in json.loads((S / "stage1/descriptions.json").read_text())}
    fam, tpl = lab.family_id.to_dict(), lab.template_id.to_dict()
    used: set = set()
    out = []

    def add(a, b, stratum):
        key = tuple(sorted((a, b)))
        if key in used:
            return False
        used.add(key)
        pair = list(key)
        rng.shuffle(pair)
        out.append({"a": pair[0], "b": pair[1], "stratum": stratum})
        return True

    fams = [g.index.tolist() for _, g in lab.groupby("family_id") if len(g) >= 2]
    n = 0
    while n < 40:
        g = rng.choice(fams)
        n += add(*rng.sample(g, 2), "same")

    tp = [p for _, g in lab.groupby("template_id") if len(g) >= 2
          for p in combinations(g.index, 2) if fam[p[0]] != fam[p[1]]]
    for p in rng.sample(tp, 25):
        add(*p, "hard_tpl")

    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    by_ref: dict = {}
    for i, d in desc.items():
        for r in d["referents"]:
            by_ref.setdefault(norm(r), []).append(i)
    rp = sorted({tuple(sorted(p)) for ids in by_ref.values() if len(ids) >= 2
                 for p in combinations(ids, 2)
                 if fam[p[0]] != fam[p[1]] and tpl[p[0]] != tpl[p[1]]})
    for p in rng.sample(rp, 15):
        add(*p, "hard_ref")

    ids = lab.index.tolist()
    n = 0
    while n < 20:
        n += add(*rng.sample(ids, 2), "random")

    rng.shuffle(out)
    for k, p in enumerate(out):
        p["pair_id"] = f"p{k:03d}"
    d = S / "user_check"
    (d / "img").mkdir(parents=True, exist_ok=True)
    (d / "pairs.json").write_text(json.dumps(out, indent=1))
    for i in {x for p in out for x in (p["a"], p["b"])}:
        im = Image.open(S / "label_images" / f"{i}.jpg")
        im.thumbnail((560, 560))
        im.save(d / "img" / f"{i}.jpg", quality=82)
    print(len(out), "pairs;", pd.Series([p["stratum"] for p in out]).value_counts().to_dict(),
          "; candidate pools:", len(tp), len(rp))


if __name__ == "__main__":
    main()
