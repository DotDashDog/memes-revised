"""WP-18 measurement 2, match audit (third agent; NOT pre-registered, decided after reading the
previous agent's 30-pair contact-sheet judgement, before looking at any new pair).

1. Regenerates m2_contact_sheet.png (0 bytes after the 18:17 crash) with the same seed, so the
   earlier by-eye judgement can be re-checked.
2. Post-level audit of the primary metric. The primary metric counts a post as "cross-posted" if
   its nearest image in another community is within Hamming 4. Posts at distance 0 were all judged
   the same meme (17/17); the doubt is the posts whose nearest cross-community image is at
   distance 1-4. Draw 40 such posts at random (seed 18016), show each next to its nearest
   cross-community image, judge by eye, and propagate the share judged "same meme" into a
   corrected estimate.

    uv run python -m experiments.wp18.m2_match_audit
"""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from experiments.wp18.common import OUT, SEED, SUBS
from experiments.wp18.m2_plot import ROOT, contact_sheet

N_AUDIT = 40


def main() -> None:
    contact_sheet()
    u = pd.read_parquet(OUT / "m2_usable_posts.parquet")
    mind = np.load(OUT / "m2_min_dist.npy")
    lab = u.community.map({s: i for i, s in enumerate(SUBS)}).to_numpy()
    other = mind.copy().astype(int)
    other[np.arange(len(u)), lab] = 999
    cross_min = other.min(axis=1)
    cand = np.flatnonzero((cross_min >= 1) & (cross_min <= 4))
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(cand, size=N_AUDIT, replace=False))
    h = u.dhash.to_numpy().astype(np.uint64)
    rows = []
    for i in pick:
        x = np.bitwise_xor(h, h[i])
        d = np.array([bin(int(v)).count("1") for v in x])
        d[lab == lab[i]] = 999
        j = int(np.argmin(d))
        rows.append(dict(i=int(i), j=j, dist=int(d[j]), post_i=u.post_id[i], post_j=u.post_id[j],
                         comm_i=u.community[i], comm_j=u.community[j],
                         same_author=bool(u.author[i] == u.author[j] and u.author[i] not in ("[deleted]", None))))
    a = pd.DataFrame(rows)
    assert (a.dist == cross_min[pick]).all()
    a.to_csv(OUT / "m2_match_audit_pairs.csv", index=False)
    idx = pd.read_parquet(OUT / "m2_shard_index.parquet").set_index("post_id")

    def load(pid):
        r = idx.loc[pid]
        with open(ROOT / "images" / f"{r.shard_id}.tar", "rb") as f:
            f.seek(int(r.offset))
            im = Image.open(io.BytesIO(f.read(int(r.length))))
            im.seek(0)
            im = im.convert("RGB")
            im.thumbnail((420, 420))
            return im

    for sheet in range(2):
        part = a.iloc[sheet * 20:(sheet + 1) * 20]
        fig, axes = plt.subplots(4, 10, figsize=(30, 15))
        for ax in axes.ravel():
            ax.axis("off")
        for k, (_, r) in enumerate(part.iterrows()):
            for side, (pid, comm) in enumerate([(r.post_i, r.comm_i), (r.post_j, r.comm_j)]):
                ax = axes[k // 5, (k % 5) * 2 + side]
                ax.imshow(load(pid))
                ax.set_title(f"#{sheet * 20 + k + 1}{'ab'[side]} r/{comm}" + (f" d={r.dist}" if side else ""),
                             fontsize=11)
        fig.suptitle(f"Match audit sheet {sheet + 1}: posts whose nearest other-community image is at "
                     f"Hamming 1-4 (random, seed {SEED})", fontsize=14, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUT / f"m2_match_audit_sheet{sheet + 1}.png", dpi=80)
        plt.close(fig)
    print("candidates d1-4:", len(cand), "of", len(u), "; d0:", int((cross_min == 0).sum()))


if __name__ == "__main__":
    main()
