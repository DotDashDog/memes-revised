"""Plots for the clustering audit (WP-04 Method §7) plus two side checks.

1. precision_recall.png   precision-recall curves, layer x stratum, all methods
2. family_contingency.png largest gold families vs clusters at the P=0.90 setting
3. umap.png               2-D map coloured by gold family (largest) per arm
4. neighbours.json/.png   clustering-free check: is an item's nearest neighbour in
                          the same gold family / template? (exploratory, not registered)
5. hdbscan in precision_recall.png as extra markers (registered as "alongside")

Usage: uv run python -m experiments.wp04.plot_audit --sample experiments/WP-04/sample-001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdbscan
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.cluster.hierarchy import fcluster, linkage

from experiments.wp04.score_audit import ARMS, THRESHOLDS, P_TARGET, per_item_pr, summarise

COL = {"image + text": "#1f77b4", "image only": "#ff7f0e", "perceptual hash": "#2ca02c",
       "every item alone": "#7f7f7f"}
LAYER_NAME = {"family_id": "same meme by meaning", "template_id": "same template (base image)"}


def load(sample: Path):
    lab = pd.read_parquet(sample / "labels.parquet")
    emb_ids = json.loads((sample / "ids.json").read_text())
    pos = {p: i for i, p in enumerate(emb_ids)}
    rows = [pos[i] for i in lab.post_id]
    X = {}
    for name, fn in ARMS.items():
        x = np.load(sample / fn)[rows].astype(np.float64)
        X[name] = x / np.linalg.norm(x, axis=1, keepdims=True)
    return lab, X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    a = ap.parse_args()
    sample = Path(a.sample)
    out = sample / "plots"
    out.mkdir(exist_ok=True)
    lab, X = load(sample)
    sc = json.loads((sample / "scores.json").read_text())
    strata = {"pooled": np.ones(len(lab), bool), "background": (lab.stratum == "background").to_numpy()}

    # HDBSCAN points
    hd = {}
    for name, x in X.items():
        pts = []
        for mcs in (2, 3, 5):
            cl = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(x)
            cl = np.where(cl < 0, -np.arange(1, len(cl) + 1), cl)  # noise = singletons
            pts.append((mcs, cl))
        hd[name] = pts

    # 1. precision-recall
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for r, layer in enumerate(("family_id", "template_id")):
        gold = lab[layer].to_numpy()
        gs = pd.Series(gold).map(pd.Series(gold).value_counts()).to_numpy()
        multi = (gs >= 2).astype(float)
        for c, (sname, mask) in enumerate(strata.items()):
            ax = axes[r, c]
            for m, mr in sc["layers"][layer]["methods"].items():
                v = mr[sname]
                style = dict(marker="o", ms=3, lw=1.5) if m != "every item alone" else dict(marker="*", ms=14, lw=0)
                ax.plot(v["recall_curve"], v["precision_curve"], color=COL[m],
                        label=f"{m}  (kept together at 90% purity: {v['recall_at_p90']:.2f}, "
                              f"95% interval {v['recall_at_p90_ci95'][0]:.2f} to {v['recall_at_p90_ci95'][1]:.2f})",
                        **style)
            for name, pts in hd.items():
                for mcs, cl in pts:
                    pi, ri = per_item_pr(gold, cl)
                    P, R = summarise(pi, ri, multi, mask.astype(float))
                    ax.plot(R, P, marker="x", ms=8, color=COL[name], lw=0)
            ax.axhline(P_TARGET, color="k", ls=":", lw=0.8)
            ax.set_title(f"Reference: {LAYER_NAME[layer]}\n{sname} sample "
                         f"({v['n_items']} images, {v['n_items_in_multi']} in groups of 2+)", fontsize=10)
            ax.legend(fontsize=7, loc="lower left")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
            if r == 1: ax.set_xlabel("Share of each true group kept together (recall, images in groups of 2+)")
            if c == 0: ax.set_ylabel("Purity of clusters (precision, all images)")
    fig.suptitle("Clustering audit. Top (meaning): at 90% purity little of each meme family stays together in any arm; "
                 "at looser purity, adding text helps.\n"
                 "Bottom (templates): image-only is about as good as a perceptual hash; image + text is clearly worse.\n"
                 "(lines: fine-to-coarse sweep; crosses: HDBSCAN; star: every image left alone; up and right is better)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "precision_recall.png", dpi=130)
    plt.close(fig)

    # 2. contingency at the P=0.90 setting (pooled, family layer), image + text
    fam = lab.family_id
    top = fam.value_counts()
    top = top[top >= 4].index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, name in zip(axes, X):
        v = sc["layers"]["family_id"]["methods"][name]["pooled"]
        Ps = np.array(v["precision_curve"])
        k = np.where(Ps >= P_TARGET)[0].max()
        Z = linkage(X[name], method="average", metric="cosine")
        cl = fcluster(Z, t=THRESHOLDS[k], criterion="distance")
        sub = lab[fam.isin(top)].assign(cluster=cl[fam.isin(top).to_numpy()])
        ct = pd.crosstab(sub.family_id, sub.cluster).loc[top]
        ct = ct.loc[:, ct.sum().sort_values(ascending=False).index]
        order = np.argsort([-ct.loc[f].max() / ct.loc[f].sum() for f in top])
        ct = ct.iloc[order]
        ct = ct.loc[:, sorted(ct.columns, key=lambda col: (int(np.argmax(ct[col].to_numpy())), -ct[col].max()))]
        ax.imshow(ct.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=max(4, ct.to_numpy().max()))
        ax.set_yticks(range(len(ct)))
        ax.set_yticklabels([f"{f} ({int(ct.loc[f].sum())})" for f in ct.index], fontsize=8)
        ax.set_xticks([])
        ax.set_xlabel(f"clusters containing these images ({ct.shape[1]} clusters; one column each)")
        frac = np.mean([ct.loc[f].max() / ct.loc[f].sum() for f in ct.index])
        ax.set_title(f"{name}, cosine threshold {THRESHOLDS[k]:.2f}\n"
                     f"largest single cluster holds on average {frac:.0%} of a family", fontsize=10)
    fig.suptitle("Where each true meme family ends up (families of 4+, at the 90%-purity setting).\n"
                 "If clustering matched meaning, each row would be one dark cell; instead most rows are scattered across many clusters",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "family_contingency.png", dpi=130)
    plt.close(fig)

    # 3. UMAP
    big = fam.value_counts().index[:8].tolist()
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, name in zip(axes, X):
        xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(X[name])
        ax.scatter(xy[:, 0], xy[:, 1], s=5, color="#cccccc", label="all other images")
        for i, f in enumerate(big):
            m = (fam == f).to_numpy()
            ax.scatter(xy[m, 0], xy[m, 1], s=28, color=cmap(i), label=f"{f} ({m.sum()})", edgecolor="k", lw=0.3)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("2-D map of the embedding (UMAP), eight largest true meme families coloured.\n"
                 "If the embedding captured meaning, each colour would form one tight clump", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "umap.png", dpi=130)
    plt.close(fig)

    # 4. nearest-neighbour check (exploratory)
    nn = {}
    rng = np.random.default_rng(42)
    for layer in ("family_id", "template_id"):
        gold = lab[layer].to_numpy()
        gs = pd.Series(gold).map(pd.Series(gold).value_counts()).to_numpy()
        multi = gs >= 2
        nn[layer] = {"n_items_in_groups": int(multi.sum())}
        for name, x in X.items():
            S = x @ x.T
            np.fill_diagonal(S, -np.inf)
            top5 = np.argsort(-S, axis=1)[:, :5]
            hit1 = gold[top5[:, 0]] == gold
            hit5 = (gold[top5] == gold[:, None]).any(1)
            # chance: random other item in same group
            nn[layer][name] = {"nearest_neighbour_same_group": float(hit1[multi].mean()),
                               "any_of_5_nearest_same_group": float(hit5[multi].mean())}
        nn[layer]["chance_nearest"] = float(((gs - 1) / (len(gold) - 1))[multi].mean())
    (sample / "neighbours.json").write_text(json.dumps(nn, indent=1))
    print(json.dumps(nn, indent=1))


if __name__ == "__main__":
    main()
