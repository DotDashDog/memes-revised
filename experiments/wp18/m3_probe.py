"""WP-18 measurement 3, step 2: can a linear probe name the community from the embedding?

Design and primary metric are fixed in experiments/WP-18/expectations.md:
out-of-fold top-1 accuracy, 7 balanced classes, `p_img_ocr`, 5-fold cross-validation with
author / near-duplicate groups kept on one side of every split.

    uv run python -m experiments.wp18.m3_probe
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.wp18.common import BACKBONE, OUT, SEED, SUBS

CS = [0.001, 0.01, 0.1, 1.0]
N_BOOT = 2000
DUP_RADIUS = 4


def make_groups(s: pd.DataFrame) -> np.ndarray:
    """Connected components of (same named author) OR (dHash within DUP_RADIUS)."""
    n = len(s)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    named = s.author.notna() & ~s.author.isin(["[deleted]", "[removed]"])
    sn = s[named]
    pos = sn.index.to_numpy()  # s has a 0..n-1 index
    for _, ix in sn.groupby("author").indices.items():
        for j in ix[1:]:
            union(int(pos[ix[0]]), int(pos[j]))
    h = s.dhash.to_numpy(dtype=np.uint64)
    for st in range(0, n, 1024):
        d = np.bitwise_count(h[st:st + 1024, None] ^ h[None, :])
        ii, jj = np.nonzero(d <= DUP_RADIUS)
        for i, j in zip(ii + st, jj):
            if i < j:
                union(int(i), int(j))
    return np.array([find(i) for i in range(n)])


def fit_predict(Xtr, ytr, gtr, Xte, seed):
    """Choose C by inner grouped CV on the training fold only, refit, predict."""
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = np.zeros(len(CS))
    for tr, va in inner.split(Xtr, ytr, gtr):
        for k, C in enumerate(CS):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300))
            m.fit(Xtr[tr], ytr[tr])
            scores[k] += balanced_accuracy_score(ytr[va], m.predict(Xtr[va]))
    C = CS[int(np.argmax(scores))]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300))
    m.fit(Xtr, ytr)
    return m.predict(Xte), C


def oof(X, y, groups, seed=SEED):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.full(len(y), -1)
    fold_acc, Cs = [], []
    for tr, te in outer.split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])
        pred[te], C = fit_predict(X[tr], y[tr], groups[tr], X[te], seed)
        fold_acc.append(float(balanced_accuracy_score(y[te], pred[te])))
        Cs.append(C)
    assert (pred >= 0).all()
    return pred, fold_acc, Cs


def boot(y, pred, groups, rng, fn=None):
    """Bootstrap over groups of the out-of-fold predictions."""
    fn = fn or (lambda a, b: float((a == b).mean()))
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == k)[0] for k in range(len(ug))]
    stats = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(ug), len(ug))
        ix = np.concatenate([members[k] for k in pick])
        stats.append(fn(y[ix], pred[ix]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    z = np.load(OUT / "m3_embeddings.npz", allow_pickle=False)
    sample = pd.read_parquet(OUT / "m3_sample.parquet").set_index("post_id")
    s = sample.loc[z["post_id"]].reset_index()
    y = s.community.map({c: i for i, c in enumerate(SUBS)}).to_numpy()
    groups = make_groups(s)
    rng = np.random.default_rng(SEED)
    counts = np.bincount(y, minlength=len(SUBS))
    chance = 1 / len(SUBS)
    majority = float(counts.max() / counts.sum())

    ts = pd.to_datetime(s.created_utc, unit="s", utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60
    X_time = np.column_stack([
        (s.created_utc - s.created_utc.min()) / 86400.0,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(4 * np.pi * hour / 24), np.cos(4 * np.pi * hour / 24),
        pd.get_dummies(ts.dt.weekday).to_numpy(dtype=float)])
    X_meta = np.column_stack([
        np.log(s.width), np.log(s.height), np.log(s.width / s.height), np.log(s.n_bytes),
        (s.n_frames > 1).astype(float), pd.get_dummies(s.ext).to_numpy(dtype=float)])
    feats = {
        "p_img_ocr (image + extracted text)": z["p_img_ocr"],
        "p_vlm_only (image only)": z["p_vlm_only"],
        "extracted-text vector only": z["ocr_text_vec"],
        "file metadata only (size, shape, format)": X_meta,
        "timestamp only (day, hour, weekday)": X_time,
    }
    res = {"n": int(len(y)), "per_class": counts.tolist(), "classes": SUBS, "seed": SEED,
           "n_groups": int(len(np.unique(groups))),
           "largest_group": int(np.bincount(np.unique(groups, return_inverse=True)[1]).max()),
           "chance": chance, "majority_class": majority, "C_grid": CS, "results": {}}
    preds = {}
    for name, X in feats.items():
        X = np.asarray(X, dtype=np.float64)
        pred, fold_acc, Cs = oof(X, y, groups)
        preds[name] = pred
        acc = float((pred == y).mean())
        res["results"][name] = {
            "accuracy": acc, "ci95": boot(y, pred, groups, rng), "fold_accuracies": fold_acc,
            "chosen_C": Cs, "gap_closed": (acc - chance) / (1 - chance),
            "per_class_recall": {c: float((pred[y == i] == i).mean()) for i, c in enumerate(SUBS)},
            "confusion_rows_true_cols_pred": confusion_matrix(y, pred).tolist(),
        }
        print(name, round(acc, 4), res["results"][name]["ci95"], [round(a, 3) for a in fold_acc], Cs,
              flush=True)

    # label-permutation control: same pipeline, labels shuffled -> must sit at chance
    yperm = np.random.default_rng(SEED + 1).permutation(y)
    pred, fold_acc, _ = oof(np.asarray(z["p_img_ocr"], dtype=np.float64), yperm, groups)
    res["results"]["label-permutation control (p_img_ocr)"] = {
        "accuracy": float((pred == yperm).mean()), "ci95": boot(yperm, pred, groups, rng),
        "fold_accuracies": fold_acc}
    print("permutation", res["results"]["label-permutation control (p_img_ocr)"], flush=True)

    # sensitivity: random (ungrouped) folds, to show what grouping removes
    from sklearn.model_selection import StratifiedKFold
    Xp = np.asarray(z["p_img_ocr"], dtype=np.float64)
    pr = np.full(len(y), -1)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xp, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300)).fit(Xp[tr], y[tr])
        pr[te] = m.predict(Xp[te])
    res["sensitivity_ungrouped_folds_C0.01_accuracy"] = float((pr == y).mean())

    # secondary probes (reported whatever they show)
    is_ext = (~np.isin(y, [SUBS.index(c) for c in BACKBONE])).astype(int)
    bacc = lambda a, b: float(balanced_accuracy_score(a, b))  # noqa: E731
    for name in list(feats)[:2]:
        X = np.asarray(feats[name], dtype=np.float64)
        pred, fa, _ = oof(X, is_ext, groups)
        res["results"][f"binary backbone-vs-extension, {name}"] = {
            "balanced_accuracy": bacc(is_ext, pred), "ci95": boot(is_ext, pred, groups, rng, bacc),
            "fold_balanced_accuracies": fa, "chance": 0.5,
            "from_7way_collapsed_balanced_accuracy":
                bacc(is_ext, (~np.isin(preds[name], [SUBS.index(c) for c in BACKBONE])).astype(int))}
        pair = np.isin(y, [SUBS.index("memes"), SUBS.index("dankmemes")])
        yp = (y[pair] == SUBS.index("dankmemes")).astype(int)
        pred, fa, _ = oof(X[pair], yp, groups[pair])
        res["results"][f"r/memes vs r/dankmemes, {name}"] = {
            "accuracy": float((pred == yp).mean()), "ci95": boot(yp, pred, groups[pair], rng),
            "fold_accuracies": fa, "chance": 0.5}
    (OUT / "m3_summary.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame({"post_id": s.post_id, "community": s.community, "group": groups,
                  **{f"pred::{k}": np.array(SUBS)[v] for k, v in preds.items()}}) \
        .to_parquet(OUT / "m3_oof_predictions.parquet", index=False)
    print(json.dumps({k: v for k, v in res["results"].items() if "binary" in k or " vs " in k}, indent=1))


if __name__ == "__main__":
    main()
