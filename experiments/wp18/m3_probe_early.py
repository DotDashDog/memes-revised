"""WP-18 measurement 3: early, partial run of m3_probe.py on the inputs that do not need OCR
(image-only embedding, file-metadata and timestamp baselines), so that a result exists even if
the long OCR step is interrupted. Same functions, folds, groups and seed as m3_probe.py; the full
run supersedes this file.

    uv run python -m experiments.wp18.m3_probe_early
"""
import json
import warnings

import numpy as np
import pandas as pd

from experiments.wp18.common import OUT, SEED, SUBS
from experiments.wp18.m3_probe import boot, make_groups, oof

warnings.filterwarnings("ignore")
z = np.load(OUT / "m3_img_vec.npz")
s = pd.read_parquet(OUT / "m3_sample.parquet").set_index("post_id").loc[z["post_id"]].reset_index()
y = s.community.map({c: i for i, c in enumerate(SUBS)}).to_numpy()
groups = make_groups(s)
rng = np.random.default_rng(SEED)
ts = pd.to_datetime(s.created_utc, unit="s", utc=True)
hour = ts.dt.hour + ts.dt.minute / 60
feats = {
    "p_vlm_only (image only)": z["img"],
    "file metadata only (size, shape, format)": np.column_stack([
        np.log(s.width), np.log(s.height), np.log(s.width / s.height), np.log(s.n_bytes),
        (s.n_frames > 1).astype(float), pd.get_dummies(s.ext).to_numpy(dtype=float)]),
    "timestamp only (day, hour, weekday)": np.column_stack([
        (s.created_utc - s.created_utc.min()) / 86400.0,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(4 * np.pi * hour / 24), np.cos(4 * np.pi * hour / 24),
        pd.get_dummies(ts.dt.weekday).to_numpy(dtype=float)]),
}
out = {"n": len(y), "n_groups": int(len(np.unique(groups))), "seed": SEED, "results": {}}
for k, X in feats.items():
    pred, fa, Cs = oof(np.asarray(X, float), y, groups)
    acc = float((pred == y).mean())
    out["results"][k] = {"accuracy": acc, "ci95": boot(y, pred, groups, rng), "fold_accuracies": fa, "chosen_C": Cs}
    print(k, round(acc, 4), out["results"][k]["ci95"], flush=True)
(OUT / "m3_probe_early.json").write_text(json.dumps(out, indent=1))
