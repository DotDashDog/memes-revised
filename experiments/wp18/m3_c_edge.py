"""WP-18 m3: the inner CV chose C=0.001, the smallest value of the pre-set grid, in every fold.
Check (not pre-registered, lower standing): grouped 5-fold accuracy of p_img_ocr at fixed C in
{1e-4, 3e-4, 1e-3}, same outer folds and groups as m3_probe.py (seed 18016).

    uv run python -m experiments.wp18.m3_c_edge
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.wp18.common import OUT, SEED, SUBS
from experiments.wp18.m3_probe import make_groups

warnings.filterwarnings("ignore")
z = np.load(OUT / "m3_embeddings.npz")
s = pd.read_parquet(OUT / "m3_sample.parquet").set_index("post_id").loc[z["post_id"]].reset_index()
y = s.community.map({c: i for i, c in enumerate(SUBS)}).to_numpy()
g = make_groups(s)
X = z["p_img_ocr"].astype(float)
out = {}
for C in [1e-4, 3e-4, 1e-3]:
    pred = np.full(len(y), -1)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g):
        pred[te] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300)).fit(X[tr], y[tr]).predict(X[te])
    out[str(C)] = float((pred == y).mean())
    print(C, out[str(C)], flush=True)
(OUT / "m3_c_edge.json").write_text(json.dumps(out, indent=1))
