"""Agreement between two independent label-resolution passes, per layer.

Reports pair-level agreement (of the pairs one pass calls "same", the share the
other also calls "same") and B-cubed precision/recall of B scored against A.

Usage: uv run python -m experiments.wp04.compare_labellers --sample experiments/WP-04/sample-001
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.wp04.score_audit import per_item_pr


def same_pairs(ids, lab):
    groups: dict = {}
    for i, l in zip(ids, lab):
        groups.setdefault(l, []).append(i)
    return {tuple(sorted(p)) for g in groups.values() if len(g) > 1 for p in combinations(g, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    a = ap.parse_args()
    d = Path(a.sample) / "stage2"
    A = pd.DataFrame(json.loads((d / "labels_A.json").read_text())).set_index("id")
    B = pd.DataFrame(json.loads((d / "labels_B.json").read_text())).set_index("id").loc[A.index]
    out = {}
    for layer in ("template_id", "family_id"):
        pa, pb = same_pairs(A.index, A[layer]), same_pairs(B.index, B[layer])
        both = pa & pb
        prec, rec = per_item_pr(A[layer].to_numpy(), B[layer].to_numpy())
        ga = A[layer].map(A[layer].value_counts()) >= 2
        gb = B[layer].map(B[layer].value_counts()) >= 2
        out[layer] = {
            "same_pairs_A": len(pa), "same_pairs_B": len(pb), "same_pairs_both": len(both),
            "share_of_A_pairs_B_agrees": round(len(both) / len(pa), 3),
            "share_of_B_pairs_A_agrees": round(len(both) / len(pb), 3),
            "pair_jaccard": round(len(both) / len(pa | pb), 3),
            "items_grouped_by_A": int(ga.sum()), "items_grouped_by_B": int(gb.sum()),
            "items_grouped_by_both": int((ga & gb).sum()),
            "items_grouped_by_exactly_one": int((ga ^ gb).sum()),
            "bcubed_B_vs_A_precision_all": round(float(prec.mean()), 3),
            "bcubed_B_vs_A_recall_on_A_multi": round(float(rec[ga.to_numpy()].mean()), 3),
        }
        only_a = sorted(pa - pb)[:0]
    (d / "labeller_agreement.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    # disagreement listing for the family layer, by A-family
    fa, fb = A.family_id, B.family_id
    rows = []
    for fam, idx in fa.groupby(fa).groups.items():
        if len(idx) < 2:
            continue
        rows.append((fam, len(idx), dict(fb.loc[idx].map(lambda x: x if not x.startswith("unique-") else "unique").value_counts())))
    rows.sort(key=lambda r: -r[1])
    for r in rows[:60]:
        print(r)
    print("B families whose members A left all unique:")
    for fam, idx in fb.groupby(fb).groups.items():
        if len(idx) >= 2 and all(x.startswith("unique-") for x in fa.loc[idx]):
            print("  ", fam, len(idx))


if __name__ == "__main__":
    main()
