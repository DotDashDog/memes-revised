"""List E1 cells whose resolved posterior (v3 > v2-rerun > original) has >10% replicates with ESS<200."""
import numpy as np, glob, os
B = "experiments/WP-02/"
for f in sorted(glob.glob(B + "e1-window-counts/cell-[0-9][0-9][0-9].npz")):
    c = os.path.basename(f)
    src = next(p for p in (B + "e1-rerun-v3/" + c, B + "e1-rerun-v2/" + c, f) if os.path.exists(p))
    e = np.load(src)["ess"]
    if np.mean(e < 200) > 0.10:
        print(c, src.split('/')[2], "frac<200 %.2f med %.0f" % (np.mean(e < 200), np.median(e)))
