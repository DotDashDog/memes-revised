"""Verify every npz under the given directories loads completely (crash/truncation check)."""
import sys, glob, numpy as np
bad = 0
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/**/*.npz", recursive=True)):
        try:
            z = np.load(f); [z[k] for k in z.files]
        except Exception as e:
            bad += 1; print("BAD", f, e)
print("bad files:", bad)
