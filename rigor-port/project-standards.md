# Project Standards (project-specific) — [FILL IN: project name]

This file holds the **project-specific** experimental standards. The general "don't fool yourself" method discipline lives in `method-discipline.md`; both are required reading. Fill each section with the standards for this domain, delete any that don't apply, and add domain-specific ones. Keep general method discipline *out* of this file so it stays portable.

---

## 1. Baselines / controls are mandatory

No result is accepted without comparison against baselines/controls. Define the minimum set every experiment must report against:

- [FILL IN: baseline 1 — e.g. a trivial / null / "no-information" model]
- [FILL IN: baseline 2 — the strongest *cheap* competitor your method must beat; if you don't beat it, the method earned nothing]
- [FILL IN: …]

Report all baselines on the same evaluation set, with the same metrics, alongside any claim.

## 2. Validation discipline

- [FILL IN: how is the train / validation / test (or equivalent hold-out) split decided — up front, not retrofitted?]
- Choices and hyperparameters are tuned on validation, **never** on the final test/eval set.
- The final test set is touched **once per real claim**; re-running it to improve numbers is a leak.
- [FILL IN: the *unit* of the split — what must be held out *together* to avoid leakage in this domain?]

## 3. Leakage / contamination [rename to this project's threat]

- [FILL IN: the project's main "the result looks too good because information leaked in" threat, and how you guard against it — e.g. train/test contamination, look-ahead/temporal leakage, double-use of the same data, pretraining overlap, an over-informative prior.]
- Report held-in vs. held-out metrics side by side; a large gap is a warning sign.

## 4. Uncertainty is reported, not hidden

- Every reported metric carries an uncertainty estimate (bootstrap CI, posterior credible interval, or equivalent). Point estimates alone are insufficient.
- [FILL IN: domain conventions — e.g. prefer fitting with a likelihood/NLL so you get calibrated per-result uncertainty for free.]

## 5. Reproducibility

- **Seeds** recorded (all RNGs). **Versions** pinned (code, data snapshot, key dependencies). **Inputs addressable** (commit hash / checksum / dataset-version tag — not "the data at `/some/path` as of today"). **Configs saved next to results.**
- [FILL IN: any domain-specific reproducibility needs — hardware, solver tolerances, pipeline/instrument versions.]

## 6. Negative and null results are recorded

- A failed experiment is a result; file it in the relevant lab-notebook log. A control/baseline that beats your method is the *most important* thing to report, not to hide. If you abandon an approach, record *why* so it isn't repeated.

## 7. Failure modes to actively watch for

List this project's known traps — if any occurs, stop and reassess:

- [FILL IN: failure mode 1 — e.g. "the result is driven entirely by <confounder/nuisance parameter>"]
- [FILL IN: failure mode 2 — e.g. "performance correlates with <something that shouldn't matter>"]
- [FILL IN: …]

## 8. [FILL IN: domain-specific standards]

[FILL IN: standards specific to this field — e.g. physical sanity checks, units/conventions, instrument or noise models, known identifiability quirks, symmetries to respect. General method discipline does not go here; it lives in `method-discipline.md`.]
