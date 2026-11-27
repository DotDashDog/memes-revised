# Scientific Rigor Standards (project-specific)

This is a research project, not a feature-shipping project. The goal is **a defensible answer to a scientific question**, not a working model at any cost. If one of the standards below seems wrong for the case at hand, raise it with the user instead of skipping it.

> **This file holds the *project-specific* standards** (baselines, validation split, contamination, failure modes). The *general* "don't fool yourself" method discipline — identifiability as a precondition, matching the test to the claim, principled pre-registration, plots-before-metrics, the anti-cargo-cult debugging rules, surfacing assumptions — lives in [`method-discipline.md`](method-discipline.md).
>
> **Section numbers are preserved for citation stability.** The gaps below (no §4, §8, §9, §11) are intentional: those sections moved to `method-discipline.md`. Do not renumber.

## 1. Baselines are mandatory

No model claim is accepted without comparison against baselines. The minimum baseline set for every prediction experiment:

- **B1 — Global mean.** Predict the mean of training-set targets, regardless of input.
- **B2 — Date-only regressor.** Predict targets from just the meme's origin date (a univariate feature).
- **B3 — Cluster-size only.** Predict targets from just the number of posts in the cluster.
- **B4 — k-NN in embedding space.** Predict the average target of the *k* nearest neighbors in the same embedding space the model uses. **This is the most important baseline.** If your model doesn't beat it meaningfully, the embedding has done all the work.
- **B5 — Zero-shot LLM** (when feasible). A frontier model given the cluster's textual description, with strict pre-cutoff filtering of evaluation memes. This is both a baseline and an upper bound on the contamination signal.

Report all baselines on the same eval set, with the same metrics, alongside any model claim.

## 2. Validation discipline

- **Split is decided up front, not retrofitted.** See [`decisions/0011-train-test-split.md`](decisions/0011-train-test-split.md) when accepted. Default expectation: time-based split with the evaluation set entirely after the embedding model's training cutoff.
- **Hyperparameters are tuned on a validation set, never the test set.**
- **The test set is touched once per real claim.** If you find yourself re-running on test to improve numbers, you have a leak.
- **Held-out memes, not held-out posts.** Splits at the post level leak information; splits should be at the cluster/meme level (and, separately, by time).

## 3. Embedding contamination is treated as a live threat at all times

- The evaluation set should contain only memes that originated after the embedding model's training cutoff date.
- Report training-set vs. evaluation-set metrics side by side. A large gap is a contamination signal even on otherwise clean splits.
- Wherever feasible, run a *contamination probe*: predict spread parameters from the embedding *without showing the model the time-series data*. If accuracy is high, the embedding already encodes virality.

## 5. Uncertainty is reported, not hidden

- Point estimates are insufficient. Every reported metric should include an uncertainty estimate: bootstrap CI over evaluation memes, posterior credible interval, or equivalent.
- For models that predict parameters, prefer training with a likelihood (e.g., Gaussian NLL on log-parameters) so you get calibrated per-prediction uncertainty for free.
- A model that predicts wide intervals for hard cases and tight intervals for easy cases is *more useful* than one that always says the same point estimate.

## 6. Reproducibility

- **Seeds are recorded.** Every experiment logs all random seeds (NumPy, PyTorch/JAX, sampler, splitter).
- **Versions are pinned.** Embedding model weights/version, dataset snapshot date, library versions noted in the work package log.
- **Inputs are addressable.** Don't run an experiment on "the data in `/some/path` as of today." Record an immutable identifier (commit hash, file checksum, or dataset-version tag).
- **Configs live with results.** When you save a model or a result, save its config next to it.

## 7. Negative and null results are recorded

- A failed experiment is a result. File it in the work package's log section.
- A baseline that beats your model is the most important thing to report, not the thing to hide.
- If you abandon an approach, write down *why* in the relevant ADR (mark `superseded`) or work package log so future agents don't repeat it.

## 10. Failure modes to actively watch for

Track these throughout the project — if any of them happen, stop and reassess:

- The model relies almost entirely on cluster size for its predictions. (Check via permutation importance or by training without that signal.)
- Performance is suspiciously good on pre-cutoff memes and bad on post-cutoff. (Contamination.)
- Performance correlates with how "famous" a meme is. (Endogeneity / virality-leak.)
- The model predicts the same parameter values for almost all inputs but achieves moderate $R^2$ by chance. (Check distribution of predictions vs. targets.)
- Improving the embedding model makes results worse. (Almost always a contamination signal — better embeddings memorize more.)
