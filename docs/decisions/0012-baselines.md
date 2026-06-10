# 0012 — Canonical baseline set for the first model run

**Status:** proposed
**Date:** 2026-05-19
**Gates:** embeddings + targets exist

## Context

`01-scientific-rigor.md` §1 mandates baselines for every model claim. This ADR locks the specific implementations and hyperparameters for the first round of modeling, so they're comparable across experiments and can't drift.

## Decision (proposed)

The canonical baseline set, applied to the same train/eval split as the model:

- **B1 — Global mean.** Predict the training-set mean of each target parameter for every test example.
- **B2 — Date regressor.** A small regression (ridge / GBM) using only the meme's origin date (numeric) as input.
- **B3 — Cluster-size regressor.** A small regression using only $\log(\text{cluster size})$ as input.
- **B4 — k-NN in embedding space.** $k=10$ default (run $k \in \{1,5,10,25\}$ as sensitivity). Predicts the mean target of the *k* nearest train-set neighbors in cosine distance.
- **B5 — Zero-shot LLM** (optional / when feasible). A frontier LLM given a textual cluster summary, with strict filtering so only pre-cutoff cluster examples are used to demonstrate the technique on training; eval examples are all post-cutoff, by construction filtered before being shown to the LLM.

All baselines report:

- $R^2$ on log-parameters.
- Spearman rank correlation.
- Bootstrap 95% CI on both.

## Alternatives considered

- **Larger k-NN sweep.** Could be done; the small sweep above is sufficient.
- **Random-forest baseline on the embedding.** Could be added later; k-NN already covers the "embedding does the work" claim.

## Consequences

- The baseline scripts are part of the project and live in code; they must be runnable end-to-end before any model claim.
- New experiments that change the train/eval split must re-run baselines on the new split.

## Open sub-questions

- Should the date and cluster-size baselines also be combined? Probably yes as a "B2+B3" multivariate baseline.
- What's the right zero-shot LLM prompt to give it a fair chance? Tracked in a future ADR.
