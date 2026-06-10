# 0001 — Target success metric

**Status:** accepted
**Date:** 2026-05-19
**Gates:** —

## Context

We need a concrete bar for "did this project work." Without one, we can't tell whether disappointing results are a model problem, a data problem, or a fundamental-limit problem. The previous iteration reported "~250% error in log spread parameters vs. ~350% baseline," which is hard to interpret and not a standard metric.

## Decision

Primary metric: **$R^2 \geq 0.3$ on log-spread-parameters on a held-out evaluation set composed entirely of memes that originated after the embedding model's training cutoff.**

Reported alongside the primary metric:

- $R^2$ separately for each predicted parameter (and, when applicable, identifiable combinations like $R_0$).
- **Spearman rank correlation** (more robust to distributional skew).
- Bootstrap 95% CI over evaluation memes for both metrics.
- The same metrics for each baseline in `01-scientific-rigor.md` §1.

## Alternatives considered

- **Mean absolute error on log-parameters.** Pros: interpretable absolute scale. Cons: depends on the distribution of the target; harder to compare across reparameterizations.
- **Median percentage error in linear parameter space.** This is roughly what the prior iteration used. Cons: unstable, hard to interpret, sensitive to outliers.
- **Held-out NLL on observed $I(t)$ curves** (predict params → integrate → compare to data). Pros: most directly measures what we care about (predicting curves). Cons: couples the prediction model with the SIR-fit pipeline; harder to compare against simpler baselines.

## Consequences

- All experiments report $R^2$ in log space and Spearman rank correlation.
- The evaluation set must be constructed to support post-cutoff filtering (see ADR 0004 and ADR 0011).
- The 0.3 bar is the *project*-level success criterion. Individual experiments do not need to hit it; they only need to make progress toward it or rule out approaches.

## Open sub-questions

- Should we adopt held-out NLL on $I(t)$ as a secondary metric? It would make end-to-end differentiable training more natural to evaluate.
- How to define $R^2$ when the prediction is a distribution rather than a point estimate? (Likely: use the predictive mean for $R^2$, report calibration separately.)
