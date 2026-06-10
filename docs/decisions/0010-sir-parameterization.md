# 0010 — SIR parameterization for prediction

**Status:** accepted  
**Date:** 2026-05-19
**Gates:** [WP-02](../work-packages/WP-02-sir-identifiability-study.md)

## Context

ADR 0005 fixes the SIR initial conditions and gives a fit parameter set of $(\beta, \gamma, s, I_0)$. However, these may not all be jointly identifiable from observed $I(t)$ curves. Predicting non-identifiable parameters is meaningless: the "errors" we'd report would be along directions of the posterior that the data simply doesn't constrain.

WP-02 tested synthetic binned Poisson SIR curves with JAX/diffrax MAP fitting, Hessian/Laplace local geometry, and selected NumPyro/NUTS posterior checks. The main result was:

- SIR curve fitting works as empirical curve compression. MAP fits and NUTS posterior-predictive curves can match the observed time series well.
- Individual natural parameters are not always recoverable. Low-count, boundary, non-positive-Hessian, and near-threshold cases can produce broad or ridge-like posteriors.
- Near-threshold cases are especially important. In `case-052`, the truth was $(\beta,\gamma,s,I_0)=(1,1,7500,0.003)$, but the posterior could move to very different natural parameters while preserving a good curve fit. The data mostly identified net growth/decay and initial amplitude, not a unique $(\beta,\gamma,s,I_0)$.

The right parameterization for *prediction* therefore differs from the parameterization for *fitting*. We want targets that are identifiable enough to be meaningful labels, useful for curve reconstruction, and scientifically interpretable.

## Decision (proposed)

Fit SIR curves in the natural parameterization $(\beta,\gamma,s,I_0)$, but do **not** treat the raw natural parameters as uniformly reliable prediction labels.

For downstream prediction, use a two-tier target set:

1. **Primary prediction targets:** curve-level summaries that are directly observable from the fitted curve and more stable under parameter degeneracy:
  - total expected count / curve mass,
  - peak time,
  - peak height,
  - early net growth or decay rate,
  - optional duration/width summary once the real engagement signal is defined.
2. **Secondary diagnostic targets:** fitted natural SIR parameters $(\beta,\gamma,s,I_0)$ in transformed coordinates, plus identifiability metadata:
  - $(\log\beta,\log\gamma,\log s,\mathrm{logit}(I_0))$,
  - Hessian eigenvalues / condition number when valid,
  - boundary-hit flags,
  - total observed count,
  - posterior or Laplace uncertainty summaries where available.

The prediction model may learn SIR parameters as auxiliary outputs, but evaluation and headline conclusions should not depend on raw SIR-parameter $R^2$ unless the case passes identifiability checks. The core evaluation target should be curve reconstruction quality and stable curve-summary prediction.

## Alternatives considered

- **Predict $(\beta,\gamma,s,I_0)$ directly for every fitted curve.** Rejected for primary evaluation. WP-02 showed that good curve fits can correspond to non-unique natural parameters, especially near threshold or with low counts.
- **Predict only shape statistics.** Attractive because these are closer to identifiable by construction, but it discards useful SIR-fitting diagnostics and makes it harder to compare with the legacy project's fitted-parameter framing.
- **Chosen: predict stable curve summaries as primary targets and retain SIR parameters as diagnostic/auxiliary labels.** This preserves the usefulness of SIR as empirical compression while avoiding overclaiming parameter identifiability.

## Consequences

- WP-08 should output fitted curves, curve summaries, natural SIR parameters, and identifiability metadata for each meme/cluster.
- Baselines and model evaluations should include posterior-predictive or curve-reconstruction quality, not only parameter $R^2$.
- If reporting $R^2$ on SIR parameters, stratify or filter by identifiability flags and report that filtering explicitly.
- The success metric in ADR 0001 may need an interpretation note: "log-spread-parameters" should mean the accepted stable target set, not blindly all raw natural SIR parameters.
- Training code should support multi-target prediction: primary curve summaries plus optional auxiliary SIR-parameter heads.

## Open sub-questions

- Exact curve-summary definitions should be finalized after ADR 0007 defines the engagement signal and binning.
- Should the model predict uncertainty / posterior distributions for the curve summaries, or point estimates plus calibrated residual intervals?
- What threshold should mark an SIR fit as identifiable enough for natural-parameter evaluation? WP-02 suggests using non-positive Hessians, boundary hits, low total counts, and near-threshold dynamics as warning signs, but the numeric thresholds can be refined during WP-08 on real data.

