# 0005 — SIR initial conditions

**Status:** accepted
**Date:** 2026-05-19
**Gates:** —

## Context

The SIR ODEs have four scalar quantities one might fit per curve: $\beta$, $\gamma$, $S(0)$, $I(0)$, $R(0)$ (with the population-fraction constraint $S+I+R = 1$). Treating all of them as free parameters introduces strong degeneracy — in particular between $R(0)$, $\beta$, and the overall amplitude/timing of the curve.

## Decision

Fix the initial conditions as:

- $S(0) = 1 - I(0)$
- $R(0) = 0$

Fit only $(\beta, \gamma, s, I(0))$, where $s$ is the scaling from population fraction to observed signal. Final prediction targets may be a reparameterization (see ADR 0010).

## Alternatives considered

- **Free $R(0)$.** Adds a parameter that's almost certainly degenerate with $\beta$ and $s$. Rejected pending WP-02.
- **Fix $I(0)$ to a small constant.** Loses information about onset; some memes have visible ramp-up phases.
- **Fix $\gamma$ from external priors** (e.g., a mean recovery rate from prior data). Premature — could be revisited if WP-02 shows $\gamma$ is barely identifiable.

## Consequences

- The fit parameter set is $(\beta, \gamma, s, I(0))$; identifiability of these is checked in WP-02.
- Memes with substantial "pre-history" (already-spread before the observation window) may fit poorly; treat as a known failure mode.

## Open sub-questions

- Should $I(0)$ be predicted by the model, or treated as a per-curve nuisance parameter fitted but not predicted? Likely the latter, but tied to ADR 0010.
