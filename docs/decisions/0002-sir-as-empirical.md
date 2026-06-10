# 0002 — SIR as an empirical compression of curve shape

**Status:** accepted
**Date:** 2026-05-19
**Gates:** —

## Context

The SIR model is *not* literally true of meme spread. Memes can be re-encountered (violating "no reinfection"), populations are not homogeneous, and the relationship between "infected" and "posted" is mediated by behavior. The justification for using SIR has shifted: originally we treated it as a mechanistic model; we now treat it as an empirical compression.

## Decision

Use SIR as a low-dimensional empirical parameterization of meme interest-over-time curves. The model is justified by:

1. Empirically good fit to most observed meme curves in the prior iteration.
2. A small, interpretable parameter set.
3. Memes empirically decay to zero interest, which SIR captures and SIRS/SIS do not (those have nonzero equilibria).

We will not interpret the fitted SIR parameters as mechanistic quantities. They are summary statistics of curve shape.

## Alternatives considered

- **SIRS / SIS** — Allows reinfection but has nonzero steady-state $I$, which contradicts observations.
- **Pure curve-shape parameterization** (e.g., a Gaussian or log-normal bump) — Loses any structural prior. Probably worth carrying as a baseline.
- **Multi-population SIR / sums of SIRs** — Captures multi-hump dynamics (see ADR for that decision when it arises) but adds parameters and risks overfitting.
- **Neural ODE with SIR as inductive bias** — A more flexible variant where a neural correction term is added to the SIR ODE; see open questions for the PINN-style approach.

## Consequences

- Fit-time and prediction-time both target $(\beta, \gamma, s)$ or some identifiable reparameterization.
- Multi-hump and event-driven memes will fit SIR poorly. We accept that (~15% loss based on prior iteration) and consider filter/exclusion rules in future ADRs.
- We do not derive scientific conclusions about epidemiology or behavior from fitted parameter values.

## Open sub-questions

- Should we routinely compare SIR fits against a simple bump-shape baseline to confirm SIR is doing useful work?
- Is there value in fitting a sum of two SIR curves for multi-wave memes, or is filtering the cleaner choice? Tracked in `open-questions.md`.
