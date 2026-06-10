# 0013 — JAX as the primary numerical backend

**Status:** accepted
**Date:** 2026-05-19
**Gates:** —

## Context

WP-02 needs differentiable SIR solves for Fisher/Hessian-based identifiability checks and for occasional full posterior sampling with NUTS. Later speculative work may also train predictors end-to-end through an ODE solver.

The user is already comfortable with JAX for Bayesian inference work and explicitly requested that WP-02 use JAX + diffrax.

## Decision

Use JAX as the default numerical backend for new research code, especially SIR fitting, differentiable simulation, and Bayesian inference. For WP-02, use diffrax for ODE solves and NumPyro for NUTS.

Ideally, the whole project will remain JAX-based for coherence. However, this ADR does not definitively choose the framework for the embedding or predictor models; those decisions should be made when the relevant work packages start. This also does not forbid PyTorch or SciPy where they are clearly better suited, but new numerical research code should start from JAX unless there is a documented reason not to.

## Alternatives considered

- **SciPy-first:** Simple and mature, but gradients through the integrator are awkward and would make Hessian/NUTS workflows less direct.
- **PyTorch-first:** Useful if later predictor models are PyTorch-based, but the project currently has stronger Bayesian-inference reasons to prefer JAX.
- **JAX-first:** Differentiable solvers, vectorization, and NumPyro make it a good default for WP-02 and likely later inference-heavy work.

## Consequences

- Runtime dependencies include `jax`, `diffrax`, and `numpyro`.
- WP-02 can compute Hessians and run NUTS through the same differentiable forward model.
- Future model-framework choices may still be revisited if embedding or predictor work has stronger PyTorch ecosystem constraints.

## Open sub-questions

- If the final predictor architecture uses a PyTorch-native embedding stack, should the predictor stay in PyTorch while SIR fitting remains in JAX?