# WP-02 — SIR identifiability study on synthetic data

**Status:** active; first-pass synthetic study complete
**Date opened:** 2026-05-19
**Owner:** unassigned
**Gates:** —
**Unlocks:** [ADR 0010](../decisions/0010-sir-parameterization.md), informs [ADR 0005](../decisions/0005-sir-initial-conditions.md)

## Goal

Quantify the identifiability of SIR fit parameters $(\beta, \gamma, s, I_0)$ from realistic noisy interest-over-time data, so that ADR 0010 can choose a prediction parameterization that targets only well-identified quantities.

This is the single highest-EV experiment we can run before any data work, because the answer changes what the prediction model is even allowed to learn.

## Definition of done

A markdown report at `experiments/WP-02/report.md` containing:

- For a grid of "true" SIR parameter values spanning realistic meme regimes:
  - The posterior covariance / Fisher information of the fit, given a realistic noise model.
  - Identifiable parameter combinations (eigendirections of the inverse Fisher) and unidentifiable ones.
- Sensitivity to: number of observation points, fraction of curve observed (only-up-to-peak vs. full decay), noise level, and presence of outlier "repost spikes."
- Recommended prediction targets: which subset of $(\beta, \gamma, s, I_0)$ to predict, or which reparameterization to use.

And a corresponding draft update to `0010-sir-parameterization.md` recording the recommendation (still `status: proposed`).

## Inputs / prerequisites

- None external. Pure synthetic study.
- A working SIR integrator. Use JAX + [`diffrax`](https://github.com/patrick-kidger/diffrax) so Fisher information can be computed via autodiff and full posterior checks can use NumPyro/NUTS.

## Method

1. **Define a realistic parameter grid.** Use rough ranges from the prior iteration's fits (or sensible defaults if those aren't easily extractable):
   - $\beta$ across roughly 0.1–10 per day (timescales from days to weeks).
   - $\gamma$ across roughly 0.05–2 per day.
   - $s$ across the expected amplitude range for share-of-activity signals.
   - $I_0$ small (e.g., $10^{-3}$–$10^{-1}$).

2. **Define a realistic noise model.** Start with binned Poisson counts:
   - $\mu_j = s \int_{t_j}^{t_{j+1}} I(t)\,dt$.
   - $y_j \sim \mathrm{Poisson}(\mu_j)$.
   - Outlier/repost spikes are deferred until the Poisson-only baseline is understood.

3. **For each grid point:**
   - Simulate an observed $I(t)$ from the noise model.
   - Fit SIR to the simulated data using MLE with Hessian / Laplace approximation.
   - Run NUTS for representative grid points to sanity-check Hessian-based posterior geometry.
   - Record the posterior covariance and its eigendecomposition.

4. **Aggregate:**
   - For each parameter, plot expected uncertainty (CI width) across the grid.
   - For pairs of parameters, plot correlation across the grid.
   - Identify any combinations with consistent high uncertainty (degeneracy).
   - Identify any well-identified combinations (e.g., does $\log R_0$ stay tight while $\log\beta$ blows up?).

5. **Sensitivity studies:**
   - Cut the observation window to "peak only" — how much identifiability is lost?
   - Increase Poisson noise — at what level does $\gamma$ become essentially unidentifiable?
   - Add occasional outliers — what fitter is robust?

6. **Write the recommendation.** Either "predict $(\beta, \gamma, s)$" or "predict $(\log R_0, \log \gamma, \log \text{peak})$" or whatever the evidence suggests. Explicitly cite which directions in parameter space are well-vs-poorly identified.

## Risks and ambiguities

- Parameter grid ranges may not match real Reddit-meme regimes — we don't have those fits yet. Worst case, the recommendation gets revised when WP-08 produces real fits. Document grid choices clearly.
- Fisher information ≠ posterior covariance for non-Gaussian posteriors. Note this where it matters; the Laplace approximation is usually fine for these problems but should be sanity-checked against NUTS on a few grid points.
- Don't over-engineer. v1 of this study should be doable in 2–3 focused sessions.

## Out of scope

- Fitting real Reddit data (WP-08).
- Multi-hump or sum-of-SIRs identifiability (deferred; flag as open question if it comes up).
- Choosing the actual fitter to use in production (deferred to WP-08).

## Log

<!-- Newest at top. -->

- 2026-11-27 — **Completion pass (Opus agent; UNCERTIFIED: checked by its author, plus the
  orchestrator reading the main plot; no independent review yet).** Pre-registration and
  deviations: `experiments/WP-02/expectations-2026-11.md`. Main plot:
  `experiments/WP-02/e1-analysis/e1_trust_map_posterior.png`; tables in the same folder.
  Method: 84 combinations of curve shape (R0 = 3, 1.5, 10, 0.86, 1), observation window
  (to 0.7x the peak time, to the peak, to 1.5x, full curve) and observed post count (10 to
  3000), 400 simulated datasets each; count how often a stated 90% interval contains the
  truth (below 85.5% is called misleading), for Hessian intervals and for the full posterior.
  *Main finding:* for a meme caught at or before its peak, only the early growth rate and
  the initial post rate can be estimated, from about 100 posts. Final size, peak time and
  peak height cannot, and their stated 90% intervals contain the truth only 41–80% of the
  time for R0 of 3 and 1.5: misleading, not merely wide. With the peak observed and at least
  about 100 posts, final size, peak height, peak time and growth rate are trustworthy; R0
  as well on complete curves. Raw beta, gamma, s, I0 need about 1000 posts on complete curves.
  *Flag for an untrustworthy fit:* posterior probability that the peak lies inside the
  observed window above 0.9. Among fits passing it, coverage is 89–90% pooled over growing
  shapes; one gap (R0 = 1.5, just past the peak, 30–300 posts: final size 69–78%). May's
  flag set (Hessian positive definite, off bounds, converged) is not enough: among the
  pre-peak fits it passes, peak-time coverage fell to 50% in the worst cell.
  *May conclusions:* "Hessian matches the posterior when well identified" held. "Low counts
  break it" is half right: it depends on target and window more than count. The 12
  non-positive Hessians were mostly fits still climbing a ridge or on a bound, not odd
  maxima. The full posterior is not a safe fallback under the inherited flat prior: where
  data do not pin a quantity down, the prior decides, and an arbitrary bound on I0 moved
  pre-peak intervals by 53–65%.
  *Noise:* with overdispersed (negative-binomial) counts, Poisson-based intervals are badly
  overconfident and get worse with more data (final-size coverage 43% at 1000 posts for
  moderate overdispersion, 16% for strong); fitting a negative-binomial likelihood restores
  85–89%. Repost-style spikes bias final size up by 14–18% and neither likelihood is robust
  to them for final size; peak height and time stay usable.
  *Bearing on ADR 0010:* its instruction not to use raw parameters as labels stands. Its
  premise that the chosen primary targets (curve mass, peak time, peak height) are stable
  does not hold for memes caught before or at their peak. Agent's options: (a) keep the
  targets but condition on the peak-in-window flag and fit with an overdispersed likelihood,
  (b) evaluate only memes observed past their peak, (c) predict only growth rate and initial
  rate for memes in progress; it recommends (a). Reopening the ADR is the user's decision.
  *Limits:* synthetic curves are exactly SIR, so real data can only be harder; fits start
  near the truth; decaying and at-threshold cells have rougher posterior numerics; not run:
  overdispersion with truncated windows, a spike-robust fitter, a hierarchical prior.
  The agent's full written report could not be saved as a file (the harness refuses report
  files from subagents); this entry and the outputs above are the record.

- 2026-05-19 — User accepted ADR 0010.
- 2026-05-19 — Drafted ADR 0010 recommendation: use stable curve-level summaries as primary prediction/evaluation targets, and retain transformed SIR parameters only as auxiliary diagnostic targets with identifiability metadata.
- 2026-05-19 — Recorded first-pass conclusion: JAX/diffrax SIR fitting works for synthetic binned Poisson curves, and MAP/NUTS posterior-predictive curves can fit well even when natural SIR parameters are not individually identifiable. Near-threshold cases such as `case-052` mostly identify net growth/decay and initial amplitude, not unique $(\beta,\gamma,s,I_0)$.
- 2026-05-19 — Accepted JAX-first ADR 0013. Ran 81-case Hessian grid sweep and six selected NUTS validations. Hessian/Laplace matched NUTS in well-identified cases, but failed in low-count, boundary, and non-positive-Hessian regimes. Added low-count uncertainty threshold study to next steps.
- 2026-05-19 — Started implementation. User chose binned Poisson counts, default natural fit parameters $(\beta, \gamma, s, I_0)$, JAX + diffrax for differentiable solves, NumPyro/NUTS for selected full posterior checks, and `corner` for posterior plots. Added a scaffold under `experiments/wp02/` and smoke outputs under `experiments/WP-02/`.
- 2026-05-19 — Package opened.

## Outcome

First-pass conclusion: SIR curve fitting is viable as empirical curve compression, especially for fitting and predicting observed time-series shape. However, individual natural SIR parameters are not guaranteed to be recoverable. Low counts, non-positive Hessians, boundary hits, and near-threshold dynamics should be treated as identifiability warnings. This should inform ADR 0010: prediction targets should either be curve-level summaries or SIR parameters accompanied by identifiability flags/uncertainty, not raw $(\beta,\gamma,s,I_0)$ treated as uniformly reliable labels.
