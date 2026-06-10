# WP-02 Report — SIR Identifiability Study

**Status:** first-pass Hessian grid sweep and representative NUTS validation complete.

## Current Modeling Choices

- Observation model: binned Poisson counts.
- Bin mean: $\mu_j = s \int_{t_j}^{t_{j+1}} I(t)dt$.
- Natural parameters: $(\beta, \gamma, s, I_0)$.
- Computational fit coordinates: $(\log\beta, \log\gamma, \log s, \mathrm{logit}(I_0))$.
- Solver: JAX + diffrax, with an auxiliary state for $\int I(t)dt$.
- Local posterior geometry: MLE plus Hessian / Laplace approximation.
- Full posterior checks: NumPyro NUTS on representative cases.
- Posterior visualization: `corner`.

## Smoke Checks

The initial smoke case was run with:

```sh
uv run python experiments/wp02/run_synthetic_case.py --days 20 --scale 800 --output-dir experiments/WP-02/smoke-test
```

Result summary:

- True parameters: $\beta=1.2$, $\gamma=0.35$, $s=800$, $I_0=0.003$.
- Total expected counts: 2156.15.
- Total simulated counts: 2097.
- MLE: $\beta=1.1887$, $\gamma=0.3918$, $s=886.96$, $I_0=0.00345$.
- Hessian eigenvalues: 38.96, 49.61, 4844.48, 15045.98.
- Corner output: `experiments/WP-02/smoke-test/hessian_corner.png`.

A tiny NUTS smoke run was also executed to verify that NumPyro can sample through the diffrax forward model:

```sh
uv run python experiments/wp02/run_synthetic_case.py --days 8 --scale 300 --output-dir experiments/WP-02/nuts-smoke --nuts --nuts-warmup 10 --nuts-samples 20
```

This was only a plumbing check, not a scientifically meaningful posterior sample. It produced `experiments/WP-02/nuts-smoke/nuts_corner.png`.

## Full Synthetic Case 001

Command:

```sh
uv run python experiments/wp02/run_synthetic_case.py --output-dir experiments/WP-02/full-synthetic-001 --nuts --nuts-warmup 1000 --nuts-samples 1000 --nuts-chains 4 --nuts-chain-method sequential --nuts-init map
```

Setup:

- True parameters: $\beta=1.2$, $\gamma=0.35$, $s=5000$, $I_0=0.003$.
- Observation window: 60 days, daily bins.
- Expected total counts: 13761.89.
- Simulated total counts: 13616.
- NUTS: 4 sequential chains, 1000 warmup + 1000 samples per chain, initialized at the MAP.

Fit result:

- MAP/MLE: $\beta=1.20117$, $\gamma=0.34661$, $s=4891.59$, $I_0=0.003054$.
- Hessian eigenvalues: 199.36, 836.30, 32235.69, 86387.50.
- NUTS diagnostics: all $\hat R \leq 1.0003$; minimum effective sample size $\approx 1946$; no divergences.

Posterior comparison:

- Hessian and NUTS posterior means agree extremely closely in unconstrained coordinates:
  - $\Delta\log\beta = 4.2\times10^{-5}$
  - $\Delta\log\gamma = -8.1\times10^{-4}$
  - $\Delta\log s = -1.2\times10^{-3}$
  - $\Delta\mathrm{logit}(I_0) = -2.2\times10^{-4}$
- Natural-parameter posterior standard deviations are also nearly identical:
  - Hessian: $\sigma_\beta=0.01151$, $\sigma_\gamma=0.00860$, $\sigma_s=146.58$, $\sigma_{I_0}=2.03\times10^{-4}$.
  - NUTS: $\sigma_\beta=0.01166$, $\sigma_\gamma=0.00866$, $\sigma_s=148.25$, $\sigma_{I_0}=2.07\times10^{-4}$.

Artifacts:

- `experiments/WP-02/full-synthetic-001/result.json`
- `experiments/WP-02/full-synthetic-001/hessian_corner.png`
- `experiments/WP-02/full-synthetic-001/nuts_corner.png`

## Hessian Grid Sweep 001

Command:

```sh
uv run python experiments/wp02/run_grid_sweep.py --output-dir experiments/WP-02/grid-sweep-001
```

Grid:

- $\beta \in \{0.3, 1.0, 3.0\}$.
- $\gamma \in \{0.1, 0.35, 1.0\}$.
- $s \in \{300, 1500, 7500\}$.
- $I_0 \in \{0.001, 0.003, 0.01\}$.
- Observation window: 60 days, daily bins.

Results:

- 81/81 optimizer runs reported convergence.
- 69/81 Hessians were positive definite with valid covariance estimates.
- 4/81 cases hit a parameter/prior boundary.
- 12/81 cases had non-positive Hessians; these are exactly the kinds of cases where the local Gaussian approximation should not be trusted.

Artifacts:

- `experiments/WP-02/grid-sweep-001/hessian_grid.csv`
- `experiments/WP-02/grid-sweep-001/summary.json`

## NUTS Validation Cases

The grid sweep selected six cases for full posterior checks:

- `case-008`: well identified.
- `case-067`: median-condition representative case.
- `case-013`: high condition number / weakly identified.
- `case-019`: worst valid-Hessian uncertainty, very low count.
- `case-018`: lowest-count and boundary case, zero observed counts.
- `case-052`: non-positive Hessian case.

Each was run with 4 sequential chains, 500 warmup draws, and 500 retained samples per chain.

Summary:

- `case-008` (`~66.9k` observed counts): Hessian and NUTS matched closely. All $\hat R \leq 1.001$, no divergences.
- `case-067` (`~4.3k` observed counts): Hessian and NUTS matched closely. All $\hat R \leq 1.002$, no divergences.
- `case-013` (`105` observed counts): Hessian covariance was extremely ill-conditioned and not a useful approximation. NUTS had poor convergence ($\hat R$ up to `1.79`) and 44 divergences.
- `case-019` (`4` observed counts): Hessian approximation was effectively meaningless. NUTS had poor convergence ($\hat R$ up to `1.21`) and 386 divergences.
- `case-018` (`0` observed counts): Hessian was non-positive and the MAP hit the $\gamma$ upper bound. NUTS mixed in rank diagnostics but had 186 divergences and a broad prior-dominated posterior.
- `case-052` (`577` observed counts): Hessian was non-positive. NUTS did not converge well ($\hat R$ up to `1.17`) and found a very different ridge-like posterior from the MAP/Hessian approximation.

Interpretation:

- In well-observed, interior cases, Hessian/Laplace is a good approximation to the NUTS posterior and is cheap enough for grid sweeps.
- Non-positive Hessians, boundary hits, very large condition numbers, and very large relative posterior widths are reliable warning signs.
- Low Poisson count cases are not just "wide"; they can be qualitatively non-Gaussian, ridge-like, and prior-sensitive. These should be handled with posterior sampling or flagged as unfit/unidentifiable.
- A rough preliminary threshold is that fits with only tens of total counts are unusable for parameter targets. Fits with hundreds of counts can still be degenerate depending on $(\beta,\gamma,I_0)$ and observation window. Thousands of counts are usually much better behaved in this first grid.

## Key Lessons For Future Agents

- The JAX/diffrax SIR fitter works for synthetic binned Poisson data. MAP fits can produce visually good posterior-predictive curves even in cases where the underlying natural parameters are not recoverable.
- Do not equate good curve fit with parameter identifiability. Case `case-052` is the clearest example: truth was $(\beta,\gamma,s,I_0)=(1,1,7500,0.003)$ and MAP recovered a nearby point, but NUTS found a very different ridge-like posterior with similarly good curves. This happens because the system is near the SIR threshold: $\beta(1-I_0)-\gamma \approx -0.003$, so the data mostly identify net growth/decay and initial amplitude rather than individual $(\beta,\gamma,s,I_0)$.
- The most robust operational rule from this pass is: trust MAP curves more readily than individual SIR parameters. Treat non-positive Hessians, boundary hits, poor NUTS diagnostics, very low total counts, and near-threshold dynamics as identifiability warnings.
- Hessian/Laplace is useful as a fast screening tool in well-identified, interior regimes, but should not be used as a final uncertainty estimate for low-count, boundary, non-positive-Hessian, or ridge-like cases.

## Next Steps

1. Decide whether WP-02 needs a second pass before ADR 0010. The likely useful additions are a targeted low-count threshold study and observation-window variants.
2. If continuing WP-02, vary total expected counts while holding curve shape fixed, and estimate the count threshold at which each parameter becomes unusably uncertain or non-Gaussian.
3. Add observation-window variants if needed: full curve, through peak only, and decay-only/truncated windows.
4. Draft the ADR 0010 recommendation. Current evidence favors either predicting curve-level summaries or using SIR parameters with identifiability flags/uncertainty, rather than treating all natural parameters as equally recoverable targets.

