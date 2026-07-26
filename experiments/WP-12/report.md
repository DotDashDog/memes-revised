# WP-12 — SIR-Hawkes / HawkesN replication — Report

**Run:** `wp12-phase-a-001` (2026-06-10) · **Status:** Phase A done; Phase B (upstream verbatim) outstanding.
**Library versions:** jax 0.10.0, diffrax 0.7.2, numpy 2.3.5, scipy 1.17.1, matplotlib 3.10.9. Seeds in `config.json`.

> **Provenance note.** Phase A (the R-independent JAX core, synthetic recovery, mean-field bridge, final-size distribution) was produced by a background agent that completed those steps and wrote all artifacts to disk. The agent was then interrupted by a network drop during Phase B (`brew install r`), so the upstream verbatim run did not execute and this synthesis report was written afterward (in the main session) directly from the on-disk result JSONs and code. No results are restated from memory; every number below is read from `experiments/WP-12/*.json`.

## 1. What was built (Phase A — R-independent)

- **JAX HawkesN core** (`experiments/wp12/hawkesn.py`): Ogata-thinning simulator; exact HawkesN log-likelihood (∑ log λ − compensator) with the finite-population depletion factor `(1 − N_t/N)` and exponential kernel `K·θ·e^{−θτ}`; MLE/MAP fitter behind the **Branch-A single-stream interface** `fit_hawkesn(event_times[, T, N_fixed, mu_fixed]) → (params, info)`, where `info` carries a Hessian/covariance for per-fit uncertainty. `mu_fixed=0` gives the pure-SIR (no-immigrant) regime. **No dependency on holistic/field machinery** — satisfies the ADR 0014 reuse constraint.
- **Mean-field bridge** (`bridge.py`), **final-size distribution** (`size_distribution.py`, CTMC + MC), shared plotting (`plots.py`), driver (`run_wp12.py`).
- **Parameter map used** (verified against the paper): γ = θ, β = Kθ, R₀ = β/γ = K, N_hawkes = N_sir, I₀ = I.0/N; bridge holds only for background μ = 0.

## 2. Mean-field bridge — Claim A certified (deterministic); Claim B open

The bridge is two distinct claims (method-discipline §2, link-naming):

- **Claim A — analytic identity** HawkesN mean-field ≡ SIR under γ=θ, β=Kθ: **CERTIFIED** (deterministic, below).
- **Claim B — the stochastic simulator converges to that mean-field at large N: OPEN** (untested here; the original attempt failed — see history).

**Claim A — certified.** A deterministic ODE-vs-ODE test (`experiments/wp12/bridge_deterministic.py`) integrates the HawkesN mean-field independently in (s, a) excitation coordinates (`ds/dt=−s·a`, `da/dt=θ·a·(K·s−1)`, then `I=a/(Kθ)`) and compares to the independent `wp02/sir.py` solve. Pre-registered metric: relative max-abs deviation; falsifier ε > 1e-3 **derived** from the solver tolerance (Dopri5 rtol=1e-7 → expected agreement ~1e-5), not hand-picked. Result: ε ≈ 6e-7…2e-6 across K=1.5/2/3 (θ=0.3) and θ=0.1/0.3/1.0 (K=2) — ~3 orders below the falsifier, at the solver precision floor; plot residuals flat with no structure (`bridge_deterministic.png`, `bridge_deterministic_theta_var.png`). **Certified by orchestrator after inspecting the plot *and* the code (non-circularity verified in implementation: the two ODEs share no code and use different state variables), user-concurred, 2026-06-11.** Scope: confirms our *implementation* and the parameter map of the identity — not the analytic theorem itself (Rizoiu et al. 2018, accepted on the literature), and **not** Claim B.

**Claim B — certified (scoped to the ε₀-fixed family).** The stochastic HawkesN simulator (`hawkesn.py`) converges to the certified mean-field as N→∞ when the initial fraction `I₀/N = ε₀` is held fixed (here 0.01). Evidence (`claimb_*`, detail in `claimb_report.md`): the mean-vs-MF discrepancy scales as **O(1/N)** (fitted log-log slopes −1.01…−1.13, all 95% CI upper bounds < 0 — the *bias-of-mean* rate, not the 1/√N the first design wrongly assumed); the single-run std scales ≈ N^(−1/2) cleanly for K=1.5 and steeper (−0.65/−0.75, benign) for K=2/3; and the finite-N **phase lead closes monotonically** — by N=8000 the stochastic mean overlies the MF (this is the failure mode that sank the withdrawn test, here resolved by fixing `I₀/N`). The large-N residual shrinks as ~1/N → genuine finite-N correction, not a bug. **Certified by orchestrator after inspecting the four decisive plots + fitted exponents, user-concurred, 2026-06-11.** Scope: validates the `ε₀`-fixed mean-field family **only** — the single-seed (I₀=1) regime, which is the actual real-meme regime, is NOT covered and is the harder case (see open-questions). The original sim-based attempt was **withdrawn** for conflating Claims A and B (phase-blind RMSE on an I₀=1, N=500 average; post-hoc conditional selection); withdrawn plots retained: `bridge_check_K{1.5,2.0,3.0}.png`.

## 3. Final-size distribution — qualitatively reproduces the bimodal/criticality story (partial)

CTMC (N=100) and MC (N=1000, 20k draws) agree and reproduce the expected subcritical→supercritical transition:

| K | small-outbreak P | large-outbreak P |
|---|---|---|
| 0.8 | 0.97 (CTMC) / 1.00 (MC) | 0.03 / 0.00 |
| 1.2 | 0.82 / 0.87 | 0.18 / 0.13 |
| 2.0 | 0.52 / 0.50 | 0.48 / 0.50 |
| 3.0 | 0.34 / 0.34 | 0.66 / 0.66 |

Subcritical (K<1) outbreaks almost always die small; supercritical mass shifts to large outbreaks, crossing ~50/50 near K≈2. CTMC and MC match closely, cross-validating both. **Partial vs. DoD:** the upstream *headline* figure is the **prior→posterior collapse** (bimodal a-priori final-size distribution becoming unimodal once part of the cascade is observed). What ran here is the a-priori distribution / outbreak-size split; the conditioning-on-partial-observation step is **not yet reproduced**. Plots: `final_size_ctmc_N100.png`, `final_size_mc_N1000.png`.

## 4. Synthetic parameter recovery — FLAG, not a clean pass (needs a diagnostic)

20 synthetic single cascades (K∈[1.5,3], θ∈[0.2,1.0], N∈[100,500], I₀=1), fit with 3 restarts. Pre-registered expectation was median recovery of the natural parameters. **Observed:**

- **The natural parameters K and θ are frequently NOT recovered individually.** Six cases pin `K_fit` against the optimiser's upper bound (K=20) with θ collapsing toward zero (cases 1, 3, 4, 11, 16, 19 — K errors of 490–1058%, θ errors >90%).
- **Their product β = Kθ is recovered substantially better** (e.g. case 3: K err 1058%, θ err 93%, but **Kθ err 24%**; case 14: K err 79%, θ err 42%, **Kθ err 3.8%**). Median |Kθ error| ≈ 28% vs. K errors routinely >100%.
- **The split is regime-dependent.** Well-recovered cases (0, 2, 8, 12, 13: all errors small) have **fast decay (θ≈0.6–0.96)** — the kernel fully relaxes inside the window. The pinned cases have **slow decay (θ≈0.2–0.41)**, where 1/θ is large relative to the observation window. In the hardest regime (low θ + low N + few events, e.g. cases 1, 4, 17) **even β degrades** (50–63%).

**Interpretation — held open per `method-discipline.md` §6 (anti-cargo-cult).** This is the K–θ identifiability ridge along Kθ = β, and it is *consistent with* WP-02's finding that low-count/near-threshold regimes are unreliable and with ADR 0010's directive to predict identifiable combinations (β = Kθ = R₀·γ) rather than K, θ separately. **But "consistent with" is not "confirmed."** The boundary-pinning could equally be a fitter pathology: the K∈[0.01,20] bound being hit, too-short observation windows (T = 30/θ may be inadequate at low θ), or insufficient restarts. Per the anti-cargo-cult rule, this is recorded as an **open question requiring a diagnostic, not a result**:

- **Diagnostic (proposed):** profile-likelihood in (K, θ) for a pinned case — if the likelihood is genuinely flat along Kθ = const, it is identifiability; if it has a clear interior optimum the optimiser missed, it is a fitter/window bug. Hypothesis: flat ridge for low-θ short-window cases. Falsifier: a sharp interior peak. Also re-run a pinned case with a longer window (T = 100/θ) and N free vs. fixed.

Results: `synthetic_recovery_results.json`, plot `synthetic_recovery.png`.

## 5. Outstanding — Phase B (upstream verbatim) NOT DONE

- **R was never installed** (no R on the machine; the `brew install r` step was interrupted by a network drop). The upstream `computationalmedia/sir-hawkes` tutorial was **not run verbatim**, so the **faithful-replication claim (DoD item 1) is not established** — only our own JAX implementation has been exercised.
- Consequently the **second cross-validation** (our JAX fits vs. upstream R fits on the same tweet cascades) is also outstanding.
- Until both are done, WP-12 stays **`active`**, not `done` (per the user's explicit instruction not to mark done while the upstream run is outstanding).

## 6. Verdict / next steps

Phase A delivered a JAX HawkesN core with the **Branch-A interface in place** for later per-cluster reuse, its **final-size criticality behaviour is reproduced**, and **both mean-field bridge claims are now certified** — Claim A (analytic identity) and Claim B (stochastic convergence, ε₀-fixed) (§2). Remaining before WP-12 closes: **parameter recovery / identifiability is flagged**, not concluded (§4); the size-distribution **prior→posterior collapse** figure is partial; and **Phase B** (real-data fits) is outstanding. To close WP-12:

1. **Recovery + identifiability study** (§4): run the profile-likelihood diagnostic on a pinned case to separate genuine (K,θ) ridge degeneracy from fitter/window/IC pathology; then replicate the paper's recovery protocol (multiple cascades, full observation) reporting *identifiable* quantities (β=Kθ, N, final size) per ADR 0010. Feeds WP-14.
3. Reproduce the **prior→posterior final-size collapse** (§3) to fully match the headline figure.
4. Complete **Phase B**: install R, run the upstream tutorial verbatim, and cross-check our fits against it.
