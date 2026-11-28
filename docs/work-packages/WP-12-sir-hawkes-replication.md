# WP-12 — SIR-Hawkes / HawkesN replication (anchor)

**Status:** active
**Date opened:** 2026-06-10
**Owner:** unassigned
**Gates:** [ADR 0014](../decisions/0014-hawkes-track-scope.md) (accepted)
**Unlocks:** WP-13, WP-14; connects to [WP-02](WP-02-sir-identifiability-study.md) / [ADR 0010](../decisions/0010-sir-parameterization.md)

## Goal

Faithfully replicate the SIR-Hawkes / HawkesN result (Rizoiu et al., WWW 2018) and reimplement its core in JAX, establishing the depleting finite-population Hawkes machinery and numerically confirming the HawkesN→SIR mean-field bridge that the project's SIR-as-empirical-compression framing relies on.

## Definition of done

A report at `experiments/WP-12/report.md` containing:
1. **Upstream replication.** The `computationalmedia/sir-hawkes` R/notebook tutorial run as-is, reproducing (a) its stochastic-SIR↔HawkesN parameter-recovery check and (b) its headline final-size-distribution figure (bi-modal prior collapsing to single-mode posterior as more of the cascade is observed). Screenshots/values vs. paper.
2. **JAX core.** A minimal JAX HawkesN simulator + log-likelihood + MLE/MAP fit, validated two ways: matches upstream R fits on the same tweet cascades (within tolerance), and recovers known $(\beta,\gamma,N)$ from synthetic streams.
3. **Mean-field bridge check.** Numerical confirmation that the HawkesN mean-field limit reproduces SIR ODE $I(t)$ fits of the kind studied in WP-02 — closing the analytic claim in `research-notes/hawkes-processes.md` §3.

## Inputs / prerequisites

- ADR 0014 accepted.
- Upstream repo + its three included tweet-cascade datasets.
- An R environment for the upstream run (reference only; not added to the project's uv/Python stack).
- JAX + diffrax (already in the stack) for the core.

## Method

1. Clone and run the upstream tutorial unmodified; record outputs and any version friction.
2. Implement HawkesN simulation (thinning/Ogata) and the marginalized event-rate likelihood in JAX. Validate the simulator's branching ratio and final-size statistics against analytic expectations.
3. Fit the JAX core to the upstream cascades; compare fitted $(\kappa,\beta,\theta,N)$ (upstream parameterization) to the R results.
4. Derive/verify the mean-field ODE and confirm it matches a diffrax SIR solve for matched parameters; reconcile parameter maps with WP-02's $(\beta,\gamma,s,I_0)$.
5. State, per `method-discipline.md` §3, the predicted relationship *before* running each comparison; record observed vs. predicted.

## Design constraint — reusable per-cluster fitter (Branch A)

Per ADR 0014, the committed WP-15 fallback ("Branch A") repurposes this differentiable HawkesN likelihood as a better per-cluster fitter than the bin-counts→SIR-ODE step (feeding WP-08). To make that reuse free, build the JAX core behind a clean interface from day one: **`(event_times[, marks]) → fitted params + uncertainty`** for a *single* event stream, with no dependence on the holistic/field machinery. The simulator and likelihood should be usable standalone on one cluster's timestamps.

## Risks and ambiguities

- R↔JAX parameterization mismatch is the most likely time sink; build the parameter-map table first.
- Tweet cascades are not memes; this WP validates *machinery and the bridge*, not meme dynamics. Do not over-read its fits.
- Final-size posterior is sensitive to the assumed $N$; document the prior.

## Out of scope

- Any neural model (WP-13).
- Real r/memes data (WP-13 single-stream onward).
- Spatial marks (WP-15).

## Log

- 2026-11-27 — **November pass (Opus agent; UNCERTIFIED: checked by its author, plus the
  orchestrator reading the coverage plot; no independent review yet).** Pre-registration,
  failed predictions and deviations: `experiments/WP-12/expectations-2026-11.md`. Outputs:
  `experiments/WP-12/nov_*` and `nov_upstream/`; code `experiments/wp12/*_2026_11.py`,
  `exact_profile.py`.
  *Fitter trustworthiness (the item later work depends on):* the per-cascade fitter should
  not be used to report K (offspring per post), theta (decay rate), N (population) or
  K*theta for individual memes. For a cascade started by one post and watched to its end,
  only the early growth rate theta*(K-1) and the final size are reliably recovered, and
  the stated 95% intervals are too narrow even for the growth rate (they contain the truth
  66–83% of the time; for K, 28–66%). In the paper's own setting (300 seeds, K = 5)
  everything recovers and intervals are near nominal. Evidence: 200 simulated cascades per
  scenario against an exact likelihood grid with no optimiser (`nov_ident_coverage.png`,
  `nov_ident_recovery.png`). Cause, tested three ways: not selection of cascades that took
  off; partly too little data; mainly observing the cascade to its end, where N piles up
  against the observed count (`nov_two_by_two.png`: stopped windows cover 93–97%, finished
  cascades 71–86%). Why over-confidence persists at about 7,000 posts was not established.
  *June conclusions:* **withdrawn** — "K*theta is the identifiable combination", "N is
  recoverable" in the meme-like regime, and the "flat profile, resolvable by pooling"
  verdict; the June profile-likelihood routine was broken (wrong-sized gradient, swallowed
  errors, up to 14 log-likelihood units off) and its gate was passed on a 4-event cascade.
  The paper-setting recovery held. The two certified bridge checks were not re-examined.
  *Code findings:* the likelihood is correct without a background rate and wrong by exactly
  -mu*T/N with one, which must be fixed before real memes; the existing optimiser lands
  more than 1.92 log-likelihood units below the true maximum on 5–34% of simulated
  cascades and 43% of 30 real tweet cascades.
  *Headline figure:* reproduced by the authors' unmodified R code and by our own (agreement
  4e-16). Caveat: the figure comes from the SIR Markov chain, which is not HawkesN's own
  final-size distribution; at K = 2 from one seed the chain gives a 0.50 chance of fizzling
  against 0.20 simulated, and overstates the remaining fizzle chance 8–26 times after five
  events. The June final-size table is right for SIR and should not be read as HawkesN.
  *Upstream run:* the tutorial reproduces on R 3.5.1 only (environments in `~/wp12-ext`,
  3.5 GB); medians match the notebook. Upstream optimises a different, unbounded objective
  and silently drops failed fits. Where it converges, the JAX fitter agrees (20 of 20). The
  cross-check on real tweet cascades could not be made: the upstream fitter returned no
  usable fit on 30 of 30.
  *Single-seed convergence (the gap left in June):* the shape of the curve converges to the
  mean-field (SIR) curve as N grows (gap shrinks as N^-0.5); the timing does not (spread of
  1.3–2.0 growth-times at every N). Take-off probability converges to the HawkesN value,
  not the SIR one.
  *Not checked:* background arrivals, unknown seed count, other kernels, real meme data,
  whether a prior or bootstrap repairs the intervals.
  *Next, per the agent:* independent review of the identifiability results; replace the
  optimiser with the closed-form-K grid search; fix the background-rate term; intervals by
  a method that survives the boundary.

- 2026-06-11 — **Mean-field bridge Claim B CERTIFIED (scoped).** Stochastic HawkesN simulator → mean-field convergence, with `I₀/N=ε₀` fixed (Sonnet subagent, design-checkpointed then run; `experiments/WP-12/claimb_*`). Orchestrator inspected 4 decisive plots + fitted exponents, user-concurred. Evidence: mean-bias scales O(1/N) (slopes −1.01…−1.13, CIs<0 — confirming the orchestrator's correction over the agent's original 1/√N design); single-run std ≈N^(−1/2) (clean for K=1.5, benignly steeper for K=2/3 — noted sub-finding); finite-N phase-lead closes by N=8000; residual is genuine finite-N, not a bug. **Scope:** ε₀-fixed family only; single-seed I₀=1 (the real-meme regime) NOT covered → logged as open question. Process note: the agent stopped at the design checkpoint (surface-before-expensive-run), did not self-certify, and honestly flagged the H2 anomaly — the safeguards worked, and orchestrator review caught the O(1/N)-vs-1/√N error the rules alone didn't. **Both bridge claims now certified.** Remaining WP-12: recovery/identifiability (§4 of report), size-distribution prior→posterior figure, Phase B real-data fits.
- 2026-06-11 — **Mean-field bridge Claim A CERTIFIED.** Replaced the withdrawn sim-RMSE test with a deterministic HawkesN-MF-ODE vs SIR-ODE comparison (`experiments/wp12/bridge_deterministic.py`, Sonnet subagent). Independently-derived (s,a) excitation ODE vs `wp02/sir.py`; principled falsifier (1e-3, derived from solver rtol); ε≈6e-7…2e-6 across K=1.5–3, θ=0.1–1.0 — at solver precision floor, residual plot flat. Orchestrator inspected plot + code (non-circularity verified), user-concurred → certified. **Scope:** confirms implementation + parameter map of the identity, *not* the simulator-convergence claim (**Claim B**, now the open bridge item) and not the literature theorem. The subagent followed the new rules cleanly (matched test to claim, derived its threshold, named the blind spot, plot agreed with number, did not self-certify). Rule earned: method-discipline §2 now requires naming which link in an identity/limit chain a test covers.
- 2026-06-10 — **Phase A done (R-independent); Phase B outstanding.** Built JAX HawkesN core (`experiments/wp12/hawkesn.py`) with Ogata simulator, exact finite-population log-likelihood, and the Branch-A single-stream fitter `fit_hawkesn(event_times[,…]) → (params, info+covariance)`. Results in `experiments/WP-12/` (report: `report.md`). Findings: (1) **mean-field bridge NOT established** — the original RMSE-based "pass" was withdrawn on review as a mis-specified test (compared a deterministic identity to a finite-N stochastic average; phase-blind, peak-deflated metric; result-dependent cond/uncond selection; the plot shows the conditional mean *leading* the SIR ODE). Correct deterministic HawkesN-ODE vs SIR-ODE test is pending. (2) **final-size criticality reproduced** (subcritical→supercritical split, CTMC≈MC); the prior→posterior *collapse* headline figure is only partially done. (3) **synthetic recovery FLAG (not concluded):** K and θ individually often unrecovered (6 cases pin K at the 20 bound, θ→0) while β=Kθ recovers far better — the K–θ ridge, consistent with WP-02/ADR 0010 but needs a profile-likelihood diagnostic to separate identifiability from fitter/window pathology. **Phase B (upstream verbatim run) did NOT execute** — R not installed; `brew install r` was interrupted by a network drop. Faithful-replication claim + JAX-vs-upstream cross-check remain open → status stays `active`. Next: profile-likelihood diagnostic (feeds WP-14), prior→posterior final-size figure, then install R + run upstream.
- 2026-06-10 — Drafted from `research-notes/hawkes-replication-plan.md` Phase 1. Awaiting ADR 0014 acceptance.

## Outcome

<pending>
