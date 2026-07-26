# WP-14 — Hawkes identifiability & endogeneity study (synthetic)

**Status:** open
**Date opened:** 2026-06-10
**Owner:** unassigned
**Gates:** [ADR 0014](../decisions/0014-hawkes-track-scope.md) (accepted), WP-12 (HawkesN simulator)
**Unlocks:** any real-data Hawkes *parameter* claim; the windowing-validity input to the §5 feasibility gate

## Goal

The Hawkes analogue of [WP-02](WP-02-sir-identifiability-study.md). Before predicting any Hawkes quantity from content, establish — on synthetic data where truth is known — which quantities are identifiable from realistically corrupted streams, and whether the interpretable headline number (branching ratio $n$) survives reposts and deletions. Highest-EV due-diligence after WP-12.

## Definition of done

A report at `experiments/WP-14/report.md` answering, each with the `method-discipline.md` §3 hypothesis→prediction→falsifier discipline stated *before* running:

1. **Branching-ratio identifiability.** For a grid of true $(n,\text{decay},N)$ spanning subcritical→near-critical: posterior/Fisher width of $n$ vs. stream length and observation fraction (up-to-peak vs. full decay). Where does $n$ become unidentifiable?
2. **Background vs. excitation (the endogeneity problem, R2).** Can $\mu$ (background birth rate) be separated from $\phi$ (self-excitation)? Map the regime where the $(\mu,\phi)$ posterior collapses to a ridge. This is "happened because a similar one just did" vs. "baseline rate is high here" in miniature.
3. **Repost/deletion robustness.** Inject (a) duplicate-repost events and (b) MNAR deletions into known-$n$ streams; measure the bias on fitted $n$. **Pre-registered prediction (plan note §1.5):** reposts bias $n$ *upward*; state the expected magnitude before running and compare.
4. **Windowing validity (feeds §5).** At what time / (later) embedding radius does truncating influence shift fitted $n$ by more than its posterior width? Determines whether locality windowing is "free" or introduces bias.

**Deliverable verdict:** a two-tier target list (mirroring ADR 0010) — which Hawkes quantities are safe to predict from content, and which carry identifiability flags.

## Inputs / prerequisites

- ADR 0014 accepted; WP-12 JAX HawkesN simulator + likelihood.
- JAX + NumPyro (already in the stack) for Fisher/Laplace and selected NUTS checks.

## Method

1. Reuse the WP-02 pattern: grid of truths → simulate corrupted stream → fit → Fisher/Laplace covariance + selected NUTS → eigendecompose.
2. For the endogeneity study, fit the joint $(\mu,\phi)$ posterior and report the ridge direction and condition number across the grid.
3. For robustness, parameterize repost rate and deletion-probability models; sweep and plot bias-on-$n$.
4. For windowing, refit with truncated influence at a sequence of radii; plot $\Delta n$ vs. radius against posterior width.
5. Quick-look plots before metrics (`method-discipline.md` §5): posterior pair-plots for a few representative grid points.

## Risks and ambiguities

- Fisher ≠ posterior for non-Gaussian/ridge cases — sanity-check against NUTS where it matters (same caveat as WP-02).
- The repost/deletion models are assumptions; document them (`method-discipline.md` §7) and treat results as "robustness under this corruption model," not universal.
- Near-critical ($n\to1$) is where both identifiability and simulation are hardest — expect and flag instability.

## Out of scope

- Real data (synthetic only).
- Marks / spatial identifiability (a marked extension of this study may spawn from WP-15).
- Choosing the production fitter.

## Log

- 2026-06-10 — Drafted from `research-notes/hawkes-replication-plan.md` Phase 3.

## Outcome

<pending>
