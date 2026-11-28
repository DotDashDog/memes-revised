# WP-02 completion study: expectations written before each run (November 2026)

Author: completion agent, 2026-11-27. Every result referenced here is UNCERTIFIED
(checked only by the agent that produced it).

Shared setup (code: `experiments/wp02/calib.py`). One "cell" = fixed true SIR
parameters + bin layout + expected number of observed posts. Per cell, R replicate
datasets are simulated; each gets (a) a Poisson maximum-likelihood fit started from the
best of 60k data-scored candidate points (never from the truth) with Laplace / delta-method
90% intervals, and (b) a full posterior under flat priors on (log beta, log gamma, logit I0)
in the first pass's box and a flat prior on log s, computed by importance sampling with s
integrated out analytically. Coverage = fraction of replicates whose 90% interval contains
the truth. With R = 400 the binomial standard error of a coverage near 90% is
sqrt(.9*.1/400) = 1.5 percentage points (pp); I treat |coverage - 90| > 4.5 pp (3 SE) as a
real deviation and anything inside as indistinguishable from nominal.

Disclosure: before writing this I ran a 100-replicate smoke test of the machinery on one
cell family (beta=0.6, gamma=0.2, full curve, 100 and 3000 posts) and saw both methods near
90% coverage there. E1 predictions for that corner are therefore not blind.

## E1. Counts x observation window (run dir `e1-window-counts/`)

Design: shapes A (beta .6, gamma .2, R0 3), B (beta 1, gamma .67, R0 1.5), C (beta .5,
gamma .05, R0 10), all I0 = .003, daily bins; windows ending at 0.7x, 1.0x, 1.5x the true
peak time and at the time 99% of posts have occurred ("full"); expected OBSERVED posts in
the window 10, 30, 100, 300, 1000, 3000. Plus full-window-only shapes D (beta .3, gamma .35,
R0 .86, pure decay) and E (beta 1, gamma 1, at threshold; the first pass's case-052 regime),
60 daily bins. R = 400, seed base 20261127.

Claims and expectations:

1. Window matters more than count. Full window, R0 > 1: all ten targets usable; posterior
   coverage within 4.5 pp of 90% at every count >= 30; 90% interval for final size and peak
   height narrower than a factor 2 (total log-width < 1.39) from 100 posts up.
2. Pre-peak window (0.7x peak time): the initial growth rate and log initial post rate stay
   identified (interval width shrinks roughly as 1/sqrt(count)); final size, peak height,
   R0, gamma are NOT identified at any count up to 3000: 90% interval for final size wider
   than a factor 10 end-to-end (log-width > 2.3). Peak time in between.
   Falsifier: final-size log-width < 1.39 (factor 2) at 0.7x peak with 3000 posts. Threshold
   reason: a factor-2 forecast of total size would be a useful prediction target; a
   factor-10 one is not better than knowing the order of magnitude of a typical meme.
3. Window ending at the peak: intermediate; I expect final size identified to within about
   a factor 3 at 1000+ posts but do not hold this firmly.
4. Laplace intervals: within 4.5 pp of nominal wherever the posterior is narrow (full window,
   >= 300 posts); under-cover and/or have non-positive-definite Hessians in >10% of
   replicates for pre-peak windows at any count and for full windows below ~30 posts.
   Falsifier of the first pass's "low counts break Laplace" framing: Laplace coverage for
   curve summaries within 4.5 pp of 90% at 30-100 posts on full windows.
5. Full posterior: coverage within 4.5 pp of 90% for well-identified targets in all cells;
   for unidentified targets (claim 2) coverage is prior-dependent and may be over or under
   90%; I expect over-coverage (wide prior-dominated intervals) more often than under.
   If the posterior UNDER-covers (< 85.5%) for a target, that target's posterior interval
   cannot be used as an honest flag in that regime.
6. Shapes D and E (no growth phase): only the initial post rate s*I0 and the decay
   rate are identified; beta, gamma, R0, s, I0 individually are not, at any count. Final
   size identified (it is nearly the observed total). Peak time is degenerate (0).

What the metrics cannot see: coverage is blind to uselessly wide intervals, so every
coverage number is paired with the median interval width; coverage averaged over
replicates can hide conditional miscalibration (e.g. only the low-realised-count
replicates failing); fixed-truth coverage of a Bayesian interval depends on the prior box,
which was inherited from the first pass, not derived from real memes; everything here is a
Poisson world with an exactly-SIR mean, so it bounds identifiability from above; real data
can only be worse (E3 probes one direction of that).

Flag rules to evaluate (fixed now, before seeing E1): (F1) Laplace usable iff Hessian
positive definite, no parameter on a bound, optimiser converged; (F2) posterior probability
that the peak lies inside the observed window > 0.9; (F3) the posterior 90% interval width
of the target itself below a stated width. For each: coverage and median absolute error
among passing fits, and pass rate.

### E1 numerical note (written while E1 was running, before looking at any coverage output)

The E1 logs show the importance sampler's effective sample size (ESS) falling to a median
of ~90 for the pre-peak, 3000-post cell of shape A (proposal "v1": Gaussians in the fit
coordinates). That is a numerical-accuracy problem of my posterior computation, not a
finding. I added proposal "v2" (extra Gaussian components in (growth rate, log gamma,
logit I0), in which truncated-window posteriors are flat slabs); on a 100-replicate smoke
test of that cell it raised median ESS from 78 to 570 with interval widths changed by <10%.
Plan fixed now: after E1 completes, every E1 cell in which more than 10% of replicates have
ESS < 200 is rerun with v2 and a 120k cloud into `e1-rerun-v2/` (same seeds for the data),
and the rerun replaces the v1 result in the summary; v1-vs-v2 differences are reported as the
numerical error bar of the posterior. E2 and E3 use v2 from the start.
(Smoke-test disclosure: that smoke run showed, for the pre-peak 3000-post cell, final-size
90% width ~5.7 log units and growth-rate coverage 0.89; so claim 2 is no longer blind for
shape A.)

## E0. Re-examination of two first-pass claims (run dir `e0-first-pass-recheck/`)

(a) Claim under test: "12 of 81 grid fits had non-positive Hessians" as a property of the
data. The first-pass optimiser was started at the true parameters and its Hessian evaluated
wherever L-BFGS-B stopped. I refit those 12 datasets (regenerated from the recorded seeds)
with the damped Fisher-scoring fitter from a data-scored start. Expectation: in most of the
12 (>= 8) a lower negative log-likelihood is found or the point moves to a bound, i.e. the
"non-positive Hessian" was an unconverged optimiser or a boundary optimum, not an interior
maximum-likelihood point (an interior maximum cannot have an indefinite Hessian). Falsifier:
NLL improvement < 0.01 nats and still indefinite Hessian at an interior point in most cases
(would mean my reading is wrong or autodiff Hessians are numerically unreliable there).
Blind spot: says nothing about whether those datasets are identifiable; only whether the
diagnostic meant what the report said it meant.

(b) Claim under test (the centrepiece example of the accepted decision): for the
near-threshold dataset "case-052" (beta = gamma = 1, 577 posts) "the posterior moves to
very different natural parameters while preserving a good curve fit". That rested on a NUTS
run with R-hat 1.17, i.e. not converged. I compute the posterior of that exact dataset by
importance sampling (600k points, 60% prior-wide) and report the posterior mass in the
region NUTS visited (I0 > 0.1) versus near the truth, and the maximised log-likelihood in
each region. Expectation (weakly held, 60/40): the high-I0 region does hold most of the
flat-prior posterior mass because of its much larger prior volume, while its best
log-likelihood is no better than the truth region's. Either outcome is informative: if the
truth region holds > 90% of the mass the first-pass example was a sampler artefact; if the
high-I0 region dominates, the example stands but is a statement about the flat prior on
logit I0 allowing "almost everyone already infected at t=0" solutions.
Blind spot: with a 577-post single dataset this is one example, not a rate.

## E2. Number of bins (run dir `e2-bins/`)

Design: shapes A and C; windows "to the peak" and "full"; 100 and 1000 expected observed
posts; the same window cut into 4, 8, 16 or the native number of daily bins (A: 16/44,
C: 18/106). R = 400, seed base 20261128, proposal v2.
Expectation: binning is a minor factor until bins stop resolving the rise. Going from native
to 8 bins widens posterior 90% intervals of final size, peak height, growth rate by < 25%;
4 bins (as many bins as parameters) widens them by > 2x and breaks Laplace (non-PD or
under-coverage). Coverage of the posterior stays within 4.5 pp of 90% for full windows.
Falsifier: > 25% widening already at 16 bins (then temporal resolution matters and the
project's binning decision needs care), or no real loss at 4 bins.
Reason for 25%: equivalent to losing about a third of the data (width ~ 1/sqrt(N)); smaller
effects are below what any downstream choice would react to.
Blind spot: all bins are equal-width and the curve is exactly SIR; real diurnal structure
and bin-edge effects are absent.

## E3. Overdispersion and repost spikes (run dir `e3-noise-spikes/`)

Design: shapes A and C; full window at 100 and 1000 expected posts, and 1.5x-peak window at
1000. Generators: Poisson (control); negative binomial with var = mu + mu^2/k, k = 5 and
k = 1; "spikes" = Poisson plus, independently in each bin with probability 5%, an extra
Poisson burst as large as the tallest bin's expected count. Four interval methods on every
dataset: Poisson Laplace; Poisson full posterior; quasi-Poisson (Poisson Laplace standard
errors times sqrt(max(Pearson chi2/dof, 1))); negative-binomial maximum likelihood with
expected-information Laplace intervals. R = 400, seed base 20261129. "Truth" for final size
and peak height is the SIR component (spike posts excluded).
Expectations:
- NB data: Poisson point estimates roughly unbiased, but Poisson intervals (Laplace and
  posterior alike) under-cover badly, worse with more posts because the variance inflation
  1 + mu/k grows with the bin mean: at 1000 posts, k = 5, shape A (tallest bin ~60) I expect
  coverage of 90% intervals for final size around 30-50%; at 100 posts around 65-80%.
  NB-fit and quasi-Poisson intervals within 4.5 pp of 90% at 1000 posts, and >= 80% at 100.
- Spikes: Poisson fit biased (final size over-estimated by roughly the spike mass: 5% of
  bins x tallest bin, i.e. tens of percent for a 44-bin curve; decay rate gamma biased low);
  Poisson coverage for final size < 50% at 1000 posts. NB fit: better than Poisson but I
  expect it NOT to reach nominal (60-85%), since NB down-weights but does not reject
  outliers. Falsifier for "NB is an adequate spike-robust fitter": NB coverage for final
  size and peak height < 85.5% at 1000 posts.
Blind spots: one spike model, one dispersion form; NB dispersion is estimated on a 42-point
grid; coverage is against the SIR-component truth, which is itself a modelling choice;
no fitter here models spikes explicitly.

### E1 deviation from plan (recorded before any coverage output was inspected)

After 37 of 84 cells the v1 proposal was clearly inadequate for shape B as well (median ESS
~100-200 in several cells), so finishing with v1 and then rerunning would have wasted about
half of the compute budget. I stopped E1 and ran the remaining 47 cells with proposal v2 and
a 120k cloud directly into the same directory; each such npz records `proposal` and
`n_cloud` (the 37 v1 files do not carry these fields). Simulated datasets are unchanged by
this (data are drawn before the cloud from the same per-cell seed). The rerun rule for
already-finished v1 cells stands (>10% of replicates with ESS < 200 -> rerun with v2/120k
into `e1-rerun-v2/`).

### E1b. Optimiser-convergence check (run dir `e1-refit-check/`), written after seeing the first 37 E1 cells

Observed in E1 (A-shape cells): for windows ending at or before the peak, the 40-iteration
damped scoring fit had a Newton decrement below 1e-4 in only 2-50% of replicates, and
20-65% of fits sat on a parameter bound. So the E1 "Laplace" numbers in those cells describe
Laplace intervals at a not-fully-converged point. Claim to test: the poor Laplace coverage
there (e.g. 57-73% for beta, s, I0, final size at the peak window with 1000-3000 posts) is
a property of the likelihood ridge, not of stopping early. Test: 400 further iterations on
cells 5, 8, 10, 11 (shape A; pre-peak 3000 posts; peak window 100, 1000, 3000 posts).
Expectation: negative log-likelihood improves by < 0.5 nats for 90% of replicates (a ridge
is flat), more fits end on a bound, and Laplace coverage for the affected targets stays
below 85.5%. Falsifier: coverage returns to within 4.5 pp of 90% after convergence; then
the E1 Laplace verdict for truncated windows was an optimiser artefact and must be withdrawn.
Blind spot: 4 cells of one shape; a different optimiser could still behave differently.

### Scope reduction for E2, E3, E1b (written before any of them ran)

The machine is heavily shared today (load average ~20 on 16 cores) and E1 cells are taking
~3x longer than timed. To stay near the two-hour budget:
- E2 (bins): shape A at the peak window and the full window, 100 and 1000 posts, and shape C
  full window 1000 posts; 4, 8, 16 bins; the native daily-bin reference is the matching E1
  cell. 15 cells, R = 300 (binomial s.e. of a 90% coverage: 1.7 pp; real-deviation line
  5.2 pp, i.e. below 84.8%).
- E3 (noise/spikes): shape A full window at 100 and 1000 posts with all four generators;
  shape C full window 1000 posts with NB k=5 and spikes. 10 cells, R = 300. The truncated
  window x overdispersion cells are dropped (not run).
- E1b (optimiser check): cells 5 and 10 only, first 150 replicates, 300 extra iterations
  (s.e. of a coverage near 70%: 3.7 pp).
Expectations for the retained cells are unchanged from the sections above.

---

## Addendum by the resuming agent (2026-11-27, after the machine restart; written before I looked at any E1/E2/E3 coverage output)

State found on disk: 64 of 84 E1 cells complete and loadable (37 with proposal v1 / 60k cloud,
27 with v2 / 120k); cells 56, 60, 61, 64, 65, 68-70, 72-83 missing; no npz was truncated (all
keys of every file load). E0 (a) and (b) numbers were written; the E0 plot was not. No E1b,
E2 or E3 output exists. I checked `calib.derived()` (peak time, peak height, final size)
against a dense ODE solve for seven parameter sets: agreement to the 0.01-day grid and to
1e-5 in log units (`e1-checks/derived_vs_ode.txt`).

Resumed runs: missing E1 cells with v2/120k into `e1-window-counts/` (same per-cell seeds), and
the pre-registered rerun of the 14 v1 cells with > 10% of replicates at ESS < 200 (cells 5, 10,
11, 15, 16, 20, 21, 28, 29, 33, 34, 35, 38, 39) into `e1-rerun-v2/`.

Soundness review of the criteria above (I keep them, with these readings):
- Claim 1 says "narrower than a factor 2 (total log-width < 1.39)". 1.39 = ln 4, i.e. factor 2
  either way of the centre (factor 4 end to end). I use the number (1.39) as the criterion and
  describe it as "factor 2 either way". Claim 2's 2.3 is factor 10 end to end, as written.
- Multiple comparisons: about 10 targets x 84 cells x 2 methods are each tested at 3 s.e.;
  at nominal coverage roughly 0.3% of tests (about 5 of ~1700) will fall outside the band by
  chance. Isolated single-cell deviations are therefore not read as findings; patterns across
  neighbouring counts/windows are.
- The criteria treat over-coverage and under-coverage alike ("within 4.5 pp"). For the
  practical question only under-coverage is harmful (an over-wide interval fails safe). I
  report both, but the "misleading" verdict is under-coverage (< 85.5%) only, as in plot_e1.
- Posterior "coverage" here is frequentist coverage at a fixed truth, which a Bayesian interval
  is not guaranteed to achieve where the prior matters. For targets the data do not identify,
  the number depends on where the truth sits inside the inherited prior box; such coverage is
  not evidence that the target is usable, and the width criterion decides those.

Added checks (fixed now):
- Conditional calibration (named blind spot): within each cell, coverage among the third of
  replicates with the lowest realised count. Expectation: no material difference from the
  cell-level coverage for the full posterior (< 5 pp); Laplace worse in the low tercile.
- Flag evaluation (F1-F3 above), on the merged E1 results: for each flag, pass rate per cell
  and coverage / median |error| among passing fits. A flag is "adequate" if, pooled over cells,
  the passing fits' 90% intervals cover >= 85.5% for every curve-summary target it is used for.
  Expectation: F1 (Laplace health) is NOT adequate on truncated windows (passes fits that
  under-cover), F2 (peak observed with posterior probability > 0.9) separates trustworthy
  from untrustworthy final size / peak height, and F3 (own posterior width) is adequate by
  construction wherever the posterior itself covers.
- Practical lookup: every E1 cell re-expressed as (total posts the whole curve would have,
  fraction of those observed); verdict per target as in plot_e1 (covers >= 85.5% AND median
  90% width <= 1.39 log units for log targets, <= |true value| for growth rate, <= true peak
  time for peak time).

### E1 numerical deviation 2 (resuming agent; written before looking at any coverage of the cells named)

The v2 rerun did not fix the importance sampler everywhere: median ESS stayed at 85 (cell 34,
shape B, peak window, 1000 posts), 23 (cell 35, 3000 posts) and 179 (cell 29, B pre-peak 3000).
Disclosure: I have seen the v1 (inadequate) coverage numbers for these cells in the
partial summary (e.g. final size ~0.6-0.7 for cells 34/35), but not the v2 ones.
Plan: every E1 cell whose current posterior (v2 or v2-rerun) still has > 10% of replicates at
ESS < 200 is recomputed with an adaptive proposal "v3" (`experiments/wp02/run_e1_v3.py`,
output `e1-rerun-v3/`): stage 1 = the identical v2 cloud; stage 2 = a 120k cloud from a mixture
of Gaussians (one per replicate, from its stage-1 posterior, sd x2) in (growth rate, log gamma,
logit I0) plus 20% uniform-prior points. The target posterior is unchanged.
Validation (must pass before v3 replaces anything): on a control cell where v2 is adequate
(cell 33: median ESS 594), v3 and v2 per-replicate 5%/50%/95% quantiles of the curve summaries
must agree to within 10% of the v2 90%-interval width in median over replicates, and coverage
must agree within 2 binomial s.e. If not, v3 is not used and those cells are reported as
"posterior not computed reliably". Reason for 10%: smaller than the cell-to-cell width
differences any verdict depends on (verdict thresholds are factor-level).
Expectation for the hard cells: if their under-coverage is a sampler artefact, v3 coverage moves
towards 90%; if it is the posterior itself (prior-dominated ridge), it stays below 85.5%. I do not
hold a strong prior; 50/50.

### E1c. Prior-box sensitivity of truncated-window posteriors (run dir `e1-prior-box-check/`; written before running it)

Seen so far (partial E1, posterior): for windows that end before or at the peak, some
"trustworthy" verdicts appear for peak time / peak height (e.g. shape C, 0.7 x peak, 1000-3000
posts: peak-time 90% width ~7-11 days against a true 18 days). Cause hypothesis: before the
peak the data fix the growth rate and s*I0 but not I0 itself; peak time ~ ln(1/I0)/growth, so the
interval is set by the prior's lower bound on I0 (logit I0 >= -14, I0 >= 8e-7), an arbitrary
choice inherited from the first pass. Test: recompute the posterior (v3 proposal, same data) for
cells 4 (A, 0.7 x peak, 1000 posts), 10 (A, at peak, 1000), 53 (C, 0.7 x peak, 3000) and control
22 (A, full curve, 1000) with the I0 lower bound at logit -14 (reference) and at logit -20
(I0 >= 2e-9). Prediction: in cells 4 and 53 the median 90% width of peak time, peak height and
final size grows by > 30% and their intervals move; in control 22 widths change < 5%. Cell 10
in between. Falsifier: < 10% width change in cells 4 and 53 (then the data, not the prior bound,
constrain those quantities). 30% reason: a third wider is a change in what one would call
"trustworthy" under the factor-2 / +-50% width rule; 5% is Monte Carlo noise level seen in the
v2/v3 control. Blind spot: one alternative bound in one direction; a real hierarchical prior on
I0 would behave differently from either box.

### Second machine restart (found 21:07; last writes 18:11) and further scope reduction

Lost in flight: E1 cell 81, E3 cells 8-9, prior-box check cells 10 and 22, and all of E2 (not
yet started). All npz files on disk load completely (`experiments/wp02/check_npz.py`). Writers
now save to `*.partial.npz` and rename, so a crash can no longer leave a truncated result.
E2 is cut further (written before any E2 run): shape A only (cells 0-11: at-the-peak and full
windows x 100 and 1000 posts x 4/8/16 bins); shape C full 1000 (cells 12-14) dropped.
Expectations unchanged.

Note (after running v3 on the subcritical cells 77 and 81, before looking at their coverage): v3
made the sampler worse there (cell 77 median ESS 253 -> 69; cell 81 532 -> 444), presumably
because their posteriors are curved ridges a per-replicate Gaussian cannot follow. Rule applied:
v3 replaces the v2 posterior only where it lowers the share of replicates with ESS < 200
(true for 11, 29, 33, 34, 35). Cells 76, 77, 81, 82 keep v2 and are reported as numerically
weaker (12-38% of replicates with ESS < 200); v3 for 76 and 82 was stopped unfinished.
