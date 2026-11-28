# WP-12 expectations log, November 2026 session

Each entry is written BEFORE the run it describes. Outcomes are appended under
"Observed" afterwards and never edited into the expectation text.
Notation: HawkesN intensity  lambda(t) = (1 - N_t/N) * K*theta * sum_i exp(-theta (t - t_i)),
no background rate (mu = 0). K = expected direct offspring per post in an untouched
population (the SIR R0), theta = decay rate of a post's influence (SIR gamma),
N = population size, I0 = number of seed events placed at t = 0.

---

## E1. The likelihood code computes the stated HawkesN likelihood  (identity)

**Claim.** `hawkesn.hawkesn_log_likelihood` (JAX, O(n) scans) equals the HawkesN
log-likelihood. Tested against two implementations that share no code with it:
(a) a brute-force O(n^2) numpy evaluation with the compensator done by numerical
quadrature of the intensity (scipy `quad` per inter-event interval), and
(b) a closed-form "sufficient statistics" evaluation written for the profile work
(`exact_profile.py`).
**Expectation.** Agreement to floating-point / quadrature tolerance: relative
difference < 1e-8 on 20 random (cascade, parameter) pairs, mu = 0, I0 in {1, 30}.
**Falsifier.** Any relative difference > 1e-6. Why this number: quadrature is run at
epsrel 1e-10 and the sums are of O(100-1000) terms in float64, so honest agreement is
1e-9 or better; 1e-6 is three orders above that and far below any indexing error
(an off-by-one in the depletion count changes the log-likelihood by O(n/N) ~ 1e-1).
**Separately expected defect (from reading the code, before running):** for mu > 0 the
background compensator uses depletion 1 - j/N for the interval after the j-th event
where it should be 1 - (j+1)/N. I expect the mu>0 comparison to FAIL by about
mu*T/N in absolute log-likelihood. This does not affect mu = 0 fits.
**Blind spot.** An identity check on the likelihood says nothing about whether the
simulator draws from that same model (that is E2), nor whether the model in the code is
the one in the paper (checked by reading the upstream R code, and by E7 if R runs).

## E2. The simulator draws from the model the likelihood describes (distributional)

**Claim.** Event streams from `simulate_hawkesn` (thinning) follow the HawkesN law.
**Test.** Time-rescaling theorem: under the true parameters the compensator increments
between consecutive events are i.i.d. Exp(1). Pool increments from 200 cascades
(K=2, theta=0.5, N=500, I0=10, T=40; seed 20261101), KS test against Exp(1). Second
check: an independently written inversion sampler (exact inverse of the compensator,
no thinning) must give the same final-size distribution (two-sample KS on 2000 runs
each, I0=1, seeds 20261102/3).
**Expectation.** KS p-values not small (> 0.01) in both; with ~50k pooled increments the
KS statistic should be < 0.008 (the 1% critical value is 1.63/sqrt(n) = 0.0073 for
n = 50k), which would detect a 1% mis-scaling of the intensity.
**Falsifier.** KS statistic above the 1% critical value for the realised n.
**Blind spot.** The censored last interval of each cascade is dropped, so an error
that only affects behaviour after the last event (e.g. wrong extinction handling) is
invisible to the rescaling test; the final-size comparison covers that.

## E3. Is the old profile-likelihood code computing a profile at all?  (estimator behaviour / code audit)

**Claim under test.** The June identifiability report's "flat K profile" came from
`identifiability_study.compute_profile_K`. Reading it, the inner optimiser is handed a
3-component gradient for a 2-parameter problem (`jac=True` with the full objective),
and exceptions are swallowed. A profile likelihood is a maximum over nuisance
parameters, so it must be continuous in K and, on the same data, can never exceed the
global maximum nor dip below the likelihood at any fixed nuisance value.
**Expectation.** On the June single-cascade data (seed 7777 scheme), the old routine's
profile will differ from an optimiser-free dense-grid profile by more than 1
log-likelihood unit at many K values (the old plot is jagged with jumps of 2-15 units,
which a true profile cannot be). If so, the June profile figure and every CI width
derived from it are void, whatever the right answer turns out to be.
**Falsifier.** Old and grid profiles agree within 0.1 log-lik units everywhere: then
the jaggedness is real structure and I am wrong about the bug.
**Why 0.1 / 1.** 1.92 log-lik units is the whole 95% interval threshold; an error of 1
moves interval endpoints arbitrarily; 0.1 is the grid profile's own resolution error
(measured by halving the grid step).
**Blind spot.** Agreement on one data set would not certify the old code in general.

## E4. Single-cascade identifiability: ridge, fitter, or window?  (estimator behaviour)

**Setup.** Exact likelihood on a dense grid, no optimiser in the loop. For fixed
(theta, N) the maximising K is available in closed form, so the profile surface over
(K, theta) and 1-D profiles of K, theta, N, beta = K*theta and r = theta*(K-1)
(early exponential growth rate) are exact up to grid step. 95% profile intervals use
the chi-square(1) cut-off 1.92.
Scenarios (all mu = 0, I0 known):
 P  paper tutorial: K=5, theta=0.2, N=1300, I0=300, T=11
 M1 meme-like small:  K=1.7, theta=0.4, N=110,  I0=1, complete cascade (T=30/theta), kept only if it takes off (>= 20 events)
 M2 meme-like medium: K=1.7, theta=0.4, N=1000, I0=1, complete
 M3 meme-like, strong: K=3,  theta=0.4, N=1000, I0=1, complete
 W  window series on one M2 and one M3 cascade: T cut where 10%, 25%, 50%, 75%, 100% of its events have occurred, and 3x beyond the last event.
**The three explanations predict different pictures:**
 - *Fitter pathology only*: grid profile has a clear interior maximum with a bounded
   95% interval for K and theta; `fit_hawkesn` lands elsewhere (log-lik gap > 0.1 below
   the grid maximum, or at the K=20 bound).
 - *Genuine ridge*: grid profile itself is flat (95% interval for K spans a factor > 3
   or is open-ended) while some combination has a tight interval; fitter and grid agree
   on the maximum value.
 - *Window artefact*: on the same cascade the K and theta intervals are open-ended for
   short windows and close once the decline of the cascade is inside the window.
**What I expect.** A mixture, in this order of importance:
 1. Complete cascades with hundreds of events (M2, M3, P): bounded intervals for all of
    K, theta, N; relative 95% half-widths roughly 20-50% for K and theta, a few % for N
    in M3/P (most of the population is used up) and wide for N in M2 (final size only
    ~70% of N, and N trades off against K through the final-size relation).
    beta = K*theta tighter than either factor in P (large I0, early phase dominated by
    seeds, intensity scale directly visible). For I0=1 cascades I expect the growth rate
    r = theta(K-1) to be at least as tight as beta, because the early phase shows
    exponential growth at rate r; I do not know which of the two wins and will report
    both with equal standing.
 2. Truncated windows (10-25% of events seen): intervals for N open to the right
    (no depletion visible yet) and K, theta intervals several-fold wider; r stays tight.
 3. M1 (about 60-80 events): intervals a factor of several wide; point estimates of K
    not usable.
 4. The existing fitter: reaches the grid maximum in most complete-cascade cases
    (gap < 0.1) but fails in a noticeable minority because of its start point (K=0.8,
    theta = 1/median gap) and only 3 restarts; when the true maximum is at K > 20 it pins.
**Falsifiers.** (a) If M2/M3/P complete-cascade K intervals are open-ended on the grid:
my expectation of practical identifiability is wrong and the ridge is genuine even with
full observation. (b) If the fitter's log-lik is within 0.1 of the grid maximum in
>= 95% of replicates across scenarios, the "fitter pathology" component is absent.
(c) If window length does not change interval width by at least a factor 2 between the
25% and 100% cuts, the window explanation is out.
**Units/thresholds.** Log-likelihood gap 0.1: an optimiser that stops 0.1 below the
maximum sits well inside the 95% region (1.92), so smaller gaps cannot matter for
reported intervals; "factor > 3 or open" for a flat profile because a parameter known
only to within a factor 3 cannot serve as a regression target.
**Blind spot.** One example cascade per scenario shows the shape for that draw only;
E5 covers sampling variability. Grid profiles are exact only inside the grid box
(theta within 1e-3..1e3 x truth, N up to 1000 x the observed count); "open-ended"
means "still inside the 95% region at the box edge".

## E5. Replicated recovery and interval coverage  (estimator behaviour)

**Claim.** For scenarios P, M2, M3 (and M1 for contrast), over 200 independent
cascades each (seeds listed in the script), the maximum-likelihood estimates of the
reportable quantities are approximately unbiased and their 95% profile-likelihood
intervals cover the truth about 95% of the time.
**Expectation.** Coverage between 91% and 98% (binomial 95% range for 200 replicates
around a nominal 95% is 92-98%; I allow one point for grid error) for K, theta, N,
beta, r in P, M2, M3. Median relative error of K within +-10% of truth; N biased low
by O(1/N) effects only. M1: coverage may be off (small-sample asymptotics) and I
expect upward-skewed K. For the Hessian-based covariance returned by `fit_hawkesn`
I expect under-coverage (< 90%) for K and theta wherever the profile is asymmetric.
**Falsifier.** Profile coverage below 88% or above 99.5% for any of K, theta, N in P,
M2, M3 (outside what binomial noise at n=200 can produce, p < 0.001), or median K
off by more than 20%.
**Blind spot.** Coverage averaged over replicates can hide a subset of cascades (e.g.
those that barely took off) with terrible intervals; I will plot interval width
against event count. All of this is under the correct model with known I0 and
mu = 0; real memes violate each of those.

### Observed, E1-E3 (run 2026-11-27, `experiments/wp12/checks_2026_11.py`, results `nov_checks_results.json`)

- **E1 held.** mu = 0: max relative difference JAX vs brute-force quadrature 6e-15, sufficient-statistics
  form vs brute force 6e-15 (falsifier 1e-6). The predicted mu > 0 defect is real and exactly as
  predicted: JAX minus brute force = -mu*T/N to 4 decimals in 6/6 cases (e.g. -0.100, -0.256).
  Not fixed (hawkesn.py left as is; it does not touch mu = 0 fits). Anyone fitting with a background rate must fix it first.
- **E2: passed, but only just, and for a reason I should have foreseen.** Rescaling KS = 0.0056 against a 1%
  critical value 0.0058 (n = 78,154), p = 0.015; mean rescaled increment 0.993 instead of 1.
  Two samplers: two-sample KS p = 0.79; small-cascade fraction 0.202 vs 0.209; mean large size 237.7 vs 237.5.
  The 0.7% shortfall in the mean has a known cause that my test ignored: a waiting time is only
  recorded if an event happens before T, which truncates the increment at the remaining compensator
  mass (d*K*A*(1-exp(-theta*(T-t_prev)))). That truncation is negligible mid-cascade and matters for the last
  few events. A borderline pass with an unmodelled bias in the direction of the deviation is not a clean pass -> E2b.
- **E3 held.** The June routine differs from the exact grid profile by up to 14.2 (4-event cascade) and
  18.3 (55-event cascade) log-likelihood units, and by more than 1 unit at 54% / 69% of K grid points; it is
  never above the exact profile (as a correct-but-badly-optimised maximum must be) and is jagged. Plot
  `nov_E3_old_profile_audit.png`. Cause, from the code: `nll_sub` passes the 3-component gradient of the full
  objective to a 2-parameter L-BFGS-B problem. Consequence: `part1_profiles.png`, the Part-1 and Part-3
  CI-width tables and the "gate passed" statement of the June identifiability report are void. Also, the
  June single-cascade "flat profile" gate was evaluated on a cascade with 4 events.
  Unplanned observation: the exact K profile on the 55-event cascade rises monotonically to the K = 20 edge.
  The best explanation of that cascade under the model is "population ~ observed count, large K, slow decay".
  This is a K-N trade-off, not only the K-theta one the June report discusses. Followed up in E4.

## E2b. Time-rescaling with the truncation handled exactly  (distributional; refinement of E2, written after seeing E2)

**Claim.** Conditional on the next event occurring before T, x = compensator increment has the Exp(1) law
truncated at c = d*K*A*(1 - exp(-theta*(T - t_prev))), so u = (1 - exp(-x)) / (1 - exp(-c)) is exactly
Uniform(0,1) for every event. Same 200 cascades (seed 20261101).
**Expectation.** KS against Uniform not significant (p > 0.01), mean of u = 0.500 +- 3 standard errors
(se = 0.289/sqrt(n) ~ 0.001).
**Falsifier.** KS above the 1% critical value 1.63/sqrt(n), or |mean - 0.5| > 0.003.
**Blind spot.** This is a post-hoc refinement, chosen by derivation and not by trying variants; it still
cannot see errors in what happens after the last event (covered by the two-sampler size comparison).

### Observed, E2b (run 2026-11-27, `checks_2026_11.py E2b`, `nov_checks_E2b_results.json`)
Held: KS = 0.0041 (1% critical 0.0058), p = 0.14, mean u = 0.4994 +- 0.0010. With the truncation handled the
simulator and likelihood agree; the E2 near-miss was the test's fault, not the simulator's.

### Defect in my own new code, found on the first E4 run and fixed before any E4 result was recorded
The first run of the exact-grid profiles put the global maximum for the complete single-seed cascades at
"N just below the observed count, K in the hundreds, theta at the grid edge", 6-23 log-likelihood units above
anything near the truth. Cause: my N grid started at (observed count - 1), and for N below the observed
count the depletion factor after the last event is negative, which my closed form turned into a negative
compensator. hawkesn.py clips that factor at zero, so it never had this problem. E1 did not catch it
because E1 only tested N >= 1.05 x the observed count. Fix: the grid now starts at N = observed count
(`exact_profile.default_N_grid`, `grid_fit`). **Consequence for E3:** the E3 "exact" curve used a grid that
included that invalid strip, so its monotone rise towards K = 20 may be this artefact. The comparison
"June routine is jagged and far below a smooth upper envelope" is unaffected in kind, but the numbers
and the unplanned K-N observation are withdrawn until E3 is re-run with the fixed grid (below, E3r).
E1 is extended with points at N = observed count and just above (E1b): same falsifier (relative
difference > 1e-6 between the sufficient-statistics form and the JAX code).

### Observed, E1b and E3r (2026-11-27)
- E1b held: sufficient-statistics form vs JAX at N = observed count (+0, +1e-3, +0.5): max relative difference 3e-15.
- E3r (re-run of E3 with the fixed grid; `nov_checks_E3r_results.json`, `nov_E3r_old_profile_audit.png`):
  June routine vs exact profile differ by up to 5.5 (4-event cascade) and 14.1 (55-event cascade) log-lik
  units, more than 1 unit at 54% / 59% of K values. E3's conclusion stands with these numbers. The first-run
  file `nov_E3_old_profile_audit.png` is superseded. (One point of the June routine sits 0.68 above the
  grid's global maximum on the 55-event cascade: it used N just below the observed count, the strip my grid now excludes.)

### Observed, E4 examples (2026-11-27, `identifiability_2026_11.py examples`; `nov_ident_examples.json`, `nov_ident_example_{P,M1,M2,M3}.png`)
One cascade per scenario (replicate index 0). 95% profile intervals, exact grid:

| scenario | events | K (true) | theta (true) | N (true) | K*theta (true) | theta(K-1) (true) | global max at | existing fitter vs grid max |
|---|---|---|---|---|---|---|---|---|
| P  | 982 | 4.3-17.1 (5) | 0.050-0.247 (0.2) | 1282-1307 (1300) | 0.85-1.09 (1.0) | 0.77-0.89 (0.8) | K=7.5, theta=0.127, N=1287 | same point, gap 0.000 |
| M1 | 65  | 5.4-628 (1.7) | <0.0004-0.051 (0.4) | 66-66.5 (110) | 0.15-0.30 (0.68) | 0.15-0.26 (0.28) | K=498, theta at grid edge, N=66=observed count | pinned at K=20, gap 0.36 |
| M2 | 678 | 12-675 (1.7) | <0.0004-0.024 (0.4) | 679-679.4 (1000) | 0.23-0.30 (0.68) | 0.233-0.276 (0.28) | K=627, theta at grid edge, N=679=observed count | K=1.73, theta=0.39, N=960 (near truth), 2.65 BELOW the global max |
| M3 | 951 | 1.8-11.6 (3) | 0.065-0.93 (0.4) | 952-1314 (1000) | 0.74-1.70 (1.2) | 0.65-0.89 (0.8) | K=2.40, theta=0.55, N=1084 | same point, gap -0.004 |

Against the expectation:
- **Wrong (falsifier (a) triggered):** I expected bounded, 20-50% intervals for K and theta on complete
  cascades with hundreds of events. In M2 the exact likelihood has two lobes along the curve
  theta*(K-1) = const (plot, middle panel): one at the truth and one running off to K -> infinity,
  theta -> 0 with N equal to the observed post count ("everyone who could post did, and posts never stop
  being infectious"). In this cascade the second lobe is the HIGHER one by 2.65 log-lik units, so the 95%
  intervals for K, theta, N and K*theta all exclude the truth. In P and M3 there is one lobe but K is known only to a
  factor 4-6.
- **Right:** the growth rate theta*(K-1) is the tight direction in every scenario (+-8% in P, +-9% in M2,
  +-16% in M3) and K*theta is not: in the second lobe K*theta is 0.25 against a true 0.68. The June
  report's "K*theta is the identifiable combination" does not hold for single-seed cascades.
- **Fitter:** the existing fitter matched the exact maximum to < 0.01 in P and M3. In M2 it found the lobe
  at the truth (a local maximum) and in M1 it ran into its K = 20 bound in the second lobe. So the
  K = 20 pinning in the June recovery table is the second lobe of the likelihood, cut off by the
  bound. It is neither an optimiser failure nor (these cascades are complete) a window effect.
These are four single cascades; how often each lobe wins is E5's job.

## E5-add. Additions to E5, decided after seeing the E4 examples and before running E5

E5 as registered stays the primary analysis (global maximum, 95% profile intervals). Because E4 showed
a second lobe at implausible parameter values, I add, with equal standing and without replacing anything:
 (i) for each replicate, which lobe holds the global maximum (classified by K_hat > 20, the existing
     fitter's own upper bound, chosen because it is already in the code and not tuned by me), and the
     log-lik difference between the best point with K <= 20 and the global maximum;
 (ii) coverage of intervals built relative to the K <= 20 constrained maximum (i.e. what a user of the
     existing fitter's parameter box would report), and the same for K <= 10 to show sensitivity to the bound;
 (iii) recovery of the growth rate r and of K*theta from the K <= 20 fit.
**Expectation.** The second lobe holds the global maximum in a substantial minority (my guess 20-50%) of M1
and M2 cascades, rarely (< 10%) in M3 and P where most of the population is used up... (no: in M3/P the
observed count is close to N so the two lobes merge; I expect no separate lobe there). With the registered
global-maximum intervals I now expect E5's coverage falsifier to FIRE for K, theta, N in M1 and M2
(coverage well below 88%), and coverage of r to stay within 88-99.5% in all four scenarios. Within the
K <= 20 box I expect coverage of K, theta, N to come back above 88% in M2 but stay poor in M1 (the box
maximum sits at the bound).
**Falsifier for the two-lobe account.** If coverage of r is also below 88% somewhere, then "growth rate is
the safe quantity" is wrong. If constrained-box coverage of K in M2 stays below 80%, the truth-side lobe is
itself biased and a bound on K does not rescue the natural parameters.
**Blind spot.** Box-constrained intervals are not likelihood-ratio intervals with a known reference
distribution; their coverage is an empirical property of these four scenarios only.

### Observed, window series (part of E4; 2026-11-27, `identifiability_2026_11.py window`; `nov_ident_window.json`, `nov_ident_window_{M2,M3}.png`)
Same M2 and M3 example cascades, window ending when 10/25/50/75/100% of events have occurred, then the full
simulated horizon (T = 75; for M3 "3x the last event time" = 62.8 fits inside it, for M2 it does not, so the
full horizon is used instead; deviation from the registered design noted).
- Growth rate theta(K-1): interval narrows steadily with window length, upper/lower ratio 7.4 -> 5.1 -> 3.2 -> 2.3 -> 1.35 -> 1.19 (M2)
  and 4.0 -> 2.7 -> 1.8 -> 1.5 -> 1.34 -> 1.37 (M3), always containing the truth.
- N: no upper bound until about half the events are in the window (as expected), then bounded.
- K and theta: in M3 the upper end of K is unbounded at every window up to 75% and closes only when the
  decline is seen (K 1.8-9.5 at 100%). In M2 it never closes: the interval reaches K > 600 at 25%, 75%, 100%
  and at the full horizon the second lobe overtakes the truth-side lobe. Falsifier (c) asked whether window
  length changes interval width by a factor >= 2 between 25% and 100%: yes for r, N and (M3) K; no for K in M2.
- Oddity I cannot explain from one cascade: M2 at the 10% window (68 events) gives a bounded K interval
  (1.1-2.7) that opens up again at 25%. Not followed up.
So "window artefact" is real for N and for K in the strong-growth scenario, and does not account for the
second lobe in M2.

## E6. What objective does the upstream R code optimise?  (identity)

**Claim (from reading `scripts/functions-SIR-HawkesN.R`, before running it).** Upstream's
`neg.log.likelihood` is NOT the textbook HawkesN likelihood that `hawkesn.py` implements. Differences:
(1) the intensity at an event time includes that event itself: the kernel sum has an extra +1 (plus any
tied events) and the depletion factor uses the count including the event, 1 - i/N in place of 1 - (i-1)/N;
(2) the window ends at the last event (no censored tail); (3) parameter `c` is unused.
Its analytic gradient (`closedGradient`) differentiates the textbook form, so objective and gradient are
inconsistent with each other.
**Test.** A numpy re-implementation of "upstream as I read it" must reproduce R's `neg.log.likelihood`
on 5 tweet cascades x 3 parameter points to relative 1e-8, and the textbook likelihood (ours, T = last
event) must differ from R's by exactly sum_i [log((1-i/N)(A_i+1)) - log((1-(i-1)/N) A_i)] (i counting
seeds), where A_i is the kernel sum before event i.
**Expectation.** Both hold. The difference is of order n*log(1 + 1/A): large (tens of log-lik units)
for sparse cascades where A ~ 1, negligible for the tutorial setting (A >= ~300).
**Falsifier.** Relative mismatch > 1e-6 between R and my "as read" version: I misread the code.
**Blind spot.** Says what the code does, not what the paper intended or which is "right".

## E7. Upstream tutorial, run unmodified  (reproduction)

**Claim.** The tutorial's code cells, run verbatim under R 4.5.3 / nloptr from conda-forge (the notebook
was made with R 3.4.3) with a fixed seed, reproduce the notebook's printed tables to within sampling
variation of 20 simulated cascades, and its final-size figure qualitatively.
**Expectation.** SIR fit: median N within 1280-1300, gamma 0.19-0.21, beta 0.95-1.05 (notebook: 1283.5,
0.200, 0.986; sd 3.6, 0.006, 0.026). HawkesN fit: median K between 4 and 7, median N 1285-1300, and theta
badly scattered in some runs (notebook: K 5.5 sd 2.1, theta median 0.18 sd 8.52, N 1292 sd 11). Because
the notebook itself shows theta sd 8.5 and mean beta 6.4 +- 13.7 against a true 1.0, I expect at least one of 20
upstream fits to be far off (theta > 1) in my run too. Figure: prior with two maxima (near 0 and near
~1290), posterior after half the events with one maximum near the top.
My own HawkesN fits on the R-simulated cascades (E7b): the JAX fitter should reach an objective value
at least as good as upstream's on every cascade when both are scored by the textbook likelihood with
T = last event, and its estimates of K should have smaller spread than upstream's.
**Falsifier.** Median SIR-fit parameters outside the ranges above, or the prior not bimodal: the
environment does not reproduce upstream. For E7b: any cascade where upstream's solution has a higher
textbook likelihood than ours by > 0.1.
**Why these ranges.** About 3 standard errors of a median of 20 given the notebook's sd, widened for K.
**Blind spot.** A seed-dependent rerun cannot reproduce the notebook's exact digits; "matches" is
distributional with n = 20. R version drift could change optimiser paths without my noticing.

## E8. JAX fitter vs upstream fitter on the same tweet cascades  (estimator behaviour)

**Setup.** First 30 cascades in `Dataset/News` index order with 50-300 tweets; times in seconds; first
tweet is the single seed; magnitudes ignored (set to 1.1 as the tutorial's converter does); upstream
`fitSeries` from the tutorial's start point (K=1, theta=0.1, N=1000); JAX `fit_hawkesn(I0=1, mu_fixed=0,
T=last event)` in time units of the cascade's median gap, theta converted back to 1/s, plus the exact
grid maximum as referee.
**Expectation.** Not a clean match, for reasons established above: (a) different objectives (E6); (b)
a likelihood with a flat direction and a second lobe (E4); (c) upstream's start theta = 0.1/s is 2-4
orders of magnitude from tweet time scales and its gradient is inconsistent with its objective.
Quantitatively: the growth-rate-like combination and N agree within a factor 1.5 in at most half the
cascades; K agrees within a factor 1.5 in fewer than half. Scored by the textbook likelihood, the JAX
or grid solution is better than upstream's in >= 90% of cascades; scored by upstream's own objective,
upstream is sometimes better (because it is a different function).
**Falsifier.** If K, theta, N all agree within 10% in >= 80% of cascades, then my expectation of
disagreement is wrong and the cross-check simply passes.
**Blind spot.** Tweets are not HawkesN data; agreement or disagreement here says nothing about which
fitter is closer to any truth, only whether they solve the same optimisation problem.

### Observed, E6 (2026-11-27, `upstream_compare.py identity`; `nov_upstream/E6_identity_results.json`)
Held. R's `neg.log.likelihood` equals my numpy transcription of "upstream as read" to relative 1.7e-15 on
15 (cascade, parameter) points; R 4.5.3. The textbook negative log-likelihood is larger than upstream's by
148 to 2222 log-lik units at those (random, not fitted) points. I did not separately code the closed-form
difference I wrote down; the transcription matching to machine precision is the stronger check.
**Consequence worked out before the tutorial run (prediction for E7):** with the self-term, the intensity
at every event is at least K*theta*(1 - i/N) however fast the kernel decays, so as theta -> infinity the
sum of log-intensities grows like n*log(theta) while the compensator stays bounded by K*n. The upstream
objective is therefore unbounded below in theta. I predict that the wildly large theta values in the
notebook's table (sd 8.52) are the optimiser sliding down this slope, i.e. in my run any upstream fit
with theta > 1 will have a LOWER upstream objective than the upstream objective at the true parameters,
and evaluating upstream's objective at theta = 1e3, 1e6 (K, N at truth) will give ever lower values.
Falsifier: upstream objective at theta = 1e6 higher than at theta = 0.2 on the tutorial cascades.

## E9. The paper's headline figure  (E9a identity; E9b reproduction)

**E9a claim.** My O(N^2) recursion for the final-size distribution of the SIR jump chain reproduces the
upstream R output (1.7M-state sparse transition matrix, 2N-1 multiplications) for the tutorial's own
case (N=1300, K=5, theta=0.2; prior from one infected; posterior after half of cascade 1's events, state
estimated as upstream does). **Expectation:** max absolute difference in any probability < 1e-9 for
both curves. **Falsifier:** > 1e-6 (both are exact algorithms for the same chain in float64; errors
accumulate over ~2600 steps at 1e-16 each, so 1e-9 is generous and 1e-6 means a different chain or state).
**Blind spot:** identical output shows I transcribed upstream's method, not that the method is right (E10).
**E9b claim.** With those parameters the distribution is two-peaked before observation (a peak at
"dies immediately" and a peak near the deterministic size ~1290) and one-peaked once a modest number of
events is seen. **Expectation:** small-outcome probability (final size < 65 = 5% of N) = 1/K = 0.20 a
priori, falling below 1e-3 by the time ~20 events are seen and the estimated infectious count is >= 5
(extinction from I infectious is about (1/K)^I = 0.2^5 = 3e-4).
**Falsifier:** prior small-outcome mass outside 0.19-0.21, or posterior after 65 events still > 0.01.

## E10. Is that distribution right for the process that generated the data?  (distributional)

Stochastic SIR and HawkesN have the same expected intensity (the paper's theorem) but are different
processes: an SIR individual infects for an exponentially distributed time (offspring count geometric,
mean K); a HawkesN event has a fixed decaying kernel (offspring count Poisson, mean K). For one seed:
P(dies out) = 1/K for SIR, and the root of q = exp(-K(1-q)) for HawkesN.
**Disclosure:** the E2 sampler check already showed a small-cascade fraction of 0.20 for HawkesN at K = 2,
before I registered this. So for the prior this is a confirmation run with a formal threshold, not a blind prediction.
**Expectations (K=2, theta=0.4, N=1000, one seed, 20,000 runs each).**
 (1) Prior: SIR Markov-chain small-outcome mass 0.50 and simulated SIR 0.50 +- 0.011 (3 se); simulated
     HawkesN 0.203 +- 0.009. So the paper's method, applied to HawkesN-generated cascades, overstates the
     chance of fizzling by a factor ~2.5 at K = 2. Large-outbreak peak: same location (~797) for both;
     I expect HawkesN's peak to be somewhat narrower (less offspring variance) but have no number.
 (2) After 30 observed events of an SIR cascade: the chain started from upstream's estimated infectious
     count (the expected number still infectious given only the infection times) vs Monte-Carlo
     continuation from the cascade's true hidden state. Expect agreement only on average: per cascade the
     remaining fizzle probability differs because the true I differs from its expectation; errors of
     either sign, typically within a factor ~3 when the probability is > 1e-3.
 (3) After 30 events of a HawkesN cascade, continuation simulated exactly: I expect the chain to
     overstate the fizzle probability systematically (HawkesN extinction from kernel mass A is
     exp(-K*z*A)-like, smaller than (1/K)^I at equal A = I).
**Falsifiers.** (1) simulated HawkesN small-outcome mass within 3 se of 0.50: the processes do not differ
as claimed. (3) chain fizzle probability below the simulated one in more than 2 of 8 HawkesN cascades
(8 cascades: under "no systematic direction" 6+ of 8 one-sided has p = 0.14 each way, so this is a weak
test; I will report counts, not a verdict).
**Blind spot.** One parameter setting; thresholds for "small" are read off the gap between the peaks
(200 of 1000) and nothing between 50 and 400 would change the numbers materially, which I will check in the histogram.

## E11. Simulator-to-mean-field convergence from a single seed post  (estimator/limit behaviour)

The June convergence check held the initial infected FRACTION fixed. Real memes start from one post.
From one seed the early phase is a branching process: it dies out with fixed probability and, if it
survives, enters the deterministic regime after a RANDOM delay whose spread does not shrink with N.
So "the simulator converges to the mean-field curve" needs splitting into three separate claims.
Setup: K in {1.5, 3}, theta = 0.4, N in {250, 1000, 4000, 16000, 64000}, 400 runs per cell
(seeds SeedSequence([20261140, K index, N index])). "Take-off" = final size > half the mean-field final size.
Mean-field curve: the (s, a) ODE from `bridge_deterministic.py`'s equations, started at s = 1 - 1/N,
a = K*theta/N, solved with scipy at rtol 1e-10.
 (a) **Take-off probability** -> 1 - q with q = exp(-K(1-q)) (0.583 for K=1.5, 0.940 for K=3), NOT the SIR
     value 1 - 1/K (0.333, 0.667). Expectation: at N >= 4000, within 3 binomial se (se ~ 0.025 / 0.012).
     Falsifier: outside 3 se at N = 16000 or 64000.
 (b) **Shape, conditional on take-off, after aligning in time**: shift each run so that it crosses
     10% of N at the same moment as the mean-field curve; the sup over the rest of the run of
     |cumulative fraction - mean-field| should shrink like N^(-1/2) per run (CLT fluctuations).
     Expectation: log-log slope of the median per-run sup deviation vs N between -0.6 and -0.4, and value
     at N = 64000 below 0.02 (about 5/sqrt(N)). The aligned MEAN curve should deviate less than single
     runs and shrink at least as fast. Falsifier: slope 95% CI (bootstrap over runs) including 0 or
     excluding the whole range [-0.75, -0.25]; or median sup deviation at N = 64000 above 0.02.
 (c) **Timing does not converge**: the sd across runs of the time at which 10% of N is reached stays
     of order 1/r (r = theta(K-1): 0.2, 0.8) for all N. Expectation: sd at N = 64000 within a factor
     1.5 of sd at N = 1000, and both > 0.5/r... (I expect roughly 1.0-1.5/r from the spread of log W).
     Consequently the un-aligned mean of take-off runs does not converge to the mean-field curve:
     its sup deviation stays above 0.03 at N = 64000. Falsifier: timing sd shrinking by more than a
     factor 2 from N = 1000 to 64000, or un-aligned sup deviation below 0.01 at N = 64000.
     Also reported: mean delay relative to the mean-field curve started from 1/N (I expect the
     mean-field-from-1/N curve to be systematically early or late by a constant of order 1/r; sign not predicted).
**Blind spot.** Alignment at one level (10%) hides any disagreement in the phase before 10%; the sup
metric after alignment is zero at the alignment point by construction, so it understates early error.
Conditioning on take-off uses the final size, which is fine for a limit statement but not available in real time.

### Observed, E11 (2026-11-27, `single_seed_convergence_2026_11.py 400`; `nov_single_seed_results.json`, `nov_single_seed_shape.png`, `nov_single_seed_scaling.png`)
- (a) held. Take-off fraction at N = 16000 / 64000: K=1.5: 0.583 / 0.578 (HawkesN theory 0.583, SIR theory 0.333);
  K=3: 0.968 / 0.925 (theory 0.940, se 0.012; SIR theory 0.667). The 0.968 is +2.4 se, inside the 3-se
  falsifier but the largest deviation in the table; the other four K=3 cells are 0.925-0.930.
- (b) held. Median per-run sup deviation after alignment, N = 250 -> 64000: K=1.5: 0.112, 0.062, 0.031, 0.016, 0.0077;
  K=3: 0.083, 0.038, 0.020, 0.0095, 0.0049. Slopes -0.49 (95% CI -0.51..-0.46) and -0.51 (-0.52..-0.49);
  both values at N = 64000 below 0.02. Aligned mean curve: 0.019 -> 0.0003 and 0.015 -> 0.0002, slope -0.74.
- (c) held, with my magnitude guess low for K = 3. Sd of the time to reach 10% of N: K=1.5: 6.4, 7.4, 7.5, 9.0, 7.4
  time units (1.3-1.8 in units of 1/r); K=3: 2.5, 2.5, 2.5, 2.4, 2.4 (1.9-2.0 /r; I had guessed 1.0-1.5).
  No shrinkage with N. Un-aligned mean of take-off runs vs mean-field: sup gap 0.050 (K=1.5) and 0.147 (K=3) at
  N = 64000, not shrinking. Take-off runs reach the 10% mark later than the mean-field curve started from
  1/N by 1.3-1.6 time units (K=1.5) and 1.1-1.3 (K=3), at every N.
- Not checked: any level other than 10% for alignment; theta other than 0.4; the conditional law of the delay.

### Observed, E9b and E10 first run (2026-11-27; `size_posterior_2026_11.py figure|truth`)
- E9b held where it could be tested: prior P(final size < 65) = 0.2002 (expected 0.19-0.21); after 5 events
  (4 estimated infectious) 1.6e-3; after 20 events (16 infectious) 8.7e-12. The right-hand peak (mode 1291-1292,
  5-95% range 1285-1296) is the same before and after observation. **My second falsifier was vacuous**: once
  65 events are seen "final size < 65" is impossible by construction, so "posterior after 65 events > 0.01"
  could never fire. The figure now reports P(size < 100) (2.0e-1, 1.6e-3, 8.7e-12, 9e-34).
  `nov_size_prior_to_posterior.png` / `.json` (seed 20261130).
- E10 (1) held: before observation the paper's SIR chain gives P(small) = 0.502; simulated SIR 0.505 +- 0.0035;
  simulated HawkesN 0.2033 (theory 0.2032). Large-outbreak peak: mean 795 (SIR) vs 796 (HawkesN); sd 29.3 vs
  21.7, so HawkesN's peak is 26% narrower (direction predicted, size not).
- E10 (2)/(3) **not testable as I designed them**: after 30 observed events the estimated infectious count is
  10-21 and the fizzle probability is 1e-6..1e-3, far below what 4000 continuation runs resolve (0 hits in 15 of
  16 cascades). What the run does show: for SIR-generated cascades the chain's 5/50/95% final-size
  quantiles match the simulated continuation within 0-5 posts (e.g. 746/798/841 vs 746/798/841); for
  HawkesN-generated cascades the medians match (798 vs 798) but the chain's 90% range is about 95 wide
  against a simulated 69 (sd 29 vs 21).

## E10b. Fizzle probability after FIVE observed events  (distributional; redesign of E10 (2)/(3))
Same setting (K=2, theta=0.4, N=1000), 8 SIR and 8 HawkesN cascades, each cut after 5 events (seed + 4
infections), 4000 continuation runs each from the true state (seeds 20261135 prefixes, 20261136 continuation).
**Expectation.** SIR data: chain (from the rounded expected infectious count) vs simulation from the true
hidden count differ per cascade by up to a factor of a few in either direction, since (1/2)^I moves by 2x
per unit of I; no systematic sign. HawkesN data: the chain is HIGHER than the simulated fizzle
probability in at least 7 of 8 cascades, typically by a factor 2 or more.
**Falsifier.** HawkesN: chain higher in 5 or fewer of 8.  **Why:** under no systematic direction,
>= 7 of 8 has probability 0.035.
**Blind spot.** Rounding I to an integer (upstream's choice) adds noise at small I that is not about
SIR-vs-HawkesN; I report the unrounded kernel mass alongside.

### Observed, E10b (2026-11-27; `size_posterior_2026_11.py truth5`; `nov_size_truth_check_E10b.{json,png}`)
Held. After 5 observed events:
- SIR-generated cascades: paper's method vs simulated continuation from the true hidden state: 0.065 vs 0.125,
  0.065 vs 0.134, 0.129 vs 0.130, 0.065 vs 0.059, 0.129 vs 0.131, 0.065 vs 0.033, 0.254 vs 0.248, 0.065 vs 0.063
  (simulation se ~0.005). Equal when the rounded estimate equals the true infectious count, off by ~2x per
  unit of error in that count, either sign. As expected.
- HawkesN-generated cascades: paper's method above the simulated truth in 8 of 8, by a factor 8 to 26
  (e.g. 0.129 vs 0.006; 0.254 vs 0.033; 0.033 vs 0.001).
After 30 events (same run): 90% range of the final size for HawkesN cascades is 95 posts wide by the paper's
method and 70 by simulation; for SIR cascades the two agree.
Not checked: other K, N; cascades with background arrivals; fitted instead of true parameters.

### Session note (2026-11-27)
The machine rebooted twice during this session (about 11:55 and again between 12:40 and 17:14), killing the
first E5 replicate run at 450/800 and the first R tutorial run. No partial results from those runs were
used. E5 was restarted from scratch with the same seeds and a checkpoint file
(`nov_ident_replicates.jsonl`); the R tutorial was restarted with the same seed.

## E12. Does the second lobe go away with a larger population?  (estimator behaviour; registered while E5 is still running, having seen only the four E4 examples and partial E5 output for scenario P)

**Setup.** M2-like cascades (K=1.7, theta=0.4, one seed, complete to T=30/theta... T chosen as
(ln N + 25)/r + 20/theta so the decline is fully inside the window at every N; kept if >= 20% of N is
reached), N in {300, 1000, 3000, 10000}, 100 cascades each (seeds SeedSequence([20261150, N index, rep])),
exact grid only (no optimiser).
**Claim under test.** If the model's parameters are identifiable in the ordinary large-sample sense,
the truth-side lobe must eventually beat the "N = observed count, K huge" lobe, because the two
describe differently shaped curves (logistic vs SIR) and the log-likelihood difference between
shapes grows in proportion to the number of events.
**Expectation.** Share of cascades whose global maximum has K > 20: falls with N, from roughly a third
or more at N = 300-1000 to under 10% at N = 10000. Median width (upper/lower ratio) of the K interval
computed inside the K <= 20 box: shrinks roughly like N^(-1/2) in log-width, to below a factor 1.5
at N = 10000. Coverage of the growth-rate interval stays 88-99.5% at every N.
**Falsifier.** Share with K > 20 at N = 10000 not lower than at N = 1000 (within binomial error: a
drop of less than 10 percentage points with 100 cascades each is not distinguishable from none): then
the second lobe is not a small-sample effect and K cannot be fitted from a single complete cascade
at any realistic size without an outside constraint.
**Blind spot.** Only one K; near-critical cascades (K close to 1) are the hardest case and K = 3 the
easiest (E4/E5 scenario M3), so this bracket may not transfer. T grows with N, so "complete" is defined by design, not by a stopping rule a practitioner could apply.

### Observed, E12 first run (2026-11-27; `lobe_vs_N_2026_11.py`; `nov_lobe_vs_N.{jsonl,json,png}`) -- partly invalid, see below
- Share of cascades whose global maximum has K > 20: 0.36 (N=300), 0.30 (N=1000), 0.18 (N=3000), 0.03 (N=10000)
  (binomial se ~0.05 at the first three). Expectation ("a third or more falling to under 10%") held; falsifier not triggered.
- Not expected: even with K <= 20 imposed, the best fit sits far from the truth at small N. Median constrained
  K estimate 9.9 (N=300), 8.8 (N=1000), 2.3 (N=3000), 1.94 (N=10000), true 1.7; at N = 1000 the quartiles are 2.0 and 19.7.
  And the true parameters lie 2.6 log-lik units below the maximum in the median cascade at N = 1000 (0.23 expected for
  a well-behaved one-parameter profile), so 95% intervals for K contain the truth in only 31-47% of cascades at
  N <= 1000, and for the growth rate in 66-79%.
- Because that looked like a bug, three checks before believing it (all at K=1.7, theta=0.4, N=1000, one seed):
  (i) brute-force quadrature likelihood reproduces the grid maximum and the value at the truth to 3 decimals on 4 cascades
  (differences max-minus-truth 4.20, 7.01, 0.36, 3.57 are real); (ii) over 3000 UNCONDITIONAL simulated cascades (seed 777001)
  the mean score at the true parameters is zero within 2 se in each of log K, log theta, log N
  (-0.16+-0.39, 0.25+-0.13, -0.63+-0.36); (iii) the mean likelihood ratio of a fixed alternative
  (K=2, theta=0.28, N=900) to the truth is 0.997 +- 0.044 (must be 1 if the simulator draws from the likelihood's model).
  So simulator and likelihood agree and the behaviour is a property of the model. These three were not pre-registered;
  they are diagnostics run after an unexpected result, with pass criteria that follow from likelihood theory.
- **Explanation I now hold (derived after the fact, consistent with all numbers above, not independently tested):**
  setting N equal to the observed post count makes the final size certain under the fitted model, whereas the true
  model spreads the final size over +-25 posts or so at N = 1000. That is worth about log(sqrt(2 pi) * 25) = 4 log-lik
  units to the wrong solution regardless of sample size (growing only like half the log of N), against a penalty
  for the wrong curve shape that grows in proportion to the number of posts. At ~700 posts the two are comparable;
  at ~7000 the shape penalty wins. Parameters "between" the two solutions cash in part of the bonus, which is why
  the constrained fits drift to large K as well.
- **Invalid part:** my theta grid has 3.5% steps. At N >= 3000 the growth rate is determined to about 1-2%, so the
  grid cannot resolve the peak and both the maximum and the profile-at-truth are under-estimated by up to ~1
  log-lik unit at random. The coverage numbers and interval widths at N = 3000 and 10000 from this run are therefore
  not usable (the lobe shares are, the two lobes being many units apart in theta). Rerun with a 0.46% theta grid: E12r.
  The same limitation applies in principle to E5; there the worst case is scenario P (growth rate known to ~3.5%,
  about one grid step, error up to ~0.1 units).

## E12r. E12 at N = 3000 and 10000 with a fine theta grid (3000 points over six decades)
Same seeds, same cascades. **Expectation:** lobe shares unchanged within 2 percentage points; coverage of K, N,
growth rate inside the K <= 20 box rises relative to the first run, to 85-97% at N = 10000 and 70-90% at N = 3000;
median K-interval ratio at N = 10000 between 1.3 and 1.6. **Falsifier:** coverage at N = 10000 still below 80% for the
growth rate: then the undercoverage is not a grid artefact and extends to large cascades.
**Blind spot:** N grid unchanged (about 6% steps in N minus observed count).

### Observed, E12r (2026-11-27; `lobe_vs_N_2026_11.py fine`; `nov_lobe_vs_N_fine.{jsonl,json}`) -- my grid explanation was WRONG
With the 0.46% theta grid: N = 3000: lobe share 0.18, coverage inside the K <= 20 box K 0.56, N 0.57, growth rate 0.73,
median K-interval ratio 2.05, median K estimate 2.33. N = 10000: lobe share 0.03, coverage K 0.71, N 0.73, growth
rate 0.79, interval ratio 1.45, median K 1.93 (true 1.7), median N estimate 8850 (true 10000).
All within 0.02 of the coarse-grid run. The falsifier fired (growth-rate coverage 0.79 < 0.80 at N = 10000): the
under-coverage is not a grid artefact. The first-run E12 numbers therefore stand as they were, and the "invalid
part" paragraph above is withdrawn.
Follow-up diagnostic (not pre-registered; seed 777002; N = 10000, 150 take-off cascades): the mean log-likelihood ratio
of three FIXED wrong parameter sets against the truth is negative, as it must be if likelihood and simulator
describe the same model: (K=1.93, N=8850) -1.50 +- 0.20; (K=2.6, N=7700) -19.6 +- 1.0; (K=1.5, N=11850) -1.61 +- 0.13,
each with the same growth rate and final size as the truth. So a 13% error in K costs only 1.5 log-likelihood
units at ~6900 posts: the direction "same growth rate, same final size" is nearly flat even at this size, and the
log-ratio's spread (sd 2.45) is larger than a well-behaved likelihood would give for that mean (sqrt(2*1.5) = 1.7).
**Status of the explanation.** Established: (1) code and simulator agree (E1, E1b, E2b, score test, two likelihood-
ratio tests); (2) for single-seed K = 1.7 cascades, likelihood-ratio 95% intervals contain the truth far less
than 95% of the time at every size tested up to ~7000 posts, and the point estimate of K is biased upward
(median 1.93 at N = 10000). Not established: why the intervals stay over-confident at 7000 posts where the
second lobe is rare. The final-size "bonus" argument explains the second lobe, not this. I could not establish it.

### Observed, E5 and E5-add (2026-11-27; `identifiability_2026_11.py replicate 200` (resumed from checkpoint after a reboot, same seeds), `identifiability_summary_2026_11.py`; `nov_ident_replicates.{jsonl,json}`, `nov_ident_replicates_summary.json`, `nov_ident_recovery.png`, `nov_ident_coverage.png`, `nov_ident_fitter_gap.png`)
200 cascades per scenario, no fitter errors.

| | P (300 seeds, K=5) | M1 (1 seed, N=110, K=1.7) | M2 (1 seed, N=1000, K=1.7) | M3 (1 seed, N=1000, K=3) |
|---|---|---|---|---|
| events, median | 983 | 76 | 688 | 940 |
| best fit anywhere has K > 20 | 0.5% | 47.5% | 26% | 0.5% |
| 95% profile-interval coverage: K / theta / N | 91.5 / 91.5 / 92 % | 27.5 / 29 / 26.5 % | 44 / 43.5 / 44.5 % | 66 / 65 / 67.5 % |
| ... K*theta / growth rate | 94.5 / 96 % | 45.5 / 66.5 % | 45.5 / 71.5 % | 67 / 82.5 % |
| coverage with K <= 20 imposed: K / growth rate | 91.5 / 96 % | 28.5 / 68.5 % | 47 / 75 % | 66 / 82.5 % |
| existing fitter's own +-1.96 se: K / theta / N | 99 / 99.5 / 81 % | 83 / 70.5 / 28.5 % | 85 / 77 / 62 % | 72 / 71.5 / 54.5 % |
| existing fitter, estimate/truth, 5th-95th pct: K | 0.76-2.0 | 0.94-11.8 | 0.89-4.9 | 0.80-2.9 |
| ... theta | 0.45-1.42 | 0.03-1.11 | 0.10-1.36 | 0.23-1.38 |
| ... N | 0.988-1.013 | 0.53-1.08 | 0.70-1.16 | 0.93-1.06 |
| ... K*theta | 0.88-1.09 | 0.36-1.20 | 0.47-1.21 | 0.66-1.17 |
| ... growth rate | 0.93-1.08 | 0.63-1.74 | 0.79-1.27 | 0.87-1.08 |
| existing fitter within 20% of truth: K / growth rate | 52.5 / 100 % | 22.5 / 41.5 % | 53 / 82.5 % | 37 / 100 % |
| exact-maximum estimate / truth, median: K / N | 1.12 / 0.997 | 9.7 / 0.72 | 5.2 / 0.71 | 1.85 / 0.95 |
| existing fitter > 0.1 log-lik below the true maximum | 0% | 38% | 47.5% | 18.5% |
| ... > 1.92 below (outside the 95% region) | 0% | 5.5% | 34% | 8.5% |
| existing fitter pinned at its K = 20 bound | 0.5% | 36.5% | 0% | 0% |

Against the expectations:
- **E5 as registered failed outside the paper's setting.** Coverage is in the expected band only for scenario P
  (91.5-96%; K, theta, N at 91.5-92% are at the low edge of the 91-98% I allowed and above the 88% falsifier).
  For every single-seed scenario the falsifier fired for K, theta and N, including M3, where I had predicted no problem.
- **E5-add:** second-lobe frequency 47.5% (M1) and 26% (M2) were in the predicted 20-50% band; under 1% in M3 and P as
  predicted. But two of my E5-add predictions were wrong: growth-rate coverage is 66-83% in the single-seed scenarios (I
  predicted 88-99.5%: its falsifier fired, so "the growth rate's INTERVAL is safe" is false, although its point
  estimate is the only one that stays near the truth), and imposing K <= 20 does not restore coverage in M2 (47%; I predicted > 88%).
- **Fitter (expectation 4 of E4):** in P it finds the exact maximum every time. In the single-seed scenarios it stops
  more than 0.1 below the true maximum in 18-48% of cascades. In M2 it lands on the truth-side local maximum more often
  than the exact search does, so its point estimates are CLOSER to the truth than the exact maximum's (median K ratio
  1.17 vs 5.2). That is an accident of its start point (K = 0.8) and box, not a property to rely on.
- Its Hessian-based standard errors under-cover in every single-seed scenario and for N everywhere (81% even in P).
- Unexplained at this point: why M3 (one lobe, only 6% of cascades die out) has 66% coverage and a median K estimate
  1.85x too high. Next entry.

## E13. Is conditioning on take-off the cause?  (estimator behaviour; registered after seeing E5/E12)

**Cause hypothesis.** Every single-seed cascade analysed here was kept only if it took off (>= 20 events), and
real memes are selected the same way. The likelihood used so far is the unconditional one. The correct
likelihood for selected data divides by P(take-off | parameters), which for a single seed is essentially a function of K alone
(1 minus the probability that a Poisson(K) branching process has fewer than 21 members in total). Evidence
that this matters: in the 3000-cascade score test the mean score along the flat direction is ~0 unconditionally
(-0.15) and +0.89 among take-off cascades, matching d log P(take-off)/d log K = 0.97 at K = 1.7.
**Test.** Recompute the K profile for the same M2 and M3 cascades, subtract log P(take-off | K), and redo the point
estimate and the 95% interval for K.
**Prediction.** M2: the correction pulls estimates down: median K estimate/truth falls from 5.2 to below 2 and K
coverage rises from 44% by at least 15 points, but not to 95% (the "N = observed count" solution is untouched by the
correction). M3: P(take-off) is nearly flat in K there (0.94 at K=3), so I predict little change (coverage within
5 points of 66%, median ratio still above 1.5). If so, M3 needs a different explanation, my candidate being that with
one seed the level of the intensity early on carries no usable information on K*theta, so K rests on curve shape alone.
**Falsifier.** For "selection is the main cause": M2 K coverage after correction below 59%. For "selection is
irrelevant in M3": M3 coverage after correction above 80%.
**Blind spot.** P(take-off) is computed for an infinite population (Borel distribution of total progeny), ignoring
depletion during the first 20 events; negligible at N = 1000. Only K's interval is recomputed.

### Observed, E13 (2026-11-27; `takeoff_correction_2026_11.py`; `nov_takeoff_correction.{jsonl,json,png}`)
The hypothesis "conditioning on take-off is the main cause" is **falsified**. With the likelihood made conditional on take-off:
M2: K coverage 44.5% -> 47% (falsifier: below 59%), median K estimate/truth 5.2 -> 4.7, share with K > 20 27.5% -> 26%.
M3: coverage 67% -> 68%, median ratio 1.86 -> 1.78 (the "little change" prediction for M3 held).
The correction is real (the conditional score check supports it) and too small to matter.

## E14. Two remaining candidate causes, separated by a 2 x 2 design  (estimator behaviour)

Candidates for why likelihood intervals fail and K is over-estimated in complete single-seed cascades but not in the
paper's setting, which differs in two ways at once (300 seeds; window ends slightly before the cascade is over):
 (A) **"finished cascade" effect**: once the cascade is observed to its end, "N = observed count" explains the stop for
     free; the likelihood is pulled towards small N and large K. Predicts: bad when complete, fine when the window stops
     while the cascade is still running, whatever the number of seeds.
 (B) **"single seed" effect**: with one seed the early intensity level carries no information on K*theta, so K rests on
     curve shape alone. Predicts: bad with 1 seed, fine with 300, whatever the window.
**Design.** K=3, theta=0.4, N=1000 (scenario M3's parameters). Seeds I0 in {1, 300}. Each cascade simulated to T=75
and analysed twice: "complete" (T=75) and "stopped" (window ends at the first moment the total count reaches 85%
of the mean-field final count: a stopping rule that uses no hindsight). 100 cascades per seed setting that reach that
level (seeds SeedSequence([20261160, I0 index, rep])); exact grid; global maximum; 95% profile intervals.
**Primary metric (fixed now):** coverage of K. Also reported: median K estimate/truth, coverage of the growth rate and N.
**Predictions.** I expect (A) to be the larger effect: stopped windows give K coverage >= 85% for both seed settings;
complete windows give <= 75% for I0 = 1 (E5 gave 66%) and a visible drop (below 88%) for I0 = 300 as well.
**Falsifiers.** (A) is wrong if stopped-window coverage with I0 = 1 is below 80%. (B) is wrong if complete-window
coverage with I0 = 300 is below 80%. Both can be partly right; all four cells are reported.
**Why 80/85/88.** With 100 cascades a true 95% gives >= 89 with probability 0.99; 80 or fewer has probability < 1e-6.
**Blind spot.** One parameter set; with a stopped window N is expected to have no upper bound, which is not a failure.
"Stopped at 85%" is still late in the cascade; an early stop would be dominated by lack of information instead.

### Observed, E14 (2026-11-27; `two_by_two_2026_11.py`; `nov_two_by_two.{jsonl,json,png}`)
K=3, theta=0.4, N=1000; 100 cascades per seed setting; coverage of 95% profile intervals:

| | events (median) | K | theta | N | growth rate | median K estimate / truth (5th-95th pct) |
|---|---|---|---|---|---|---|
| 1 seed, window stopped at 85% | 798 | 94% | 94% | 93% | 94% | 3.1 (0.93-674) |
| 1 seed, seen to the end | 940 | 71% | 73% | 71% | 84% | 1.46 (0.84-3.3) |
| 300 seeds, window stopped at 85% | 517 | 97% | 97% | 96% | 97% | 1.35 (0.49-918) |
| 300 seeds, seen to the end | 661 | 79% | 78% | 81% | 86% | 1.27 (0.68-2.7) |

- (A) held: with the window stopped before the end the intervals are honest for both seed settings (93-97%); seen to
  the end they are not (71-86%). The prediction "<= 75% for one seed, below 88% for 300 seeds" held (71%, 79%).
- (B)'s falsifier fired (complete window with 300 seeds: 79% < 80%): having many seeds does not protect. The number of
  seeds matters for how tight the estimate is, not for whether the interval is honest.
- The cost of honesty: with a stopped window the K interval is honest because it is enormous (5% of point
  estimates are more than 600x too high); K is simply not determined before the decline has been seen.
- Interpretation: once a cascade has run its course, the population size has a hard lower limit at the
  observed count, and the likelihood piles up against that data-dependent limit. Standard likelihood-ratio and
  Hessian intervals assume no such boundary. This accounts for E5 (M1-M3), E12 and the residual 91.5% in scenario P
  (whose window ends just before the end). It is an account supported by this 2 x 2 test at one parameter setting, not a proof.

### E7, version friction (2026-11-27)
Under R 4.5.3 / nloptr 2.2.1 the tutorial's cells 1-13 ran (simulation, SIR fit: median N 1284, gamma 0.199,
beta 0.981, sd 5 / 0.0075 / 0.041, inside the expected ranges) and then the HawkesN-fit cell stopped with
`'length = 5' in coercion to 'logical(1)'`. Cause: upstream `fitSeries` evaluates `is.na(model) || ...` on a
5-element list, which R >= 4.3 treats as an error (older R used the first element with a warning). The fits
themselves had finished; the error is in the check after them. Console: `nov_upstream/tutorial_run/console_log_ckpt.txt`.
Decision: do not patch upstream code. Installed a second user-level environment with R 4.2.3 / nloptr 2.0.3
(`~/wp12-ext/renv42`, conda-forge) and re-ran the whole tutorial from the same seed in
`nov_upstream/tutorial_run_R42/`. The notebook was produced with R 3.4.3; 4.2.3 is the newest release line on
which the code runs unmodified. E7's expectations apply to this run unchanged.

### E7, version friction, continued (2026-11-27)
R 4.2.3 / nloptr 2.0.3 (`nov_upstream/tutorial_run_R42/`): simulation and SIR fit identical to the R 4.5 run
(median N 1284, gamma 0.1987, beta 0.981), then every HawkesN fit returned NA in 8 seconds. Cause (reproduced on
one cascade): upstream passes `xtol_rel = "1e-8"` as a character string; nloptr 2.0.3 rejects it
(`REAL() can only be applied to a 'numeric'`), upstream's tryCatch swallows the error and returns NA, and the
tutorial's table prints NA. So the published code runs unmodified on neither current R (4.3+: the `||` error) nor
R 4.2 with its matching nloptr. Third attempt: R 3.5.1 / nloptr 1.0.4 / Matrix 1.2.18 / data.table 1.12.8 from
conda-forge (`~/wp12-ext/renv35`), the closest available to the notebook's R 3.4.3, in `nov_upstream/tutorial_run_R35/`.

### Observed, E7 / E7b / E9a (2026-11-27; R 3.5.1 run `nov_upstream/tutorial_run_R35/` (console_log_ckpt.txt, csv exports, tutorial_size_distribution_R.png); `upstream_compare.py tutorial`; `size_posterior_2026_11.py rcheck`)
- **E7 held.** Unmodified tutorial, seed 20261127 (harness: per-cell workspace checkpoints only).
  SIR fit: median N 1284 (notebook 1283.5), gamma 0.1987 (0.200), beta 0.981 (0.986); sd 5 / 0.0075 / 0.041 (3.6 / 0.006 / 0.026).
  HawkesN fit (20 min on 4 cores): median K 5.3 (notebook 5.5), theta 0.18 (0.18), N 1296 (1292); sd 1.6 (2.1), 7.39 (8.52), 14 (11).
  Converted to SIR parameters, row labelled "median" in the notebook but computed as a mean: N 1300, gamma 2.0, beta 3.9
  (notebook 1295, 3.5, 6.4; sd 14 / 7.4 / 11.9 vs 11 / 8.5 / 13.7). As expected one fit ran away (cascade 7: theta = 30.7), and
  additionally 3 of the 20 upstream fits returned NA (the table silently drops them with na.rm).
  Figure: prior two-peaked, posterior after 641 of 1282 events one-peaked; visually the same as the repo's `images/output_29_1.png`.
- **My E6 follow-up prediction was half wrong.** Upstream's objective is unbounded as predicted (on all 20 cascades it is lower
  at theta = 1e3 than at the truth and lower again at 1e6, e.g. -3952 -> -4261 -> -10904). But the runaway fit is NOT the
  optimiser profiting from that: at its returned point (theta = 30.7) upstream's own objective is -3532, worse than at the
  truth (-4012). The optimiser stopped at a bad point (nloptr reports "generic failure" on the one cascade I ran by hand),
  consistent with the gradient not matching the objective, but I did not establish the mechanism.
- **E7b held (falsifier not triggered), one sub-prediction wrong.** On the 17 cascades where upstream returned numbers, the JAX
  fitter's textbook log-likelihood is never below upstream's (difference min 0.000, median 0.20, max 808 for the runaway);
  JAX is within 0.1 of the exact grid maximum on 20/20. Estimates: JAX median K 5.28, theta 0.184, N 1300, K*theta 0.995,
  growth rate 0.792 (truth 5, 0.2, 1300, 1.0, 0.8); sd of K*theta 0.083 and of the growth rate 0.055. Wrong sub-prediction:
  JAX's K spread is not smaller than upstream's (sd 2.30 over 20 cascades vs 1.62 over upstream's 17 survivors).
  Plot: `nov_upstream/E7b_tutorial_cascades_fits.png`.
- **E9a held.** Our O(N^2) recursion vs the R output: largest difference in any probability 1.1e-16 (prior) and 4.2e-16
  (posterior after 641 events, estimated state S = 659, I = 534). `nov_size_rcheck.{json,png}`.

### Observed, E8 first step (2026-11-27)
Upstream `fitSeries` from the tutorial's start point (K=1, theta=0.1 per second, N=1000) returned NA for 29 of the 30 tweet
cascades within 0.1 s each (`nov_upstream/E8_upstream_fits_R.csv`). So the comparison as registered cannot be made:
there is nothing to compare on 29 cascades. This is not what I expected (I expected poor fits, not no fits).
I did not find in the repository the start points or time units the authors used for real cascades.

## E8b. Same cascades, upstream fitter given workable start points  (estimator behaviour; redesign after E8)
Two start points per cascade, both my choice and therefore not "upstream as published": (s1) generic: K = 1,
theta = 1/(median gap between tweets), N = 2 x cascade size; (s2) the JAX fitter's solution.
**Expectation.** From s2 upstream moves away from the JAX solution on most cascades (because its objective differs,
E6) towards larger theta, and scored by the textbook likelihood ends up worse than where it started in >= 80% of
cascades. From s1 it returns a finite answer on most cascades, agreeing with JAX on K within a factor 1.5 in fewer
than half. **Falsifier:** upstream started at the JAX solution stays within 10% of it on K, theta, N in >= 80% of
cascades: then the two objectives have nearly the same optimum on real cascades and the E6 difference is immaterial in practice.
**Blind spot:** as E8; plus, with start points of my choosing this tests upstream's objective and optimiser, not the authors' workflow.

### Observed, E8 / E8b (2026-11-27; `upstream_compare.py fit`, `fit2`; `nov_upstream/E8_fit_comparison.json`, `E8b_upstream_fits_R.csv`, `E8b_fit_comparison.{json,png}`, `E8b_summary.json`)
**The JAX-vs-upstream cross-check on real tweet cascades could not be made.** Under R 3.5.1 / nloptr 1.0.4, on 30 News cascades (50-300 tweets):
- tutorial start point: 29 of 30 return NA at once ("gradient of objective in x0 returns NA");
- generic start (K=1, theta = 1/median gap, N = 2n): 17 NA; the other 13 return the start point unchanged after 12 evaluations with nloptr status -1;
- started at the JAX solution: 1 NA; 28 of the remaining 29 return the start point unchanged (status -1); one moved theta by 2.5x.
E8b's falsifier ("stays within 10% of the JAX solution in >= 80%") fired, but for a reason that makes it meaningless: the
optimiser did not move, so this says nothing about whether the two objectives share an optimum. My test design did not
anticipate a fitter that returns its input. What is established about upstream is therefore limited to: its objective is
not the textbook likelihood (E6), and on the tutorial's simulated cascades, where it does run, it agrees with JAX when it
converges (E7b). I did not establish how the paper's real-data fits were produced.
**By-product that matters for reuse:** on these 30 real cascades the JAX fitter is within 0.1 log-lik units of the exact maximum of
its own likelihood on 57% and more than 1.92 below it on 43% (largest gap 213 units); in 33% the exact maximum has N equal to the
observed tweet count. Not pre-registered as a claim; it is the same fitter-vs-grid comparison as E5, on real data.

---
## End-of-session status (2026-11-27). Everything above is UNCERTIFIED: produced and checked by one agent only.
Not done: no entry was written to `docs/` (this session was told not to edit it); the work-package log, state snapshot,
changelog and open-questions list still need the items above. `hawkesn.py` was not modified; its known defects are
listed in E1 (background-rate compensator), E5/E8 (optimiser stops short of the maximum; K <= 20 bound) and it still
defaults the window end to the last event. Scripts added this session: `checks_2026_11.py`, `exact_profile.py`,
`identifiability_2026_11.py`, `identifiability_summary_2026_11.py`, `lobe_vs_N_2026_11.py`, `takeoff_correction_2026_11.py`,
`two_by_two_2026_11.py`, `single_seed_convergence_2026_11.py`, `size_posterior_2026_11.py`, `upstream_compare.py`,
`r_upstream/` (all under `experiments/wp12/`). R environments and the upstream clone are outside the project in `~/wp12-ext` (3.5 GB).
