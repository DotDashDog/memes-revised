# Research Note — Neural-Hawkes Track: Critique & Phased Replication Plan

**Status:** speculative → *plan drafted*. The plan below is proposed; the work packages it spawns (WP-12…WP-15) are `open` drafts and the scoping decisions are recorded in `decisions/0014-hawkes-track-scope.md` (`proposed`). Nothing here is binding until the user accepts ADR 0014.
**Builds on:** [`hawkes-processes.md`](hawkes-processes.md) (concepts, neural families, literature), `../open-questions.md` → "Holistic spatio-temporal point-process model".
**Governed by:** `../01-scientific-rigor.md` (§1 baselines, §10 failure modes) and `../method-discipline.md` (§1 identifiability, anti-cargo-cult); every standard in both applies.

---

## 0. TL;DR

The instinct to *replicate before inventing* is correct and is the spine of this plan. But the honest finding from the literature pass is: **no existing paper does what this project ultimately needs** (interpretable, marked neural Hawkes on a ~$10^3$-d embedding mark, at $5\text{M}+$ events). So "replicate a paper" can only carry us to *building blocks*. The composite architecture sketched in `hawkes-processes.md` §5 is a genuine invention, and must be labelled as such — not smuggled in under the word "replication."

The plan therefore stages the work so that **each phase replicates something that actually exists with code**, and the novel integration is attempted only after its load-bearing assumptions have each been validated on data where we know the answer:

- **Phase 1 (WP-12):** Replicate **SIR-Hawkes / HawkesN** (Rizoiu et al., WWW 2018) — the one paper that *is* the SIR↔Hawkes bridge. Faithful run of the upstream R tutorial, then a JAX reimplementation validated against it. Establishes the depleting-Hawkes machinery and connects directly to WP-02.
- **Phase 2 (WP-13):** Validate the **neural-TPP training machinery**, temporal-only, on synthetic Hawkes and on a *single real meme's* post stream pulled from the archive we already have. Anchored on EasyTPP (maintained PyTorch benchmark); minimal JAX neural-Hawkes core validated against it.
- **Phase 3 (WP-14):** The **identifiability / endogeneity due-diligence study** — the Hawkes analogue of WP-02. Can background-rate be separated from self-excitation? Synthetic, gated *before* any real-data parameter claim. This is the highest-EV item after Phase 1.
- **Phase 4 (WP-15):** The first **marked** model: low-dim-projection + analytic-kernel STPP on a *small* embedding subset, measuring whether held-out future log-likelihood beats independent per-cluster SIR. Gated on the cluster baseline existing (the note's original guard, preserved for real-embedding spatial work only).

A **scaling go/no-go gate** (§5) sits between Phase 3 and any full-corpus run: measured per-event cost × event count, computed before committing compute.

The four architecture-critical decisions taken at kickoff (anchor = SIR-Hawkes; marks staged temporal-first in a *frozen* low-d projection; upstream-code + JAX-core; parallel synthetic track) are recorded in ADR 0014.

---

## 1. Critique of the existing note and plan

Read this as "checking the user," as requested — these are the places the current framing is over-optimistic, internally tense, or missing a gate.

### 1.1 The §5 "suggested architecture" is an invention, not a replication target
`hawkes-processes.md` §5 stacks: a Family-B attention encoder, a Family-A additive head with **per-content decay** $\beta_\theta(\mathbf{x}_i)$ and a **learned spatial kernel** $s_\theta$, a **learned low-dim projection** $W$, *and* a **PINN-style SIR penalty**. Each piece exists in *some* paper; the combination exists in none, and several pieces are in tension (see 1.2). Building this first would violate the user's own "hew close to an existing paper" directive. **The replication phases deliberately do not build it.** It remains the aspirational target the staged work is *de-risking toward* — promote it to a real architecture ADR only if Phases 1–4 each clear.

### 1.2 The interpretability ⟂ scalability ⟂ tractability trilemma is real and under-acknowledged
The note wants, simultaneously: spatial + temporal interpretability, $5\text{M}+$ scalability, and a tractable compensator integral. These pull against each other:
- **Family A** (additive kernel) keeps the *named* quantities — branching ratio $n$, decay timescale $1/\beta$ — that give the SIR bridge and the interpretability. But the spatial compensator $\int\!\int\lambda^*$ is the cost bottleneck, and additive kernels scale worst.
- **Family B** (attention encoder → $\lambda^*=g_\theta(\mathbf{h}(t),\mathbf{x})$) scales best, but $\mathbf{h}(t)$ is a black box; "branching ratio" and "per-content $\gamma$" survive *only* if the head is constrained back into additive form — which fights the very flexibility that made the encoder worth using.
- **Family C** (intensity-free / CNF, e.g. the Chen et al. paper the original note said to "start here") sidesteps the compensator but **discards the named kernel entirely** — it is the *least* interpretable, directly opposed to the stated dual-interpretability goal. This is why the kickoff decision moved the anchor *off* Chen et al. and onto SIR-Hawkes.

Consequence for the plan: prove the **interpretable-but-small** thing works and is identifiable (Phases 1–3) *before* paying for scale, and treat "do the named quantities survive when we scale the encoder?" as an explicit measured question in Phase 4, not an assumption.

### 1.3 Identifiability of background vs. excitation needs the WP-02 treatment, up front
The note flags background-vs-excitation as "the endogeneity problem in disguise" (§6.2) but leaves it as prose. WP-02 earned its priority by being "the single highest-EV experiment before any data work" — the *exact same logic* applies here: if $n$ (branching ratio / the $R_0$ analogue, the headline interpretable quantity) is not identifiable from realistically-corrupted streams, then predicting it from content is meaningless in the same way predicting raw $(\beta,\gamma)$ was. **This is elevated to a gated work package (WP-14), not a paragraph.**

### 1.4 The holistic model does not, by itself, serve the stated end goal
The end goal is *"predict how a given meme will grow and evolve in time."* A marked STPP gives you $\lambda^*(t,\mathbf{x}\mid\mathcal{H}_t)$ — the next-event intensity over the *whole field given full history*. That is not the same object as "this one meme's future trajectory from its content." You recover a per-meme forecast only by **conditioning the intensity on a sub-region of embedding space and integrating it forward** — which requires the spatial kernel $s_\theta$ to actually *localize* memes (it is implicitly doing HDBSCAN's job). If $s_\theta$ does not cleanly separate memes, per-meme forecasts are ill-defined and the holistic model is "scientifically interesting" but no better than cluster-SIR at the actual deliverable. **Phase 4 must include an explicit test: can the learned field be sliced per-meme and forecast forward, and does that forecast beat per-cluster SIR?** If not, that is a real (publishable) negative result, not a tuning failure.

### 1.5 Corrupted event stream biases the headline quantity in a known direction
Reposts, mod deletions, and posts≠infections (R5) interact badly with NLL, which treats every observed event as a true draw. Specifically: **duplicate repost "events" inflate measured self-excitation exactly where we're trying to estimate it**, biasing the branching ratio $n$ *upward* (memes look more contagious than they are); deletions are missing-not-at-random and bias it in a data-dependent direction. This is not a nuisance to footnote — it's a systematic bias on the one number the whole interpretable story rests on. WP-14 must include a repost/deletion robustness ablation, and any reported $n$ must carry that caveat (`method-discipline.md` §7, surface assumptions).

### 1.6 "Replicate a paper" cannot reach the target in one hop — and that's the key honesty point
To make the dependency on novelty explicit:

| What we want | Closest paper with code | What's missing |
|---|---|---|
| SIR↔Hawkes bridge, rise-peak-fall | **SIR-Hawkes / HawkesN** (R, tutorial) | no spatial marks; not neural; not at scale |
| Neural TPP training machinery | **EasyTPP** (PyTorch, maintained) | temporal-only marks (event *types*, not high-d vectors) |
| Marked neural STPP | **Neural STPP / Chen et al.** (PyTorch, *archived* 2023) | low-d marks (≤~3-d); CNF kills interpretability; not at scale |
| Interpretable low-rank kernel STPP | **DNSK** (no public code found) | implement from equations; synthetic-only validation; low-d |

There is **no** existing artifact at the intersection (interpretable + marked-high-d + scalable). The plan's job is to assemble validated blocks and to be loud about which step is the unreplicated invention. Linear Hawkes alone is also insufficient: it has *no depletion* and cannot peak-and-decay, so a linear-Hawkes replication teaches machinery but not meme dynamics — which is precisely why the anchor is HawkesN (depleting), not vanilla Hawkes.

### 1.7 Minor / housekeeping
- **Data readiness is better than the note implies for the temporal-only phase.** We don't need clusters or embeddings to start Phase 2: the existing `reddit_data/reddit/subreddits23/memes_submissions.zst` (Pushshift NDJSON) has per-post `created_utc`. A single known meme's stream is extractable *now*. (One-time format verification is a WP-13 prerequisite, not an assumption.)
- The note's literature is solid and verified; this plan adds the maintenance/format facts (Neural STPP archived Oct 2023; SIR-Hawkes is R+notebook tutorial, 3 tweet-cascade datasets; EasyTPP is the live PyTorch benchmark) so nobody rediscovers them.

---

## 2. What "success" means for this track (so we don't move goalposts)

Stated up front per rigor §1–2. This track is judged on **two** things, separately:

1. **Forecasting (the user's end goal):** held-out *future* log-likelihood and next-event time/location error, and — the deliverable that matters — **per-meme trajectory forecast** quality, each compared against the mandatory baselines, *plus* one track-specific baseline: **independent per-cluster SIR** (the cluster-SIR pipeline this is an alternative to). If the holistic model does not beat per-cluster SIR at per-meme forecasting, it has not earned its complexity.
2. **Interpretability (the scientific claim):** are the named quantities — branching ratio $n$, per-content decay $\beta_\theta(\mathbf{x})$, background field $\mu(\mathbf{x})$ — *identifiable and stable*? An interpretable parameter you cannot identify is not interpretable. WP-14 is where this is earned or denied.

A model that wins (1) but fails (2) is a useful black-box forecaster but not the "spatially + temporally interpretable" object the user asked for. A model that has clean (2) but loses (1) to cluster-SIR is a nicer story that doesn't predict. We report both, always, side by side.

---

## 3. Phased plan

Each phase is a work-package draft (linked). Phases 1–3 are **synthetic / single-stream and run in parallel with WP-02** (ADR 0014 decision). Phase 4's *real-embedding spatial* work stays gated on the cluster baseline.

### Phase 1 — WP-12: SIR-Hawkes / HawkesN replication (anchor)
**Why first:** it is the SIR↔Hawkes bridge made concrete, it connects to the already-accepted SIR identifiability work (WP-02 / ADR 0010), and it gives us depleting-Hawkes dynamics (rise-peak-fall) that linear Hawkes cannot.
**Replicate:** run the upstream `computationalmedia/sir-hawkes` R/notebook tutorial as-is; reproduce its headline figure (final-size distribution collapsing from bi-modal prior to single-mode posterior as more of the cascade is observed) and its stochastic-SIR↔HawkesN parameter-recovery check.
**Build:** a minimal **JAX** HawkesN simulator + likelihood, validated two ways — against the upstream R fits on the same tweet cascades, and against synthetic streams with known $(\beta,\gamma,N)$ ground truth.
**Bridge to WP-02:** confirm numerically that the HawkesN mean-field limit reproduces the SIR ODE fits WP-02 studied — closing the loop the `hawkes-processes.md` §3 claim asserts analytically.

### Phase 2 — WP-13: Temporal-only neural TPP machinery
**Why:** before adding a high-d mark we must know the NLL/compensator training loop is correct. Marks excluded entirely (ADR 0014: temporal-first).
**Replicate:** reproduce a small EasyTPP result (e.g. NHP or THP on its synthetic Hawkes dataset) to verify our environment and our understanding of the loss.
**Build:** a minimal JAX neural-Hawkes (continuous-time decay state à la Mei-Eisner, or a small attention encoder) and validate it recovers the branching ratio on synthetic Hawkes, matching EasyTPP within tolerance.
**Real data, single stream:** extract one well-known meme's post-time series from the existing archive and fit temporal-only neural Hawkes vs. parametric HawkesN (from WP-12) vs. the rigor §1 baselines. First contact with real, corrupted timestamps.
**Cost probe (feeds §5 gate):** measure *per-event* training/eval cost here, where it's cheap, so the full-corpus extrapolation in §5 is grounded in measurement, not guesswork.

### Phase 3 — WP-14: Hawkes identifiability & endogeneity study (the WP-02 analogue)
**Why:** highest-EV due-diligence after Phase 1; gates every real-data *parameter* claim. Synthetic.
**Questions:** (a) Given a realistic corrupted stream, is the **branching ratio $n$** identifiable, and how does its posterior width scale with stream length / observation fraction (only-up-to-peak vs. full decay)? (b) Can **background $\mu$ be separated from excitation $\phi$** — the endogeneity problem — and under what conditions does it collapse into a ridge? (c) **Repost/deletion robustness:** inject duplicate-repost and MNAR-deletion corruption (per 1.5) and measure the bias on $n$; confirm direction and magnitude (`method-discipline.md` §3: state the predicted upward bias from reposts *before* running).
**Deliverable:** which Hawkes quantities are safe to predict from content, mirroring ADR 0010's two-tier target verdict for SIR.

### Phase 4 — WP-15: First marked STPP on a small embedding subset
**Gate:** cluster baseline exists (real-embedding spatial work only — the note's original guard, preserved here).
**Replicate-ish:** smallest faithful marked STPP we can stand up — analytic (Gaussian) spatial kernel for a closed-form spatial compensator, on a **frozen** low-d projection (ADR 0014: PCA 8–16d fit on training-period embeddings only, so NLLs are comparable across models; learned projections enter later as ablations *scored in the same frozen space*).
**The two decisive tests:**
1. Does held-out *future* log-likelihood beat **independent per-cluster SIR**? (§2 item 1.)
2. Can the learned field be **sliced per-meme and forecast forward** (§1.4)? Does that per-meme forecast beat cluster-SIR and the rigor §1 baselines?
**Failure is a result:** if either test fails, file it (rigor §7) — it tells the project the holistic view doesn't pay for itself at the deliverable, which is genuinely valuable.

---

## 4. Architecture-critical decisions (taken at kickoff → ADR 0014)

| Decision | Choice | One-line rationale |
|---|---|---|
| Anchor paper | **SIR-Hawkes / HawkesN** | only paper that *is* the SIR↔Hawkes bridge; connects to WP-02; depleting → rise-peak-fall |
| Mark staging | **Temporal-first; marks in a *frozen* low-d projection** | comparability of NLL across models; projection kept off the leak/contamination surface until ablated |
| Code strategy | **Run upstream as-is + build a JAX core (ADR 0013)** | faithful replication claim *and* a differentiable core we own; separates "paper discrepancy" from "our bug" |
| Scheduling | **Parallel synthetic track w/ WP-02; real-embedding spatial work stays gated on cluster baseline** | Phases 1–3 need no embeddings; only Phase 4 touches the gated pipeline |

These were the user's decisions on the four questions posed at kickoff; ADR 0014 records them as `proposed` pending formal acceptance.

---

## 5. The 5M+ feasibility gate (explicit, before any full-corpus compute)

The user's hard constraint is computational feasibility at $5\text{M}+$ memes/events. Make this a **go/no-go gate**, not a hope:

- **The cost cliff is entirely in the marked, full-corpus phase.** Phases 1–3 are cheap (single streams / small synthetic). So feasibility is decided *between Phase 3 and any full run*.
- **Naive STPP NLL is ~$O(N^2)$** (each event's intensity sums over history) **plus a spatial integral per event.** At $N=5\text{M}$ this is infeasible without **locality/windowing** (cap influence to recent-in-time and near-in-embedding events).
- **But windowing is itself an unvalidated modeling assumption** — it asserts far-apart events don't influence each other. So the cost model and the windowing assumption must be **validated together** (WP-14/WP-15), on data where the true influence range is known. Don't adopt windowing for speed and then report its biased $n$ as truth.
- **The gate, concretely:** take the measured per-event cost from WP-13's cost probe (§3 Phase 2) and the windowed-complexity factor validated in WP-14, extrapolate to $5\text{M}$, and require a *budgeted* estimate (GPU-hours on the local 3090, or a costed cloud plan) **before** launching a full run. If the extrapolation says infeasible, the deliverable is "holistic model is infeasible at target scale with current methods" — a legitimate stopping point, reported, not a reason to silently shrink $N$.

---

## 6. What I am NOT proposing (scope discipline)

- Not building the `hawkes-processes.md` §5 composite architecture yet (see 1.1).
- Not using the archived Chen et al. CNF model as the anchor (interpretability cost, see 1.2).
- Not touching the test set or making any real-data parameter claim before WP-14 clears (rigor §2; `method-discipline.md` §1 identifiability).
- Not running anything on the full 5M corpus before the §5 gate.
- Not amending ADR 0013 (JAX-first): upstream PyTorch/R is run *as reference*; our core stays JAX.

---

## 7. Open questions this plan surfaces (logged, not silently answered)

These are added to `../open-questions.md`:
- **Per-meme forecast extraction (1.4):** what is the precise operator that turns $\lambda^*(t,\mathbf{x})$ into "meme $m$'s future trajectory," and is it well-posed when $s_\theta$ doesn't cleanly localize?
- **Windowing as modeling vs. approximation (§5):** at what time/embedding radius does truncating influence change the fitted $n$ by more than its posterior width? (Determines whether windowing is "free" or a bias.)
- **Repost/deletion correction (1.5):** is there a principled MNAR correction (thinning model, dedup prior) or only a robustness ablation?
- **Does the cluster-SIR↔HawkesN mean-field equivalence (WP-12) hold under the *real* engagement signal** (ADR 0007), or only for raw counts?

---

## 8. Stopping rule & escalation ladder (the "don't kill a good idea prematurely" policy)

**Decided (2026-06-10, user):** (a) the WP-15 negative-result fallback is **Branch A — redirect, don't abandon**: a failed *holistic* model still leaves a differentiable HawkesN likelihood that becomes a better per-cluster fitter than the bin-counts→SIR-ODE step, flowing back into WP-08. (b) The neural STPP is high-potential enough that we **do not abandon it on a first failure** — we confirm the failure is *fundamental* first.

This is correct research instinct, but "explore until it works" is precisely the cargo-cult / Millikan-anchoring failure mode `method-discipline.md` §6 exists to prevent, and "test on full data, try changes" can quietly burn the eval set (rigor §2) and rabbit-hole (`method-discipline.md` §6 stop rule). So the policy is the **disciplined** version: *diagnosis-first, pre-registered, eval-protected, feasibility-gated.*

### Rules that bound the exploration
- **Diagnosis before change.** On any failure, the first move is a *diagnostic that isolates which assumption broke* — not a fix. Climb the rung that matches the diagnosed cause; do not climb all rungs blindly (`method-discipline.md` §6 anti-rabbit-hole: two same-class fixes → stop and diagnose).
- **Eval-set protection.** All ladder exploration happens on a **validation** split. The post-cutoff eval/test set is touched **once** for the final confirmed claim, never while iterating (§2). This is the guard the "try changes" instinct would otherwise violate.
- **Pre-registration per rung** (`method-discipline.md` §3). Each attempted change states hypothesis → predicted effect (direction *and* magnitude) → falsifier *before* running; movement in the right direction is not success.
- **Feasibility governs "test on full data."** "Confirm at scale" is gated by the §5 go/no-go. If the gate says full-corpus is unaffordable, **inability-to-scale is itself a fundamental failure for this project's 5M+ requirement** — the idea may be valid in general but not feasible here; record it that way rather than shrinking $N$ silently.

### The ladder (climb only the rung the diagnostic points to)
0. **Diagnose.** Power/sample-size? Projection adequacy (WP-15 step-0 separation diagnostic)? Kernel flexibility? Identifiability of $n$ (WP-14)? Slicing operator ill-posed (§1.4)? Pick the matching rung.
1. **Scale.** If the diagnostic suggests a small-sample artifact, retest at the largest *affordable* scale (and full data iff the §5 gate passes). A failure that vanishes at scale was never fundamental.
2. **Projection.** If meme-separation is marginal: frozen PCA → frozen nonlinear → learned $W$ (scored in the frozen space).
3. **Intensity head.** If forecasting loses but the field *does* localize: move along the interpretability⟂flexibility frontier — additive Gaussian (Family A) → flexible encoder (Family B) → the §5 composite — tracking at each step whether the named quantities survive.
4. **SIR inductive bias.** Sweep the PINN-style depletion penalty $\eta$; this both probes accuracy and *measures how much SIR structure the data wants*.

### Definition of "fundamental failure" (checkable — all hold)
1. Does not beat per-cluster SIR **and** the rigor §1 baselines on held-out future data, at the largest affordable scale, *after the diagnosed rungs have been climbed on validation*; **and**
2. either the named quantities are unidentifiable (WP-14 verdict) **or** the field cannot be sliced per-meme under any pre-registered projection/kernel variant; **and**
3. each failed rung's diagnostic pointed to a *structural* cause, or all matched rungs were exhausted without crossing threshold.

Only when all three hold do we invoke Branch A (redirect the machinery to the per-cluster fitter) and file the negative result (§7). Anything short of that is "not yet diagnosed," not "abandoned."
