# 0014 — Neural-Hawkes track: scope & architecture-critical decisions

**Status:** approved
**Date:** 2026-06-10
**Gates:** none to draft; informs WP-12…WP-15
**Feeds from:** `../research-notes/hawkes-replication-plan.md`, `../research-notes/hawkes-processes.md`

## Context

The "holistic spatio-temporal point-process" open question is being promoted from a speculative note to a concrete, staged track. A literature pass (recorded in the plan note) found that **no existing paper does interpretable, marked neural Hawkes on a high-dimensional embedding mark at $5\text{M}+$ events**, so the work must assemble validated building blocks rather than replicate one paper end-to-end. Four architecture-critical decisions were needed before drafting work packages; the user chose all four at kickoff. This ADR records them.

## Decision

1. **Anchor paper = SIR-Hawkes / HawkesN (Rizoiu et al., WWW 2018).** It is the concrete SIR↔Hawkes bridge, connects to WP-02 / ADR 0010, and (being a *depleting* finite-population Hawkes) reproduces the rise-peak-fall dynamics that linear Hawkes cannot. Replication starts from the upstream `computationalmedia/sir-hawkes` tutorial.

2. **Marks are staged temporal-first, then introduced in a *frozen* low-dimensional projection.** Phases 1–3 are temporal-only. When marks enter (Phase 4), every model is evaluated in one frozen projection (default: PCA to 8–16 d, fit on training-period embeddings only) so held-out NLLs are comparable across models. Learned projections ($W$ in the §5 note architecture) are admitted later only as **ablations scored in that same frozen space**, never as the primary comparison axis, to keep the projection off the contamination/leak surface.

3. **Code strategy = run upstream as-is + build our own JAX core.** Upstream code (R for SIR-Hawkes, PyTorch/EasyTPP for neural TPP) is run unmodified to establish the faithful-replication claim; we additionally build a minimal JAX Hawkes/STPP core (per ADR 0013) validated against both upstream fits and synthetic ground truth. This separates "paper discrepancy" from "our implementation bug." **ADR 0013 (JAX-first) is not amended**; upstream stacks are reference only.

4. **Scheduling = parallel synthetic track.** WP-12…WP-14 (replication, neural machinery, identifiability) run in parallel with WP-02 and need no embeddings or clusters. Only WP-15's *real-embedding spatial* work remains gated on the cluster pipeline producing a baseline. The `hawkes-processes.md` guard ("do not build this before the cluster pipeline produces a baseline") is hereby narrowed to apply to real-embedding spatial work only.

## Alternatives considered

- **Anchor on Neural STPP / Chen et al. (the note's original "start here").** Rejected as anchor: the repo is archived (Oct 2023), uses continuous-time normalizing flows that **discard the named kernel** (opposing the dual-interpretability goal), and handles only low-d marks. Retained as the reference for Phase 4 marked-STPP machinery, not as the interpretable anchor.
- **EasyTPP-first / PyTorch-standardize the track.** Rejected: would diverge from ADR 0013 and defer the SIR bridge that motivates the whole track. EasyTPP is used as a *validation reference* in Phase 2 instead.
- **Learned projection from the start (note §5).** Rejected for the primary comparison: makes cross-model NLLs incomparable and adds a leak surface. Deferred to ablations.
- **Build the §5 composite architecture directly.** Rejected: it is an unreplicated invention; building it before its assumptions are validated violates the replicate-first directive.

## Consequences

- Spawns WP-12 (SIR-Hawkes replication), WP-13 (temporal-only neural machinery), WP-14 (Hawkes identifiability/endogeneity), WP-15 (first marked STPP). WP-12–14 are unblocked now; WP-15 gates on the cluster baseline.
- A **5M+ feasibility go/no-go gate** (plan note §5) must be cleared before any full-corpus run, using measured per-event cost from WP-13.
- Every track claim reports against the rigor §1 baselines **plus** independent per-cluster SIR, and reports forecasting and interpretability/identifiability separately (plan note §2).

## Open sub-questions

- ~~If Phase 4 shows the holistic field cannot be cleanly sliced per-meme, is the track redirected rather than abandoned?~~ **Resolved 2026-06-10 (user):** yes — **Branch A (redirect, don't abandon)** is the committed fallback. A failed holistic model still yields a differentiable HawkesN likelihood that becomes a better per-cluster fitter than bin-counts→SIR-ODE, flowing back into WP-08. Consequence: the WP-12/13 JAX core is built with a reusable "single cluster's event times → fitted params + uncertainty" interface from day one. Further, the track is **not abandoned on a first failure** — failure must be shown *fundamental* per the pre-registered, diagnosis-first, eval-protected, feasibility-gated escalation ladder in `../research-notes/hawkes-replication-plan.md` §8.
- ~~Does the frozen-PCA projection retain enough meme-separating structure?~~ **Resolved 2026-06-10:** frozen projection stays the default (user-chosen), refined to **L2-normalized PCA** so the kernel geometry matches the cosine geometry the embeddings are trained for. Adequacy is no longer assumed — WP-15 measures it up front via a same-meme-vs-cross-meme separation diagnostic on WP-13's hash-labeled streams, falling back to a frozen *nonlinear* projection only if that diagnostic fails. Decision 2 above should be read as "frozen, L2-normalized PCA (Euclidean-Gaussian kernel) or a vMF kernel on the normalized sphere," with the choice made in WP-15 step 0.
