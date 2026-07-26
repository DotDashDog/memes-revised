# WP-15 — First marked STPP on a small embedding subset

**Status:** open (draft; gated on cluster baseline existing)
**Date opened:** 2026-06-10
**Owner:** unassigned
**Gates:** [ADR 0014](../decisions/0014-hawkes-track-scope.md) (accepted), WP-13, WP-14, and a cluster-SIR baseline (WP-06/WP-08 outputs)
**Unlocks:** decision on whether the holistic STPP earns its complexity; a possible architecture ADR for the `hawkes-processes.md` §5 composite

## Goal

The first **marked** model: a small, faithful spatio-temporal point process with an analytic spatial kernel on a *frozen* low-dim projection of real embeddings, on a small subset. Answer the two decisive questions: does it beat independent per-cluster SIR at held-out future likelihood, and can its learned field be sliced per-meme and forecast forward (plan note §1.4)?

## Definition of done

A report at `experiments/WP-15/report.md` with:
0. **Projection sanity check (precondition, do first).** Before building anything, verify the frozen mark space actually preserves *meme-separating* structure, not just variance. **L2-normalize embeddings before PCA** (cosine is the geometry CLIP-style spaces are trained for; PCA + a Gaussian kernel otherwise assumes Euclidean geometry that may not match). Then, using WP-13's image-hash-labeled meme streams as ground-truth same-meme groups, confirm that same-meme posts sit closer than cross-meme posts in the frozen 8–16 d space (e.g. silhouette / same-vs-cross distance ratio). If this fails, **switch to a frozen nonlinear projection** (UMAP or a frozen pretrained projection head) before proceeding — do not build the STPP on a projection that can't separate memes. Also decide here whether the kernel is Euclidean-Gaussian on normalized PCA coords or a von Mises–Fisher kernel on the sphere, and state which geometry it assumes.
1. **Marked STPP fit** on a small embedding subset, spatial kernel = Gaussian (closed-form spatial compensator), marks = the frozen projection validated in step 0 (default: L2-normalized PCA 8–16 d, fit on training-period embeddings only, per ADR 0014).
2. **Decisive test 1 — forecasting.** Held-out *future* log-likelihood vs. **independent per-cluster SIR** and vs. rigor §1 baselines (B1–B5). Bootstrap CIs.
3. **Decisive test 2 — per-meme slicing.** Define the operator that conditions $\lambda^*(t,\mathbf{x})$ on a meme's embedding sub-region and integrates forward; produce per-meme trajectory forecasts; compare to per-cluster SIR and baselines. Report whether the slicing is well-posed (does $s_\theta$ localize memes?).
4. **Contamination probe** (rigor §3): can spread be predicted from the frozen mark *without* the time-series? If high, the embedding already encodes virality.
5. **Verdict**, following the plan-note §8 stopping rule: a first failure triggers the diagnosis-first escalation ladder (on validation), not abandonment; a negative result is filed (rigor §7) only when failure is *fundamental* by §8's checkable definition, at which point Branch A redirects the machinery to the per-cluster fitter.

## Inputs / prerequisites

- ADR 0014 accepted; WP-13 (machinery), WP-14 (which quantities are safe + windowing validity).
- A real embedding subset with timestamps **and** a cluster-SIR baseline to compare against (from the WP-05/06/08 pipeline). This is the gated dependency.
- JAX core extended with an analytic spatial kernel.

## Method

1. Take a small, time-bounded embedding subset; L2-normalize; fit the frozen PCA projection on the training period only; run the DoD-step-0 separation diagnostic before going further.
2. Fit the Gaussian-kernel marked STPP; validate the spatial compensator is closed-form and correct against MC.
3. Run both decisive tests and the contamination probe.
4. **If both clear:** scope an ablation introducing a learned projection $W$ *scored in the frozen space*, and scope the §5 full-corpus feasibility extrapolation. **If either fails:** do *not* declare defeat — enter the diagnosis-first escalation ladder (plan note §8). Only when failure is *fundamental* by that section's checkable definition do we invoke **Branch A** (redirect the differentiable HawkesN core to the per-cluster fitter, feeding WP-08) and file the negative result (rigor §7). All ladder exploration is on a validation split; the post-cutoff eval set is touched once for the final claim.

## Risks and ambiguities

- **Per-meme slicing may be ill-posed** if the kernel doesn't localize — this is the central risk, and a clean negative answer is a valid deliverable.
- Frozen PCA may destroy meme-separating structure — but this is now *measured up front* by the DoD-step-0 diagnostic rather than assumed. The risk is real because high-*variance* directions (what PCA keeps) need not be the *meme-separating* directions (what the kernel needs); in CLIP-style spaces top variance is often global style/format, not meme identity. If the diagnostic fails, a frozen nonlinear projection precedes any STPP fit.
- Spatial compensator correctness in 8–16 d must be checked against MC before trusting any NLL.
- Contamination: a strong probe result reframes *all* downstream claims (rigor §10).

## Out of scope

- Full 5M-corpus runs (require the §5 gate to pass first).
- The `hawkes-processes.md` §5 composite (per-content $\beta_\theta$, learned $s_\theta$, PINN SIR penalty) — only scoped if this pilot clears, via a new architecture ADR.

## Log

- 2026-06-10 — Drafted from `research-notes/hawkes-replication-plan.md` Phase 4. Remains gated on the cluster baseline per ADR 0014.

## Outcome

<pending>
