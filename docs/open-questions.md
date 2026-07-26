# Open Questions

Questions surfaced during the project that are not yet answered, and don't yet have a work package or ADR. Add new questions here as you encounter them; don't silently answer them. If a question is upgraded to an ADR or WP, link it and leave a stub here pointing to where it lives.

Format per entry:

- **Question:** the question itself.
- **Context:** why it matters / where it came from.
- **Owner / status:** who's thinking about it (or "unowned"); current state (open, parked, promoted-to-X).

---

## Embedding & contamination

- **Are there modern embedding models trained on contamination-controlled corpora (e.g., Wikipedia-only, or pre-meme-era data) that are competitive in quality?**
Context: ADR 0004 chose the time-cutoff strategy with older models, but a clean-corpus modern model would be strictly better. Likely needs a literature/web survey.
Owner: unowned. Status: open. May spawn WP-03.
- **Could adversarial / contrastive debiasing remove virality signal from a contaminated embedder?**
Context: Speculative idea from the design discussion. If feasible, would let us use stronger modern models.
Owner: unowned. Status: parked (low-priority research direction).
- **Contamination probe design:** how exactly do we run "predict virality from embedding without showing the time-series" cleanly?
Context: `01-scientific-rigor.md` §3 mandates this. Concrete plan needed. Likely promoted to WP-11.
Owner: unowned. Status: open.

## Signal construction

- **Best functional form for upvote weighting $w(u, c)$ in the engagement signal?**
Context: ADR 0007 proposes a log default; should be validated against data once available.
Owner: unowned. Status: open, blocks final form of ADR 0007.
- **Are comments better used as a *signal* (contributing to $I(t)$) or as *additional content* (going into the embedding)?**
Context: They carry different information. The discussion suggested embedding-side use is more novel and less contamination-prone, but both are plausible.
Owner: unowned. Status: open, will likely be an ablation in WP-10.

## Modeling & methodology

- **End-to-end differentiable training:** instead of fitting SIR per cluster and then training a predictor, train the predictor end-to-end through a differentiable ODE solver.
Context: Plausibly more robust to identifiability issues and to noise in per-cluster fits. Implementable with diffrax. Probably tried after the two-stage pipeline produces a baseline.
Owner: unowned. Status: parked (revisit after first model claim).
- **Neural-correction SIR:** $\dot I = (\text{SIR RHS}) + f_\theta(I, S, R, t; \text{embedding})$, with $f_\theta$ regularized.
Context: Lets the model deviate from SIR where data demands. A more interesting version of end-to-end training.
Owner: unowned. Status: parked.
- **Predict full posteriors / distributions rather than point estimates.**
Context: `01-scientific-rigor.md` §5 wants this. Implementation detail TBD; likely Gaussian NLL on log-parameters as a v1.
Owner: unowned. Status: open, will be a sub-decision when WP-10 starts.
- **Pre-training on simulated SIR data.**
Context: Generate synthetic embeddings + SIR curves, pre-train, then fine-tune. Cheap data-efficiency win.
Owner: unowned. Status: parked (try after baselines if v1 is data-starved).

## Single-seed regime & extinction

- **The mean-field / SIR description is weakest exactly in the single-seed (I₀=1) regime that real memes live in.**
Context: WP-12 certified that the stochastic HawkesN simulator converges to the mean-field (≡ SIR) ODE — but **only with the initial fraction `I₀/N=ε₀` held fixed** as N→∞. Real cascades start from a single seed (I₀=1), where I₀/N→0 is a *different, harder* limit: large take-off-time jitter, a non-negligible probability of early **extinction**, and survival-conditioning bias. In that regime a per-cluster SIR/HawkesN fit is fitting a curve whose mean-field justification is much weaker, and the corpus is selected for survival (only memes that *didn't* go extinct are observed — directly the R2 endogeneity/selection problem). Likely recurring: extinction/survival selection will resurface in per-cluster fitting (WP-08), identifiability (WP-14), and any final-size or popularity prediction.
Owner: unowned. Status: open. Touch points: WP-08, WP-14, R2 (selection), R5 (posts≠infections). Distilled evidence in `experiments/WP-12/report.md` §2 (Claim B) and `claimb_report.md`.

## Causal / selection

- **Causal-inference framing for the endogeneity problem.**
Context: Inverse-propensity weighting or similar could partially correct for "only-viral-memes-exist" selection. Speculative; may or may not be tractable.
Owner: unowned. Status: parked.

## Multi-hump / event-driven memes

- **How to handle multi-wave memes — filter, fit largest wave, or fit a mixture?**
Context: ~15% of memes in prior iteration. Filtering is simplest; mixture is principled but risks overfitting.
Owner: unowned. Status: open, will need an ADR before WP-08 finalizes its fitter.
- **How to distinguish event-driven memes from organic-spread memes a priori?**
Context: Their dynamics are likely different enough that mixing them hurts model training. No clear automated detector.
Owner: unowned. Status: open.

## Scope & data

- **Add other meme-style subs (r/dankmemes, r/wholesomememes)?**
Context: Broadens coverage at the cost of mixing community styles.
Owner: unowned. Status: open, sub-decision in ADR 0003.
- **Which non-meme control subs?** (Cf. ADR 0008.)
Owner: unowned. Status: open, sub-decision in ADR 0008.
- **Cross-platform spread (Twitter/TikTok/Discord) data sources.**
Context: Out of scope per ADR 0003 unless data becomes accessible. Tracked here so it isn't lost.
Owner: unowned. Status: parked.

## Long-term / speculative

- **Holistic spatio-temporal point-process model** (Cox / Hawkes process over the embedding × time space, with an SIR-style local prior as a soft constraint).
Context: Discussed in `00-background.md` "Speculative directions." Cluster-based pipeline is the limiting case under specific assumptions. Worth keeping in peripheral vision so present-day pipeline choices don't preclude it. **Concepts/literature: `research-notes/hawkes-processes.md`. Critique + phased replication plan: `research-notes/hawkes-replication-plan.md`.**
Owner: unowned. Status: **promoted to a phased track** — WP-12…WP-15, scoped by ADR 0014 (`proposed`). WP-12–14 run in parallel with WP-02; WP-15 stays gated on the cluster baseline. Note: keep raw timestamps + embeddings throughout the pipeline so this remains tractable later.

### Sub-questions surfaced by the Hawkes replication plan (`research-notes/hawkes-replication-plan.md` §7)

- **Per-meme forecast extraction:** what operator turns the holistic intensity $\lambda^*(t,\mathbf{x})$ into "meme $m$'s future trajectory," and is it well-posed when the spatial kernel doesn't cleanly localize memes? Status: open, tested in WP-15.
- **Windowing — modeling assumption or approximation?** At what time/embedding radius does truncating influence change the fitted branching ratio by more than its posterior width? Status: open, measured in WP-14; gates the 5M+ feasibility extrapolation.
- **Repost/deletion correction:** principled MNAR correction (thinning model, dedup prior) or only a robustness ablation? Status: open, ablation in WP-14.
- **Mean-field equivalence under the real signal:** does cluster-SIR↔HawkesN hold under the ADR 0007 engagement signal, or only for raw counts? Status: open.
- **Meme generation conditioned on target spread parameters.**
Context: The aspirational long-term application. Out of scope until the prediction problem is solved.
Owner: unowned. Status: parked.