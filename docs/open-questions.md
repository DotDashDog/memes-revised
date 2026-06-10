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
Context: Discussed in `00-background.md` "Speculative directions." Cluster-based pipeline is the limiting case under specific assumptions. Worth keeping in peripheral vision so present-day pipeline choices don't preclude it.
Owner: unowned. Status: parked. Note: keep raw timestamps + embeddings throughout the pipeline so this remains tractable later.
- **Meme generation conditioned on target spread parameters.**
Context: The aspirational long-term application. Out of scope until the prediction problem is solved.
Owner: unowned. Status: parked.