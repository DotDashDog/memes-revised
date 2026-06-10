# Background

## Project goal

Determine whether — and with what accuracy — the *content* of an internet meme predicts how it spreads online. The end deliverable is a model that, given a meaning-embedding of a meme, predicts the parameters of its interest-over-time curve.

"Meme" here means any viral internet phenomenon, not only text-over-image memes.

## The spread model: SIR (as empirical compression)

We model a meme's interest-over-time curve $I(t)$ using the classical SIR epidemiological model:

$$\frac{dS}{dt} = -\beta I S \qquad \frac{dI}{dt} = \beta I S - \gamma I \qquad \frac{dR}{dt} = \gamma I$$

where $S$, $I$, $R$ are population fractions and $\beta$ (infectiousness), $\gamma$ (recovery rate) are intrinsic to the meme. A scaling parameter $s$ relates the SIR $I(t)$ fraction to the observed signal (post count, weighted by engagement).

**SIR is treated as an empirical compression of the curve shape, not a mechanistic model.** It is justified on the grounds that (a) it fits most observed meme interest curves well, (b) it provides a small, interpretable parameter set, and (c) memes empirically die off to zero, which SIR captures and SIRS/SIS do not. See [`decisions/0002-sir-as-empirical.md`](decisions/0002-sir-as-empirical.md).

The prediction targets are $(\beta, \gamma, s)$ (or some reparameterization with better identifiability; see [`decisions/0010-sir-parameterization.md`](decisions/0010-sir-parameterization.md) when accepted).

## Pipeline overview

1. **Ingest**: Reddit posts (currently r/memes; expandable to top ~40k subs). See [`decisions/0003-reddit-only.md`](decisions/0003-reddit-only.md).
2. **Embed**: Multimodal embedding per post (image + post text + possibly OCR'd in-image text). Pipeline TBD; depends on [`decisions/0004-embedding-contamination.md`](decisions/0004-embedding-contamination.md) and OCR-vs-VLM benchmark.
3. **Cluster**: Group embeddings; each cluster ≈ one meme. Algorithm and granularity TBD.
4. **Aggregate**: Build $I(t)$ for each cluster from post timestamps weighted by engagement.
5. **Fit**: Extract SIR parameters per cluster (likely with a Bayesian / gradient-friendly fitter).
6. **Predict**: Train embedding → parameters model.

## Prior iteration (2023) — what we learned

- Built a dataset of ~1500 memes scraped from KnowYourMeme + Wikipedia, with descriptions embedded.
- Fitted Google Trends interest curves; ~700 memes survived quality filters.
- Trained an NN: ~250% error in log-spread-parameters vs. ~350% for a no-content baseline.

Identified failure modes (carry over to this iteration unless mitigated):

1. **Sparse / noisy Trends data** — many memes had unusable curves.
2. **Few memes; weak descriptions** — single-sentence descriptions, low coverage.
3. **Description-level virality contamination** — the fact of being in KnowYourMeme/Wikipedia is itself a virality signal.
4. **Embedding-model contamination** — the embedder had likely seen meme content (and discussion of meme virality) during pretraining.
5. **Outdated embedders** — embedding quality has improved substantially since.

This iteration aims to address (1, 2) by scraping a large social platform directly, (3) by removing the human-curated description step, and (4, 5) by careful embedding-model selection and post-cutoff evaluation.

## Key risks (live throughout the project)

These are not solved; they're the dominant uncertainties.

- **R1 — Embedding contamination.** Modern embedders have read the post-meme internet. Even pre-cutoff memes are described in training data. Strategy: post-cutoff evaluation, older/smaller embedder, possibly Wikipedia-only-trained models. See [`decisions/0004-embedding-contamination.md`](decisions/0004-embedding-contamination.md).

- **R2 — Selection / endogeneity.** Every meme in our corpus has already been posted at least once and thus is selected for some level of virality. The "non-viral counterfactual" is largely missing. Strategy: include low-upvote posts and broaden to non-meme subreddits to recover failure cases.

- **R3 — Identifiability of SIR parameters.** $(\beta, \gamma, s, I_0)$ are likely degenerate given noisy counting data with finite observation windows. We may be forced to predict only identifiable combinations (e.g., $R_0$, time-to-peak, peak amplitude). Quantify before fitting.

- **R4 — Cluster ≠ meme.** Embedding clusters at different granularities give different "memes" with different spread curves. The clustering choice is a hidden hyperparameter affecting all downstream results.

- **R5 — Posts ≠ infections.** Post counts in a subreddit are a noisy, nonlinearly-related proxy for true meme exposure/transmission. Mediated by mod rules, repost culture, subreddit growth, posting demographics.

## Success metric

**Target: $R^2 \geq 0.3$ on log-spread-parameters in the held-out (post-embedding-cutoff) evaluation set**, beating the strongest baseline (currently expected to be k-NN in embedding space) by a meaningful margin.

This is a deliberately conservative bar. Anything above ~0.3 would be a scientifically interesting result; we are not aiming for engineering-grade prediction. See [`decisions/0001-success-metric.md`](decisions/0001-success-metric.md).

## Speculative directions (not active)

These are tracked but not currently being built. See [`open-questions.md`](open-questions.md).

- **End-to-end differentiable training (PINN-style).** Use a differentiable ODE solver so gradients flow from observed $I(t)$ → SIR ODE → predicted parameters → embedding network. Optionally add a residual neural correction to the SIR ODE.
- **Holistic spatio-temporal point-process model.** Treat the entire stream of posts as a Cox / Hawkes process over the embedding × time space, rather than first clustering and then fitting. Cluster-based SIR is the limiting case under particular assumptions.
- **Meme generation.** Once spread can be predicted, generate memes with target spread profiles.
