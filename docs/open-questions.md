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

## Data ingestion

- **Re-ingest the submissions archive: 63% of it was never processed.** *(Resolved question, actionable work — kept here until it has a WP.)*
Context: The raw `memes_submissions.zst` is verified complete (clean zstd stream, 31.5 GB decompressed, **11,332,165 submissions, 2008-07-11 → 2023-12-31**), but the ingested parquet holds only 4,121,376 rows ending 2020-03-11. **~7.2M submissions covering 2020-03 → 2023-12 are already on disk and simply unprocessed.** Re-ingestion needs no download. A briefly-raised worry that the 2020 ceiling would leave ADR 0004 without a post-cutoff eval window is **withdrawn** — it was an artifact of the partial ingestion; data through 2023-12 postdates OpenCLIP's LAION-2B crawl. Open sub-decisions for the re-ingest: output format (per ADR 0015, shards not loose files), whether to keep the `chunks=10` chronological split, and whether to filter to image posts at ingest time or later.
Owner: unowned. Status: open, high value, cheap. Touch points: ADR 0004, ADR 0011, ADR 0015, WP-05.

- ~~Should we pull the newer Academic Torrents release?~~ **Done 2026-07-26.** The 2025-12 cut (`3e3f64de…`) is downloaded, verified, and is now WP-16's input. It is a faithful superset of the 2023-12 release. Fetcher kept at `experiments/wp16/fetch_archive.py` for the next cut.

- **The recent data is clean but thin — how should ADR 0011 split a corpus whose volume fell 97.4%?**
Context: r/memes went from 415,953 submissions/month (Aug 2020) to 10,796 (Dec 2025); 2024+2025 hold 450,082 submissions against 4.3M in 2020 alone. This creates a direct tension the project has not yet faced: **ADR 0004 wants the eval window as late as possible (past the embedder's training cutoff), while statistical power wants it as early as possible (where the data is).** They point in opposite directions, and more data cannot resolve it — the recent posts simply do not exist in the earlier volume. Compounding factors: fewer posts per meme means noisier $I(t)$; fewer clusters reach the size where an SIR fit is identifiable at all (WP-02 established that low-count regimes are exactly where fits fail); and the viral tail — the phenomenon of interest — shrinks fastest. Options not yet weighed: accept a small eval set and report wide uncertainty; use a *contamination-controlled* embedder instead (already an open question above) so the cutoff constraint relaxes; or broaden to sibling subreddits (ADR 0003 sub-question) to recover volume in recent years.
Owner: unowned. Status: open, **blocks ADR 0011**. Touch points: ADR 0004, ADR 0011, ADR 0003, WP-02 (low-count identifiability), WP-16.

- **`memes_comments.zst` has never been ingested — and it is much bigger than the submissions side.**
Context: Verified complete: **100,181,097 comments, 125.7 GB decompressed, 2009-07-09 → 2023-12-31** — ~8.8 comments per submission. The ingested parquet has zero rows flagged `is_comment`. Whether to ingest them at all depends on the unresolved question above about comments-as-signal vs. comments-as-content, and that decision should be made deliberately rather than inherited from a 2025 run that only touched submissions. Scale is not prohibitive — 100M mostly-short-text rows should land in the low tens of GB as parquet, which fits the NVMe — but it is large enough that the decision should precede the work. Note the `created_utc` string-vs-float inconsistency across the archive's history.
Owner: unowned. Status: open. Touch points: ADR 0007, ADR 0015, WP-05.

- **The image corpus does not exist yet, and fetching it is the real scaling cost.**
Context: All 11.3M submissions carry an `img_url`, but only 2296 images have ever been downloaded. Building the corpus means millions of network fetches, which is network- and rate-limit-bound rather than disk-bound — a different bottleneck from the one ADR 0015 addresses, and probably the single most expensive step in the project. Two things need deciding: how far back to fetch (link rot is severe on pre-2018 posts that point at external hosts like imgur, far milder on Reddit-hosted `i.redd.it`), and what the retry/failure accounting is, since silently dropping dead links biases the corpus toward surviving — i.e. more-viral — posts, which is the R2 selection problem arriving through the back door.
Owner: unowned. Status: open. Touch points: ADR 0015, R2 (selection), WP-05.

## Cluster evaluation

- **How large, and how sampled, must a cluster-quality evaluation set be before its metrics mean anything?**
Context: The 90-image WP-01 clustering check produced weak purity/silhouette, which was initially logged as a property of the embedding pipeline. It is not — at n=90 randomly drawn from a corpus of hundreds of meme families, nearly every true cluster is a singleton, so the metrics are uninformative by construction and would look the same for a perfect embedder. Two things need deciding before the next clustering audit: (a) the minimum sample size, derived from the actual family-size distribution rather than picked by feel; (b) whether to keep sampling uniformly at random or to **sample by family** (e.g. dhash-seeded near-duplicate expansion, so each drawn meme brings its siblings) — the latter gets within-cluster support at far lower cost but biases the cluster-size distribution, which is itself one of the reported metrics. Applies to the pending stronger-text-embedding audit and to WP-04's gold set.
Owner: unowned. Status: open. Touch points: WP-01 (closed but reinterpreted), WP-04, WP-09/ADR 0009.

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

- **Add other meme-style subs?** — **now backed by measurement, still undecided.**
Context: Broadens coverage at the cost of mixing community styles. Researched 2026-07-26 against the torrent manifest and live Arctic Shift counts; full data in `research-notes/supplement-subreddits.md`. Headline: pooling ten general image-meme subs gives **~5.4×** r/memes' current image-post volume, but still leaves recent years ~8× below the 2020 baseline. Note the two names this open question originally proposed both score poorly — **r/dankmemes has collapsed harder than r/memes** (3,608 → 77 posts/day, −97.9%) and **r/wholesomememes is at 9 posts/day**. The strong current candidates are r/whenthe, r/shitposting, r/196, r/meirl, r/meme. **The decision is not "pick the biggest," though:** activity migrated rather than vanished, so the largest subs today (r/whenthe, r/196) barely existed in 2020, which would confound "year" with "community" precisely inside the eval window. Subs that span the whole period (r/meirl, r/dankmemes, r/comedyheaven) are weaker on volume but sounder for a time-split comparison. Unresolved before any ADR: cross-posting overlap (measurable now with `experiments/wp01/dhash.py`), and that pooling breaks ADR 0005's normalized single-population $S(0)=1-I(0)$.
**2026-07-26 update — now carried into a proposed ADR.** `decisions/0016-multi-subreddit-corpus.md` (`proposed`) resolves the "which subs" part as a **backbone** (r/memes, r/dankmemes, r/meirl, r/comedyheaven — span the whole period, composition-matched) plus a held-out **extension** (r/whenthe, r/196, r/shitposting) used only for community-generalization eval, with Tier 2 topic-locked subs excluded from v1. It also carries the mitigations for heterogeneous community size: $N_{c,t}$ as an exposure term (preserving rather than breaking ADR 0005), hierarchical partial pooling, and a factorial split amending ADR 0011. Three preconditions are unrun: per-community-month unique-author counts, dhash cross-post overlap, and a community-shortcut audit.
Owner: unowned. Status: open, sub-decision in ADR 0003; **proposed resolution in ADR 0016**; interacts with ADR 0011 and ADR 0005.
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