# Project State

**Last updated:** 2026-07-26 (agent-protocol loading fixed for Claude Code)

This document is the live snapshot. Update it after any substantive change.

## Active focus

The project is in the **bootstrap phase**. WP-01 is complete; WP-02 remains active.

- [WP-02: SIR identifiability study on synthetic data](work-packages/WP-02-sir-identifiability-study.md) — `status: active`

ADR 0006 and ADR 0010 are now accepted. The next embedding work should scale the accepted hybrid pipeline and audit semantic clustering quality with stronger text embeddings.

## What is decided

See `decisions/` for the full ADR set. Accepted at bootstrap:

- 0001 — Success metric: $R^2 \geq 0.3$ on log-spread-parameters on post-cutoff eval set.
- 0002 — SIR treated as empirical curve compression, not mechanistic.
- 0003 — Reddit-only as initial platform; r/memes as first subreddit.
- 0004 — Embedding contamination mitigated via time-cutoff evaluation; final cutoff window pending ADR 0011.
- 0005 — SIR initial conditions fixed: $S(0) = 1 - I(0)$, $R(0) = 0$.
- 0006 — Embedding pipeline: hybrid VLM image embedding + filtered EasyOCR text embedding.
- 0010 — SIR parameterization for prediction: primary targets are stable curve summaries; transformed SIR parameters are auxiliary diagnostic targets with identifiability metadata.
- 0013 — JAX as the primary numerical backend for new research code; predictor framework choice remains open.

## What is NOT decided

These ADRs exist as `proposed` placeholders and are gated by experiments:

- 0007 — Engagement signal definition: how to convert posts → $I(t)$.
- 0008 — Non-viral baseline data source.
- 0009 — Clustering algorithm and granularity.
- 0011 — Train/test/eval split strategy.
- 0012 — Baseline set (the canonical list is in `01-scientific-rigor.md` §1; this ADR will lock specifics for the first model run).

## Environment

- Package manager: **uv** (see `03-environment.md`).
- Python: pinned at **3.12** via `.python-version`.
- `pyproject.toml` and `uv.lock` now include WP-02 runtime dependencies: `jax`, `diffrax`, `numpyro`, `corner`, `matplotlib`, `scipy`; WP-01 runtime dependencies: `torch`, `transformers`, `open-clip-torch`, `pandas`, `pyarrow`, `opencv-python-headless`, `easyocr`, `pytesseract`, `paddleocr`, `paddlepaddle`, `sentence-transformers`, `openai`, `umap-learn`, `hdbscan`, `scikit-learn`, `pillow`, `requests`, `tqdm`; dev dependency: `pytest`.
- Local hardware: NVIDIA 3090 GPU available.

## Data inventory

| Asset                           | Location                                   | Coverage           | Notes                                                                                 |
| ------------------------------- | ------------------------------------------ | ------------------ | ------------------------------------------------------------------------------------- |
| r/memes post archive            | `reddit_data/` (existing)                  | up to ~early-2020s | Inherited from prior iteration; format and integrity should be verified before use    |
| Top-40k-subreddits archive      | external (described in `00-background.md`) | TBD                | Not yet ingested                                                                      |
| Previous-iteration meme dataset | `dataset_saves/` (existing)                | ~700 fitted curves | From the 2023 iteration; useful as a sanity-check reference, not as a training source |
| Hardware                        | local 3090 GPU                             | —                  | Sufficient for local embedding/inference at moderate scale                            |

The legacy code in this repo (`memes_base/`, `train_script.py`, `test_ocr.py`, etc.) predates this framework. **It should not be assumed correct or representative of current decisions.** Treat it as reference material; reuse code only after reading and understanding it.

## Models

| Model            | Status                | Version / weights                         | Notes                                                |
| ---------------- | --------------------- | ----------------------------------------- | ---------------------------------------------------- |
| Embedding model  | chosen for first pass | OpenCLIP ViT-B-16 + filtered EasyOCR      | ADR 0006 accepted hybrid image/OCR pipeline          |
| Text embeddings  | under comparison      | MiniLM L6 v2, BGE base en v1.5, E5 base v2 | Cached on 1000 examples for semantic clustering work |
| Spread predictor | not yet trained       | —                                         | Architecture deferred until embedding pipeline lands |

## Eval set

Not yet defined. Will be locked in ADR 0011. Default expectation: held-out memes originating after the chosen embedding model's training cutoff.

## Recent changes log

Add a one-line entry every time you change the state of the project. Newest at top.

- 2026-07-26 — **Corrected a false tooling assumption: Claude Code never loaded the protocol.** The 2026-06-10 move to `/AGENTS.md` assumed Claude Code reads `AGENTS.md`. It does not — it auto-loads `CLAUDE.md` / `CLAUDE.local.md` only (verified empirically on CLI 2.1.220 by querying a session with the file-reading tools disabled; the sole `AGENTS.md` handling in the binary is the Codex importer, which converts it to `CLAUDE.md`). Every Claude Code session in this repo since 2026-06-10 therefore ran **without** the agent protocol loaded — including the anti-self-certification rules — unless it happened to open the file on its own. Fixed by adding `/CLAUDE.md`, which imports `@AGENTS.md`; `AGENTS.md` remains the single tool-neutral source of truth and Cursor is unaffected. Also added `.claude/settings.json` (permission allowlist; `uv run python` deliberately left prompting so §"surface the criterion before the expensive run" still bites), untracked `.DS_Store`, and gitignored `.claude/settings.local.json`. **Treat any Claude-Code-produced result recorded between 2026-06-10 and 2026-07-26 as having been generated without the protocol in context.**
- 2026-05-19 — Built a cached 1000-example r/memes semantic embedding set with downloaded images, filtered EasyOCR OCR text, OpenCLIP image embeddings, and local MiniLM/BGE/E5 sentence embeddings; OpenAI embedding skipped pending API key.
- 2026-05-19 — Ran a 90-image cached WP-01 clustering check; HDBSCAN now forms non-noise clusters, but proxy purity/silhouette remain weak, so hand-labeled semantic evaluation is still needed before relying on clustering metrics.
- 2026-05-19 — Completed WP-01 and accepted ADR 0006: use hybrid OpenCLIP image embeddings plus filtered EasyOCR (`easyocr_filtered:none`) transcript embeddings; larger OCR/cluster audit remains follow-up work.
- 2026-05-19 — Ran WP-01 cached real r/memes pilot on 24 manually transcribed images; OpenCLIP B/16 image→text top-1 was weak (16.7%) and clustering produced no non-noise clusters, so ADR 0006 remains proposed.
- 2026-05-19 — Implemented WP-01 text-fidelity benchmark harness and ran a synthetic Impact-caption smoke test; ADR 0006 remains proposed and blocked on real r/memes + manual-transcript evidence.
- 2026-05-19 — User accepted ADR 0010.
- 2026-05-19 — Drafted ADR 0010 recommendation: primary prediction/evaluation targets should be stable curve summaries; transformed SIR parameters remain auxiliary diagnostic targets with identifiability metadata.
- 2026-05-19 — Recorded WP-02 first-pass conclusion: SIR curve fitting works for posterior-predictive curves, but natural parameters are not always identifiable; near-threshold and low-count regimes should be flagged rather than trusted as clean labels.
- 2026-05-19 — Ran WP-02 81-case Hessian grid and six selected NUTS validations; Hessian works in well-identified regimes and fails in low-count/ridge regimes. Low-count uncertainty threshold study added as optional next WP-02 step.
- 2026-05-19 — Accepted ADR 0013 (JAX-first numerical backend). Reduced representative NUTS check default to 4 sequential chains with 500 warmup + 500 retained samples per chain.
- 2026-05-19 — Started WP-02. Added JAX/diffrax SIR forward model, Poisson binned-count likelihood, MLE/Hessian scaffold, optional NumPyro/NUTS posterior path, corner-plot generation, and smoke tests.
- 2026-06-10 — Moved the agent protocol from `.cursor/rules/project-protocol.mdc` to `/AGENTS.md` for tool-neutral compatibility (Cursor + Claude Code + others). Deleted the Cursor rule; `ocr-research.mdc` retained as a contextual rule. **[Correction, 2026-07-26: the Claude Code half of this claim was false — Claude Code does not read `AGENTS.md`. See the 2026-07-26 entry at the top of this log.]**
- 2026-06-11 — **Bridge Claim B certified (scoped).** Stochastic HawkesN→mean-field convergence certified for the `I₀/N=ε₀`-fixed family (Sonnet subagent; orchestrator inspected plots+exponents, user-concurred). Mean-bias O(1/N), single-run std ≈N^(−1/2), phase-lead closes with N. **Both bridge claims (A analytic identity, B stochastic convergence) now certified.** Scope excludes single-seed I₀=1 → new open question logged (extinction/take-off-jitter make the SIR/mean-field picture weakest exactly in the real-meme regime; bears on WP-08/WP-14, ties to R2 selection). Remaining WP-12: parameter recovery/identifiability (flagged K–θ ridge), size-distribution prior→posterior figure, Phase B real-data fits.
- 2026-06-11 — **Bridge Claim A certified; method-discipline §2 extended.** Deterministic HawkesN-MF-ODE vs SIR-ODE test (Sonnet subagent) certified the analytic identity's *implementation* (ε≈1e-6, at solver floor; orchestrator inspected plot+code for non-circularity, user-concurred). The subagent followed the strengthened rules cleanly and did not self-certify — a positive test of the safeguards. Added the "name which link you are testing" rule to `method-discipline.md` §2. Open bridge item is now **Claim B** (stochastic simulator → mean-field convergence at large N), the link the original withdrawn test conflated.
- 2026-06-10 — **Bridge-test post-mortem & safeguards.** Withdrew the WP-12 mean-field-bridge "PASS" (mis-specified test: deterministic identity vs finite-N stochastic average, phase-blind peak-normalized RMSE, result-dependent cond/uncond selection; the plot contradicted the number). Added structural anti-self-certification rules to `AGENTS.md` (no agent-certified passes; surface the criterion before recorded runs; route plots + expected-appearance to the grader). Split the rigor doc: `01-scientific-rigor.md` now holds project-specific standards only; new **`method-discipline.md`** holds the general anti-cargo-cult discipline (identifiability, match-test-to-claim, principled pre-registration, name-the-blind-spot, plots-beat-metrics, debugging) and is high-priority required reading. Section numbers preserved for citation stability (gaps §4/§8/§9/§11 in the project file); all `rigor §N` citations to moved sections repointed.
- 2026-06-10 — Started WP-12 (`active`). Phase A (R-independent) done: JAX HawkesN core + Branch-A single-stream fitter; **mean-field bridge NOT established** (the RMSE "pass" was withdrawn on review — mis-specified: deterministic identity vs finite-N stochastic average, phase-blind metric, result-dependent metric selection; correct deterministic ODE-vs-ODE test pending); final-size criticality reproduced (prior→posterior collapse figure still partial); **synthetic recovery flagged** — β=Kθ recovers but K,θ individually pin at the K=20 bound (K–θ ridge; profile-likelihood diagnostic needed before calling it identifiability vs. fitter pathology). Phase B (upstream verbatim R run) NOT done — R uninstalled, `brew install r` interrupted by network drop; faithful-replication claim outstanding. Artifacts in `experiments/WP-12/` (`report.md`). Recovered from disk after the background agent was interrupted.
- 2026-06-10 — ADR 0014 accepted. Resolved its two open sub-questions: (1) WP-15 negative-result fallback = **Branch A** (redirect the differentiable HawkesN core to a per-cluster fitter feeding WP-08, not abandon) — so WP-12/13 must build a reusable `(event_times[,marks])→params+uncertainty` single-stream interface; the track is not abandoned on first failure but only on *fundamental* failure per the new escalation ladder. (2) Frozen mark space = **L2-normalized PCA** (matching cosine geometry), with adequacy measured up front in WP-15 via a same-meme-vs-cross-meme separation diagnostic on WP-13's hash-labeled streams. Added plan-note §8 (diagnosis-first, eval-protected, feasibility-gated stopping rule).
- 2026-06-10 — Promoted the holistic point-process idea to a phased **neural-Hawkes track**: added `research-notes/hawkes-replication-plan.md` (critique + plan), ADR 0014 (scoping decisions: anchor = SIR-Hawkes/HawkesN, marks temporal-first in a frozen low-d projection, upstream+JAX-core, parallel synthetic scheduling), and draft work packages WP-12 (SIR-Hawkes replication), WP-13 (temporal-only neural TPP + single-meme stream), WP-14 (Hawkes identifiability/endogeneity), WP-15 (first marked STPP, gated on cluster baseline). WP-12–14 run in parallel with WP-02.
- 2026-06-10 — Added `docs/research-notes/` and distilled note `hawkes-processes.md` (speculative; feeds the holistic point-process open question). No change to active plan.
- 2026-05-19 — Adopted uv for package management. Bootstrapped `pyproject.toml`, `.python-version` (3.12), added `docs/03-environment.md` and updated protocol rule.
- 2026-05-19 — Framework bootstrapped (`docs/`, ADRs 0001–0012, WP-01, WP-02).
