# Work Packages

A work package is a unit of work with a clear "done" condition, written so it can be picked
up without holding the whole project in mind.

- Keep the `Status:` header current, and log results, surprises, and dead ends in the
  package's **Log** section. Negative results are part of the deliverable.
- Work that falls outside the package becomes a new package (status `open`) or an entry in
  `../open-questions.md`, so the user can see scope changing.

## Status values

- **open** — not yet started.
- **active** — currently being worked on.
- **blocked** — waiting on an external decision, person, or other package.
- **done** — completed, results documented, ADRs updated.
- **superseded** — replaced; preserved for history.

## Status table


| #                                           | Title                                                                   | Status  | Gates               | Unlocks                        |
| ------------------------------------------- | ----------------------------------------------------------------------- | ------- | ------------------- | ------------------------------ |
| [WP-01](WP-01-ocr-vs-vlm-benchmark.md)      | OCR vs. VLM-native embedding benchmark                                  | done    | —                   | ADR 0006                       |
| [WP-02](WP-02-sir-identifiability-study.md) | SIR identifiability on synthetic data                                   | active  | —                   | ADR 0010                       |
| WP-03                                       | Embedding-model survey (incl. contamination-controlled candidates)      | planned | —                   | ADR 0006 (refines)             |
| [WP-04](WP-04-clustering-gold-set-audit.md) | Gold-set construction + powered clustering audit                        | active  | ADR 0015, ADR 0017  | ADR 0009 validation, WP-06     |
| WP-05                                       | Full embedding pipeline (chosen model + multimodal combine)             | planned | ADR 0006            | clustering                     |
| WP-06                                       | Clustering pipeline + multi-granularity eval                            | planned | WP-05               | ADR 0009                       |
| WP-07                                       | Engagement-signal construction + sub-activity normalization             | planned | ~~data ingest~~ unblocked | ADR 0007                 |
| WP-08                                       | SIR fitter on real cluster data + identifiability check on real fits    | planned | WP-06, WP-07, WP-02 | predictor training             |
| WP-09                                       | Baselines (B1–B5) implementation                                        | planned | WP-08               | first model claim              |
| WP-10                                       | Predictor model v1 (embedding → params)                                 | planned | WP-08, WP-09        | success metric eval            |
| WP-11                                       | Contamination probe (predict params from embedding without time-series) | planned | embeddings exist    | confidence in any model result |
| [WP-12](WP-12-sir-hawkes-replication.md)    | SIR-Hawkes / HawkesN replication (anchor)                               | open    | ADR 0014            | WP-13, WP-14; bridges WP-02    |
| [WP-13](WP-13-temporal-neural-tpp.md)       | Temporal-only neural TPP machinery + single-meme stream                 | open    | ADR 0014, WP-12     | WP-14, WP-15; §5 cost input    |
| [WP-14](WP-14-hawkes-identifiability.md)    | Hawkes identifiability & endogeneity study (synthetic)                  | open    | ADR 0014, WP-12     | real-data Hawkes param claims  |
| [WP-15](WP-15-marked-stpp-pilot.md)         | First marked STPP on a small embedding subset                           | open    | ADR 0014, WP-13/14, cluster baseline | holistic-track go/no-go |
| [WP-16](WP-16-submissions-reingest.md)      | Full submissions re-ingest (11.78M rows, 2008–2025)                     | **complete** | —              | WP-05, WP-06, WP-07; ADR 0011 |
| WP-17                                       | Image corpus fetch into ADR 0015 shards                                 | planned | WP-16, 2025-12 archive cut | WP-05                   |


Planned packages are stubs until promoted to `open`; they may merge, split, or change shape.

WP-12…WP-15 form the **neural-Hawkes track** (`../research-notes/hawkes-replication-plan.md`, scoped by ADR 0014). WP-12–14 are synthetic / single-stream and run in parallel with WP-02; only WP-15's real-embedding spatial work is gated on the cluster pipeline.

## Dependency graph (high level)

```
WP-01 ──► ADR 0006 ──► WP-05 ──► WP-06 ──► WP-08 ──► WP-09 ──► WP-10
                                            ▲                     ▲
WP-02 ──► ADR 0010 ─────────────────────────┘                     │
                                                                  │
WP-03 ──► ADR 0006 (refines)                                      │
WP-04 ──► WP-06 validation                                        │
WP-07 ──► WP-08                                                   │
WP-11 ──► confidence check ───────────────────────────────────────┘
```

WP-01 and WP-02 have no gates and can be done in parallel.

Neural-Hawkes track (parallel synthetic, gated only on ADR 0014):

```
ADR 0014 ──► WP-12 ──► WP-13 ──► WP-15 (also gated on cluster baseline)
                 └────► WP-14 ──┘
WP-02 ───────────────► (bridges into WP-12's mean-field check)
```

## Template

Copy into a new file `WP-NN-short-slug.md`:

```markdown
# WP-NN — <Title>

**Status:** open
**Date opened:** YYYY-MM-DD
**Owner:** <agent or user name>
**Gates:** <work packages or accepted ADRs required before starting>
**Unlocks:** <which ADRs or work packages this enables>

## Goal

<One paragraph. What question does this package answer, or what artifact does it produce?>

## Definition of done

<Concrete, checkable. E.g., "A markdown report in `experiments/WP-NN/` with metrics for each pipeline, and a recommendation noted in this work package's `Outcome` section.">

## Inputs / prerequisites

<Data, decisions, models, code that must exist before this can start.>

## Method

<The approach, and the reasoning behind the choices that are not obvious.>

## Risks and ambiguities

<What could go wrong? What sub-decisions might come up that require an ADR?>

## Out of scope

<Things this package will explicitly not address, so the agent doesn't drift.>

## Log

<Newest at top. Date entries. Record progress, surprises, dead ends, and links to artifacts.>

## Outcome

<Filled in when done. Summary of result and links to the ADR(s) the outcome informs.>
```

