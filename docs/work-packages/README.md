# Work Packages

A work package is a **self-contained unit of work** with a clear "done" condition. Anyone — agent or human — should be able to pick up an open work package, read it, and execute it without holding the whole project in working memory.

## Rules

- One agent works one package at a time. Stay in scope.
- Update the package's `Status:` header whenever it changes.
- Log results, surprises, and dead ends in the package's **Log** section. Negative results are part of the deliverable.
- If a package spawns new work, draft a new work package (status `open`) and cross-reference. **Do not silently expand scope.**
- If a package requires a design decision not already covered by an `accepted` ADR, draft a `proposed` ADR and stop until the user accepts.

## Status values

- **open** — not yet started.
- **active** — currently being worked on (only one per agent).
- **blocked** — waiting on an external decision, person, or other package.
- **done** — completed, results documented, ADRs updated.
- **superseded** — replaced; preserved for history.

## Status table


| #                                           | Title                                                                   | Status  | Gates               | Unlocks                        |
| ------------------------------------------- | ----------------------------------------------------------------------- | ------- | ------------------- | ------------------------------ |
| [WP-01](WP-01-ocr-vs-vlm-benchmark.md)      | OCR vs. VLM-native embedding benchmark                                  | done    | —                   | ADR 0006                       |
| [WP-02](WP-02-sir-identifiability-study.md) | SIR identifiability on synthetic data                                   | active  | —                   | ADR 0010                       |
| WP-03                                       | Embedding-model survey (incl. contamination-controlled candidates)      | planned | —                   | ADR 0006 (refines)             |
| WP-04                                       | Gold-set construction (~50 hand-labeled memes)                          | planned | —                   | ADR 0009 validation            |
| WP-05                                       | Full embedding pipeline (chosen model + multimodal combine)             | planned | ADR 0006            | clustering                     |
| WP-06                                       | Clustering pipeline + multi-granularity eval                            | planned | WP-05               | ADR 0009                       |
| WP-07                                       | Engagement-signal construction + sub-activity normalization             | planned | data ingest         | ADR 0007                       |
| WP-08                                       | SIR fitter on real cluster data + identifiability check on real fits    | planned | WP-06, WP-07, WP-02 | predictor training             |
| WP-09                                       | Baselines (B1–B5) implementation                                        | planned | WP-08               | first model claim              |
| WP-10                                       | Predictor model v1 (embedding → params)                                 | planned | WP-08, WP-09        | success metric eval            |
| WP-11                                       | Contamination probe (predict params from embedding without time-series) | planned | embeddings exist    | confidence in any model result |


Planned packages are stubs until promoted to `open`; they may merge, split, or change shape.

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

<Step-by-step plan, at the level a less-capable agent could follow.>

## Risks and ambiguities

<What could go wrong? What sub-decisions might come up that require an ADR?>

## Out of scope

<Things this package will explicitly not address, so the agent doesn't drift.>

## Log

<Newest at top. Date entries. Record progress, surprises, dead ends, and links to artifacts.>

## Outcome

<Filled in when done. Summary of result and links to the ADR(s) the outcome informs.>
```

