# Research Notes

Distilled, self-contained write-ups of exploratory research that is **not yet** an accepted decision (`decisions/`) or a ready-to-execute work package (`work-packages/`).

A research note exists to let another agent absorb the useful conclusions of an exploration **without** reading the conversation or literature it came from. Each note should:

- State its **status** (speculative / informing-an-open-question / promoted-to-WP-or-ADR).
- Link the `open-questions.md` entry or ADR it feeds.
- Be readable standalone: definitions, conclusions, and a curated citation list — not a transcript.
- Distinguish clearly between what is **established in the literature** and what is **our speculation** for this project.

Notes are not binding. They do not authorize work. When a note matures into a decision or a plan, promote it to an ADR or work package and leave the note as background.

## Index

| Note | Topic | Status | Feeds |
|---|---|---|---|
| [hawkes-processes.md](hawkes-processes.md) | Hawkes & neural spatio-temporal point processes; relation to SIR and to the holistic rate-field idea | speculative | `open-questions.md` → "Holistic spatio-temporal point-process model" |
| [hawkes-replication-plan.md](hawkes-replication-plan.md) | Critique of the Hawkes note + phased replication plan (WP-12…WP-15); 5M+ feasibility gate | plan drafted (→ ADR 0014, WP-12…15) | same open question; builds on `hawkes-processes.md` |
