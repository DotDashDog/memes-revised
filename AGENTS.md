# Meme Spread Project

A long-running research project: predict how memes spread on Reddit from their content
(embeddings → SIR-style curve summaries, plus a neural-Hawkes track). The goal is a
defensible answer to a scientific question, not a model that works at any cost.

## Where things are

`docs/` is the project record. Read what the task needs; nothing here has to be read
front to back.

| Path | What it is |
|---|---|
| `docs/00-background.md` | What we are trying to do and why |
| `docs/02-state.md` | Current snapshot, and the list of things waiting on the user. Start here |
| `docs/changelog.md` | History, newest first |
| `docs/work-packages/` | Units of work (`WP-NN`), with status table and per-package logs |
| `docs/decisions/` | Design decisions (ADRs) and their status |
| `docs/open-questions.md` | Unresolved questions |
| `docs/01-scientific-rigor.md`, `docs/method-discipline.md` | The experimental standards for this project. Read both before running or evaluating an experiment |
| `docs/03-environment.md` | uv, GPU/WSL2 setup, measured storage characteristics |
| `docs/research-notes/` | Background research; informative, not binding |
| `experiments/wpNN/` (lowercase) | Experiment **code**, tracked |
| `experiments/WP-NN/` (uppercase) | Experiment **output**, mostly gitignored. A different directory from the one above |
| `memes_base/`, loose scripts at the repo root | The 2023–2024 iteration. Read before reusing; several bugs have been found in it |
| `rigor-port/` | A template for exporting these standards to other projects. Not in force here |

## How we work

The user has little time for this project and supervises it lightly. Agents are trusted to
run it day to day, which makes two things matter more than they otherwise would: the science
has to be checked by someone other than its author, and what reaches the user has to be
readable cold.

### Who decides what

The user decides anything project-critical:

- the research question, the success metric, the evaluation set and train/test split;
- the model family, the dataset, and other choices recorded as ADRs;
- anything that would invalidate or reinterpret results already in the record;
- anything that costs a lot: roughly, more than a couple of hours of machine time, tens of
  GB of download or disk, any paid API spend, or a meaningful amount of the user's own time;
- anything hard to undo (deleting data, merging to `main`, publishing).

Everything else is yours: implementation, tooling, experiment details, thresholds you can
derive, how to sequence the work. Decide, and note the reasoning in the work-package log
where a later reader would otherwise wonder.

When a decision is the user's, add it to "Waiting on the user" in `docs/02-state.md` with
the options, the evidence, your recommendation, and what it blocks. For a design decision
also draft the ADR as `proposed`. Then carry on with whatever does not depend on the answer.
A whole session spent waiting is the outcome to avoid; so is a decision quietly made
because asking was inconvenient.

### Checking results

Whoever produced a result is the worst-placed to grade it. A false "pass" has been recorded
on this project before (the WP-12 bridge check: an arbitrary threshold on a metric that hid
a disagreement the plot showed plainly). The standards in `docs/01-scientific-rigor.md` and
`docs/method-discipline.md` exist because of failures like that one, and reduced
supervision is the situation they were written for.

- Before a run whose result will enter the record, write in the work-package log what you
  expect to see, what would count as failure, and why that criterion tests the claim.
- Afterwards, have the result checked by a reviewer who did not produce it: a fresh agent
  given the claim, the criterion, the plots, and the raw numbers, and asked to find what is
  wrong. Give it the artifacts, not your conclusion.
- Record the result with who has checked it: `UNCERTIFIED` (author only),
  `agent-reviewed`, or `user-reviewed`. Results that settle an ADR or that the project's
  headline claim rests on need the user; the rest can stand as `agent-reviewed`.
- A number that passes while the plot disagrees is an open finding. So is a reviewer's
  objection you could not resolve. Report both as they are.

### Reporting to the user

The user reads updates between other work, months of context away from the details.

- Lead with what was found and what, if anything, is needed from them.
- Write for someone who has not seen the internal labels: say "the clustering audit" and
  what it is for, with the `WP-04` tag only as a reference. Define any term the project
  coined. Give numbers with units and with what they are being compared to.
- Show a plot wherever a result has a shape. Say what the plot would look like if things
  were fine, and what it actually shows.
- State how much confidence the result deserves and who has checked it.

### Housekeeping

- **Keep the record current.** When work lands, update the work package's status and log,
  the snapshot in `docs/02-state.md` if it changed, and add a line or two to
  `docs/changelog.md`. Record negative results and dead ends too. Questions you cannot
  settle go in `docs/open-questions.md`.
- **Python goes through uv.** `uv run` for every invocation, `uv add` / `uv remove` for
  dependencies, so that results stay reproducible from the lockfile.
- **`.env` holds `OPENAI_API_KEY`.** Keep its contents out of output, files, and commits.
