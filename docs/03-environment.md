# Environment & Package Management

This project uses **[uv](https://docs.astral.sh/uv/)** for Python interpreter, environment, and dependency management. All Python work goes through uv. Do not invoke system `python` or `pip` directly — that bypasses the locked environment and breaks reproducibility (see `01-scientific-rigor.md` §6).

## One-time setup (new contributor or agent)

1. **Install uv** (if you don't have it). Official installer:
  ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
   See [the docs](https://docs.astral.sh/uv/getting-started/installation/) for alternatives.
2. **Sync the environment** from the repo root:
  ```sh
   uv sync
  ```
   This will:
  - Install the Python version pinned in `.python-version` (currently 3.12) if not present.
  - Create `.venv/` (gitignored) and install everything from `uv.lock`.
3. **Verify**:
  ```sh
   uv run python -c "import sys; print(sys.version)"
  ```

You do **not** need to manually activate the venv. `uv run` handles that.

## Daily commands


| Task                                               | Command                                |
| -------------------------------------------------- | -------------------------------------- |
| Run a Python script                                | `uv run python path/to/script.py`      |
| Run a tool (e.g. pytest, jupyter)                  | `uv run pytest` / `uv run jupyter lab` |
| Add a runtime dependency                           | `uv add <pkg>`                         |
| Add a dev-only dependency                          | `uv add --dev <pkg>`                   |
| Remove a dependency                                | `uv remove <pkg>`                      |
| Sync env to lockfile                               | `uv sync`                              |
| Update lockfile after manual `pyproject.toml` edit | `uv lock`                              |
| Bump Python version                                | edit `.python-version`, then `uv sync` |
| Quick REPL                                         | `uv run python`                        |


## Conventions

- **All dependencies live in `pyproject.toml`.** Runtime under `[project.dependencies]`, dev tools under `[dependency-groups].dev`. Ad-hoc `pip install` packages vanish on next `uv sync` and are never reproducible.
- `**uv.lock` is committed.** It is part of the project state. Regenerate it (`uv lock`) only when intentionally bumping versions, and commit the resulting diff.
- `**.python-version` pins the interpreter.** Bump deliberately; do not let it drift.
- `**.venv/` is never committed.** Already in `.gitignore`.
- `**uv run` for every Python invocation.** Scripts, notebooks, ad-hoc REPLs — all go through uv. This is a non-negotiable rule (see `/AGENTS.md`).
- **Pin within a sensible range.** Prefer `uv add 'numpy>=2.0,<3'` style for libraries with known major-version churn. For most things, `uv add numpy` is fine and `uv.lock` does the pinning.
- **Optional / experimental dependency groups.** For deps tied to a specific work package (e.g., a JAX vs. PyTorch choice not yet made), use `[dependency-groups]`. Install on demand with `uv sync --group <name>`. Document in the relevant work package.
- **Pre-existing code in the repo predates this setup.** If you run something from `memes_base/` or any legacy script and hit `ModuleNotFoundError`, add the dep via `uv add` and log what you added (and why) in the current work package. Do not assume legacy code's deps are still the right ones.

## GPU / ML libraries

Local hardware: NVIDIA 3090. PyTorch and JAX both ship CUDA-bundled wheels on PyPI (`torch`, `jax[cuda12]`). For most cases, `uv add torch` or `uv add 'jax[cuda12]'` is sufficient. The concrete framework choice for modeling is deferred to a later ADR; whichever is chosen, install via `uv add`.

If a specific CUDA-version index is required, configure it in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
```

Document any index addition in this file's Log section below and explain why.

## Reproducibility checklist

Per `01-scientific-rigor.md` §6, every recorded experimental result should be reproducible from:

1. The git commit hash.
2. The committed `uv.lock`.
3. The recorded random seeds (in the experiment's config/log).
4. The recorded dataset identifier or checksum.

Do not bypass uv with `uv pip install` for anything you want kept. If you find yourself wanting to, stop and use `uv add`.

## Troubleshooting

- **"Why does `python script.py` fail but `uv run python script.py` work?"** You're hitting your system Python, which doesn't have the project deps. Always prefix with `uv run`.
- **"My VS Code linter doesn't see installed packages."** Point your editor at `.venv/bin/python` as the interpreter. uv creates that on `uv sync`.
- **"`uv sync` is slow on first run."** uv is downloading the Python toolchain and resolving the dep graph. Subsequent runs are fast.

## Log

- 2026-05-19 — Added WP-02 runtime dependencies via uv: `jax`, `diffrax`, `numpyro`, `corner`, `matplotlib`, `scipy`; added dev dependency `pytest`. `uv.lock` is now part of the reproducible environment.
- 2026-05-19 — Bootstrapped uv: `pyproject.toml` (empty deps), `.python-version` = 3.12. `uv.lock` will be created on first `uv sync`.

